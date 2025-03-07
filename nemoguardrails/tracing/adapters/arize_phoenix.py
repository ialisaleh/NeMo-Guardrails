# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nemoguardrails.tracing import InteractionLog

# Attempt to import OpenTelemetry modules, raising an error if not installed.
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Attributes
except ImportError:
    raise ImportError(
        "Could not import OpenTelemetry. Install it with: `pip install opentelemetry-api opentelemetry-sdk`."
    )

# Attempt to import Arize-Phoenix OTEL dependencies, raising an error if missing.
try:
    from openinference.semconv.trace import SpanAttributes
    from phoenix.otel import (
        PROJECT_NAME,
        BatchSpanProcessor,
        GRPCSpanExporter,
        HTTPSpanExporter,
        Resource,
        SimpleSpanProcessor,
        TracerProvider,
    )
except ImportError:
    raise ImportError(
        "Could not import arize-phoenix-otel. Install it with: `pip install arize-phoenix-otel`."
    )

# Attempt to import LangChain Instrumentor for OpenInference instrumentation.
try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
except ImportError:
    raise ImportError(
        "Could not import LangChainInstrumentor. Install it with: `pip install openinference-instrumentation-langchain`."
    )

from nemoguardrails.tracing.adapters.base import InteractionLogAdapter


class ArizePhoenixAdapter(InteractionLogAdapter):
    """
    Adapter for integrating OpenTelemetry tracing with Arize Phoenix.
    This class sets up OpenTelemetry tracing for interaction logs using
    Arize Phoenix or OpenInference instrumentation.
    """

    name = "ArizePhoenix"

    def __init__(
        self,
        project_name="nemo_guardrails",
        arize_phoenix_batch: Optional[str] = None,
        resource_attributes: Optional[Attributes] = None,
        **kwargs,
    ):
        """
        Initializes the tracing adapter for Arize Phoenix.

        Args:
            project_name (str): Name of the project used for tracing metadata.
            arize_phoenix_batch (Optional[str]): Enables batch processing if set to truthy values.
            resource_attributes (Optional[Attributes]): Additional resource attributes for OpenTelemetry tracing.
        """
        # Convert batch processing flag to a boolean.
        arize_phoenix_batch = arize_phoenix_batch in {"true", "t", "yes", "y", "1"}
        resource_attributes = resource_attributes or {}

        # Fetch Arize-related environment variables.
        arize_api_key = os.getenv("ARIZE_API_KEY")
        arize_space_id = os.getenv("ARIZE_SPACE_ID")
        arize_collector_endpoint = os.getenv(
            "ARIZE_COLLECTOR_ENDPOINT", "https://otlp.arize.com"
        )
        enable_arize_tracing = bool(arize_api_key and arize_space_id)
        arize_endpoint = f"{arize_collector_endpoint}/v1"
        arize_headers = {
            "api_key": arize_api_key,
            "space_id": arize_space_id,
            "authorization": f"Bearer {arize_api_key}",
        }

        # Fetch Phoenix-related environment variables.
        phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        phoenix_collector_endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"
        )
        enable_phoenix_tracing = bool(phoenix_api_key)
        phoenix_endpoint = f"{phoenix_collector_endpoint}/v1/traces"
        phoenix_headers = {
            "api_key": phoenix_api_key,
            "authorization": f"Bearer {phoenix_api_key}",
        }

        # If neither Arize nor Phoenix tracing is enabled, exit initialization.
        if not (enable_arize_tracing or enable_phoenix_tracing):
            return False

        # Configure OpenTelemetry resource attributes.
        attributes = {
            PROJECT_NAME: project_name,
            "model_id": project_name,
            **resource_attributes,
        }
        resource = Resource.create(attributes=attributes)
        tracer_provider = TracerProvider(resource=resource, verbose=False)
        span_processor = (
            BatchSpanProcessor if arize_phoenix_batch else SimpleSpanProcessor
        )

        # Configure Arize tracing if enabled.
        if enable_arize_tracing:
            tracer_provider.add_span_processor(
                span_processor(
                    span_exporter=GRPCSpanExporter(
                        endpoint=arize_endpoint, headers=arize_headers
                    )
                )
            )

        # Configure Phoenix tracing if enabled.
        if enable_phoenix_tracing:
            tracer_provider.add_span_processor(
                span_processor(
                    span_exporter=HTTPSpanExporter(
                        endpoint=phoenix_endpoint, headers=phoenix_headers
                    )
                )
            )

        # Set the tracer provider for OpenTelemetry.
        trace.set_tracer_provider(tracer_provider)
        self.tracer_provider = tracer_provider
        self.tracer = trace.get_tracer(__name__)

        # Instrument LangChain interactions.
        LangChainInstrumentor().instrument(
            tracer_provider=self.tracer_provider, skip_dep_check=True
        )

    def transform(self, interaction_log: "InteractionLog"):
        """
        Transforms an InteractionLog into OpenTelemetry spans.

        Args:
            interaction_log (InteractionLog): The interaction log containing tracing information.
        """
        spans = {}
        for span_data in interaction_log.trace:
            parent_span = spans.get(span_data.parent_id)
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )
            self._create_span(span_data, parent_context, spans, interaction_log.id)

    async def transform_async(self, interaction_log: "InteractionLog"):
        """
        Asynchronous version of `transform` to create OpenTelemetry spans.

        Args:
            interaction_log (InteractionLog): The interaction log containing tracing information.
        """
        spans = {}
        for span_data in interaction_log.trace:
            parent_span = spans.get(span_data.parent_id)
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )
            self._create_span(span_data, parent_context, spans, interaction_log.id)

    def _create_span(self, span_data, parent_context, spans, trace_id):
        """
        Creates an OpenTelemetry span from the given span data.

        Args:
            span_data: The data representing the span.
            parent_context: The parent span context (if available).
            spans: Dictionary to store created spans.
            trace_id: ID of the overall trace.
        """
        with self.tracer.start_as_current_span(
            span_data.name, context=parent_context
        ) as span:
            for key, value in span_data.metrics.items():
                span.set_attribute(key, value)

            span.set_attribute("span_id", span_data.span_id)
            span.set_attribute("name", span_data.name)
            span.set_attribute("parent_id", span_data.parent_id)
            span.set_attribute("resource_id", span_data.resource_id)
            span.set_attribute("start_time", span_data.start_time)
            span.set_attribute("end_time", span_data.end_time)
            span.set_attribute("duration", span_data.duration)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "GUARDRAIL")
            span.set_status(trace.Status(trace.StatusCode.OK))
            spans[span_data.span_id] = span
