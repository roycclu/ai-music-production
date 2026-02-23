"""
Arize Phoenix observability for the Music Video Pipeline.

Connects to the running Phoenix server at http://localhost:6006 and routes
all traces to the 'music-video-pipeline' project.  Does NOT launch Phoenix —
assumes it is already running (systemd user service).
"""

import sys

PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
PHOENIX_PROJECT  = "music-video-pipeline"


def setup_observability() -> bool:
    """Register OTLP exporter and instrument the Anthropic SDK.

    Returns True on success, False if a required package is missing.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource = Resource({"phoenix.project.name": PHOENIX_PROJECT})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _instrument_anthropic(provider)

        print(
            f"[Observability] Tracing → Phoenix project '{PHOENIX_PROJECT}' "
            f"at http://localhost:6006",
            file=sys.stderr,
        )
        return True

    except ImportError as exc:
        print(
            f"[Observability] Missing package ({exc}) — tracing disabled. "
            "Run: pip install opentelemetry-exporter-otlp-proto-http",
            file=sys.stderr,
        )
        return False
    except Exception as exc:
        print(f"[Observability] Setup failed: {exc}", file=sys.stderr)
        return False


def _instrument_anthropic(provider) -> None:
    """Instrument the Anthropic SDK (sync + async) if available."""
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument(tracer_provider=provider)
    except ImportError:
        pass
