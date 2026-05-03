"""CLI: run a single inference from a JSON request on stdin."""

from __future__ import annotations

import argparse
import json
import sys

from src.domain.models import InferenceRequest
from src.infrastructure.inference import LlamaCppInferenceAdapter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a single aipet inference from stdin JSON.")
    parser.add_argument("--model-path", default="models/aipet.gguf", dest="model_path")
    args = parser.parse_args(argv)

    try:
        request = InferenceRequest.model_validate(json.load(sys.stdin))
    except Exception as exc:
        print(f"ERROR: invalid request JSON — {exc}", file=sys.stderr)
        sys.exit(1)

    adapter = LlamaCppInferenceAdapter(model_path=args.model_path)
    print(adapter.infer(request).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
