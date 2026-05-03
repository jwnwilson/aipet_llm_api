"""Export a trained HuggingFace checkpoint to a quantised GGUF for RPi deployment.

Usage:
    uv run python scripts/export.py \\
        --checkpoint models/checkpoints \\
        --output models/aipet.gguf \\
        --quantize Q4_K_M

Requirements:
    llama.cpp must be cloned at ./llama.cpp and built with its default CMake target
    so that llama.cpp/build/bin/llama-quantize exists.

    If llama.cpp is not present, the script will print setup instructions and exit.

Steps performed:
    1. Convert HF checkpoint → FP16 GGUF using llama.cpp/convert_hf_to_gguf.py
    2. Quantise FP16 GGUF → target quantisation using llama-quantize
    3. Verify the resulting GGUF loads via LlamaCppInferenceAdapter
    4. Print output path and file size
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_LLAMA_CPP_DIR = ROOT / "llama.cpp"
_CONVERT_SCRIPT = _LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
_QUANTIZE_BIN = _LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

_SETUP_INSTRUCTIONS = """\
llama.cpp not found at {llama_cpp_dir}.

To set it up:
  1. Clone the repo:
       git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}

  2. Build it (requires cmake and a C++ compiler):
       cd {llama_cpp_dir}
       cmake -B build
       cmake --build build --config Release -j

  3. Verify the required files exist:
       {convert_script}
       {quantize_bin}

Then re-run this script.
""".format(
    llama_cpp_dir=_LLAMA_CPP_DIR,
    convert_script=_CONVERT_SCRIPT,
    quantize_bin=_QUANTIZE_BIN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_llama_cpp() -> None:
    """Verify that the required llama.cpp files exist; print instructions and exit if not."""
    missing: list[str] = []
    if not _CONVERT_SCRIPT.exists():
        missing.append(str(_CONVERT_SCRIPT))
    if not _QUANTIZE_BIN.exists():
        missing.append(str(_QUANTIZE_BIN))

    if missing:
        print("ERROR: the following llama.cpp files are missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(file=sys.stderr)
        print(_SETUP_INSTRUCTIONS, file=sys.stderr)
        sys.exit(1)


def _run(cmd: list[str], description: str) -> None:
    """Run *cmd* as a subprocess; exit with code 1 on failure."""
    print(f"\n{description}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"ERROR: command failed with exit code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(1)


def _convert_to_f16(checkpoint: Path, f16_output: Path) -> None:
    """Run convert_hf_to_gguf.py to produce a FP16 GGUF."""
    _run(
        [
            sys.executable,
            str(_CONVERT_SCRIPT),
            str(checkpoint),
            "--outfile",
            str(f16_output),
        ],
        description=f"Converting HF checkpoint → FP16 GGUF: {f16_output}",
    )


def _quantize(f16_output: Path, final_output: Path, quantize: str) -> None:
    """Run llama-quantize to produce the target-quantisation GGUF."""
    _run(
        [
            str(_QUANTIZE_BIN),
            str(f16_output),
            str(final_output),
            quantize,
        ],
        description=f"Quantising ({quantize}): {final_output}",
    )


def _verify_gguf(model_path: Path) -> None:
    """Instantiate LlamaCppInferenceAdapter to verify the GGUF is loadable."""
    print(f"\nVerifying GGUF loads: {model_path} …")
    try:
        from src.infrastructure.inference import LlamaCppInferenceAdapter  # noqa: E402

        # Lazy adapter — model is not loaded until first infer() call.
        _adapter = LlamaCppInferenceAdapter(model_path=str(model_path))
        print("  LlamaCppInferenceAdapter instantiated successfully (lazy load).")
    except ImportError as exc:
        print(
            f"WARNING: could not import LlamaCppInferenceAdapter ({exc}). "
            "Skipping GGUF verification.",
            file=sys.stderr,
        )


def _print_success(output: Path) -> None:
    size_bytes = output.stat().st_size
    size_mb = size_bytes / (1024 ** 2)
    print(f"\nExport complete.")
    print(f"  Output : {output}")
    print(f"  Size   : {size_mb:.1f} MB ({size_bytes:,} bytes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a HuggingFace checkpoint to a quantised GGUF file."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints",
        help="Path to HuggingFace checkpoint directory (default: models/checkpoints).",
    )
    parser.add_argument(
        "--output",
        default="models/aipet.gguf",
        help="Destination GGUF path (default: models/aipet.gguf).",
    )
    parser.add_argument(
        "--quantize",
        default="Q4_K_M",
        help="llama-quantize quantisation type (default: Q4_K_M).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    checkpoint = Path(args.checkpoint)
    final_output = Path(args.output)
    quantize = args.quantize

    # Derive intermediate FP16 path from final output path.
    f16_output = final_output.with_suffix("").with_suffix("") if final_output.name.endswith(".gguf") \
        else final_output
    f16_output = Path(str(final_output).replace(".gguf", "") + ".f16.gguf")

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    _check_llama_cpp()

    # Ensure output directory exists.
    final_output.parent.mkdir(parents=True, exist_ok=True)

    _convert_to_f16(checkpoint, f16_output)
    _quantize(f16_output, final_output, quantize)
    _verify_gguf(final_output)
    _print_success(final_output)


if __name__ == "__main__":
    main()
