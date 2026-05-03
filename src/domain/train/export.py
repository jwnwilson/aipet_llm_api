"""GGUF export logic — converts a HuggingFace checkpoint to a quantised GGUF for RPi."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SETUP_TEMPLATE = """\
llama.cpp not found at {llama_cpp_dir}.

To set it up:
  1. Clone:  git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}
  2. Build:  cd {llama_cpp_dir} && cmake -B build && cmake --build build --config Release -j
  3. Verify: {convert_script} and {quantize_bin} exist

Then re-run.
"""


def _check_llama_cpp(llama_cpp_dir: Path) -> None:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    missing = [p for p in (convert_script, quantize_bin) if not p.exists()]
    if missing:
        print("ERROR: missing llama.cpp files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            _SETUP_TEMPLATE.format(
                llama_cpp_dir=llama_cpp_dir,
                convert_script=convert_script,
                quantize_bin=quantize_bin,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def _run(cmd: list[str], description: str) -> None:
    print(f"\n{description}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode}).", file=sys.stderr)
        sys.exit(1)


def export(
    checkpoint: Path,
    output: Path,
    quantize: str = "Q4_K_M",
    llama_cpp_dir: Path = Path("llama.cpp"),
) -> None:
    """Convert HF checkpoint → FP16 GGUF → quantised GGUF, then verify it loads."""
    _check_llama_cpp(llama_cpp_dir)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    f16_output = Path(str(output).replace(".gguf", "") + ".f16.gguf")
    output.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [sys.executable, str(convert_script), str(checkpoint), "--outfile", str(f16_output)],
        description=f"Converting HF checkpoint → FP16 GGUF: {f16_output}",
    )
    _run(
        [str(quantize_bin), str(f16_output), str(output), quantize],
        description=f"Quantising ({quantize}): {output}",
    )

    print(f"\nVerifying GGUF loads: {output} …")
    try:
        from infrastructure.inference import LlamaCppInferenceAdapter
        LlamaCppInferenceAdapter(model_path=str(output))
        print("  LlamaCppInferenceAdapter instantiated successfully (lazy load).")
    except ImportError as exc:
        print(f"WARNING: could not verify GGUF ({exc}).", file=sys.stderr)

    size_mb = output.stat().st_size / (1024 ** 2)
    print(f"\nExport complete.")
    print(f"  Output : {output}")
    print(f"  Size   : {size_mb:.1f} MB")
