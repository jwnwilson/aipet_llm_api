"""GGUF export logic — converts a HuggingFace checkpoint to a quantised GGUF for RPi."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

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
        log.error(
            "missing llama.cpp files: %s\n%s",
            ", ".join(str(p) for p in missing),
            _SETUP_TEMPLATE.format(
                llama_cpp_dir=llama_cpp_dir,
                convert_script=convert_script,
                quantize_bin=quantize_bin,
            ),
        )
        sys.exit(1)


def _has_bnb_tensors(checkpoint: Path) -> bool:
    """Return True if the checkpoint's tensor files contain bitsandbytes quantization tensors."""
    import json as _json

    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        try:
            data = _json.loads(index.read_text())
            return any(".absmax" in k for k in data.get("weight_map", {}))
        except Exception:
            pass

    try:
        from safetensors import safe_open
        for sf in sorted(checkpoint.glob("*.safetensors"))[:1]:
            with safe_open(str(sf), framework="pt") as f:
                return any(".absmax" in k for k in f.keys())
    except ImportError:
        pass

    return False


def _strip_bnb_quantization(checkpoint: Path) -> None:
    """Ensure the checkpoint has no bitsandbytes artefacts before llama.cpp conversion.

    Some PEFT versions leave Linear4bit layers un-dequantized after
    merge_and_unload(), saving .absmax/.quant_map tensors alongside the packed
    4-bit weights.  convert_hf_to_gguf.py cannot map those tensor names and
    raises ValueError.

    When bitsandbytes tensors are detected we reload the model in float16
    (requires CUDA) and overwrite the checkpoint.  If CUDA is unavailable the
    function exits with a clear message; re-train with the current code, which
    explicitly dequantizes at merge time.
    """
    import json as _json

    config_file = checkpoint / "config.json"
    if not config_file.exists():
        return

    cfg = _json.loads(config_file.read_text())
    qc = cfg.get("quantization_config") or {}
    is_bnb_config = qc.get("quant_method") == "bitsandbytes" or qc.get("quant_type") in ("nf4", "fp4")

    if _has_bnb_tensors(checkpoint):
        # The tensor files still have packed 4-bit weights — need CUDA to dequantize.
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA")

            import bitsandbytes.nn as bnb_nn
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Checkpoint has bitsandbytes tensors — reloading in float16 (CUDA) …")
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint), device_map="auto", trust_remote_code=True
            )
            for name, module in list(model.named_modules()):
                if not isinstance(module, bnb_nn.Linear4bit):
                    continue
                parent_path, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_path) if parent_path else model
                with torch.no_grad():
                    w = module.weight.dequantize().to(torch.float16)
                linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=module.bias is not None, dtype=torch.float16)
                linear.weight = torch.nn.Parameter(w)
                if module.bias is not None:
                    linear.bias = torch.nn.Parameter(module.bias.to(torch.float16))
                setattr(parent, child_name, linear)

            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
            for f in list(checkpoint.glob("*.safetensors")) + list(checkpoint.glob("*.bin")) + list(checkpoint.glob("*.safetensors.index.json")):
                f.unlink()
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            log.info("Checkpoint resaved as float16.")

        except RuntimeError as exc:
            if "no CUDA" in str(exc) or "CUDA" in str(exc):
                log.error(
                    "Checkpoint contains bitsandbytes 4-bit tensors that cannot be converted "
                    "without CUDA. Options: (1) Re-train — current trainer dequantizes at merge time. "
                    "(2) Export from a CUDA machine."
                )
                sys.exit(1)
            raise

    if is_bnb_config:
        # Config still says bitsandbytes even though tensors are clean — strip it.
        cfg = _json.loads(config_file.read_text())
        cfg.pop("quantization_config", None)
        config_file.write_text(_json.dumps(cfg, indent=2))
        log.info("Stripped bitsandbytes quantization_config from %s", config_file)


def _run(cmd: list[str], description: str) -> None:
    log.info("%s  $ %s", description, " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("command failed (exit %d): %s", result.returncode, " ".join(cmd))
        sys.exit(1)


def export(
    checkpoint: Path,
    output: Path,
    quantize: str = "Q4_K_M",
    llama_cpp_dir: Path = Path("llama.cpp"),
) -> None:
    """Convert HF checkpoint → FP16 GGUF → quantised GGUF, then verify it loads."""
    _check_llama_cpp(llama_cpp_dir)
    _strip_bnb_quantization(checkpoint)

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

    log.info("Verifying GGUF loads: %s …", output)
    try:
        from adapters.inference import LlamaCppInferenceAdapter
        LlamaCppInferenceAdapter(model_path=str(output))
        log.info("LlamaCppInferenceAdapter instantiated successfully (lazy load).")
    except ImportError as exc:
        log.warning("could not verify GGUF: %s", exc)

    size_mb = output.stat().st_size / (1024 ** 2)
    log.info("Export complete  output=%s  size=%.1f MB", output, size_mb)
