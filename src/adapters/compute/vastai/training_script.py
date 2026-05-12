"""Training entry point executed inside a Vast.ai instance.

Delegates to the shared S3-based training script — all config is read
from environment variables set by VastAiTrainingAdapter.submit().

Run as: python -m adapters.compute.vastai.training_script
"""
from adapters.compute.runpod.training_script import main

if __name__ == "__main__":
    main()
