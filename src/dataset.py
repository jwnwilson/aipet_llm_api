"""
Synthetic dataset builder.
Replace / extend generate_samples() with your real game logs.
"""
import random
from pathlib import Path

from datasets import Dataset

# ── sample game scenarios ─────────────────────────────────────────────────────
INTENTS = ["attack", "flee", "trade", "quest_hint", "idle_chat", "taunt"]

TEMPLATES = {
    "attack":     ["Die, adventurer!", "You dare challenge me?!", "En garde!"],
    "flee":       ["I must retreat!", "This fight is lost!", "Another day..."],
    "trade":      ["Care to barter?", "I have wares if you have coin.", "Fine goods here!"],
    "quest_hint": [
        "They say the old mine hides secrets.",
        "Follow the river north.",
        "Beware the cursed tower.",
    ],
    "idle_chat":  ["Lovely weather today.", "Have you heard the news?", "Stay safe out there."],
    "taunt":      ["Is that all you've got?", "Pathetic!", "My grandmother hits harder!"],
}


def generate_samples(n: int = 2000) -> list[dict]:
    """Generate (context, response) pairs for fine-tuning."""
    samples = []
    for _ in range(n):
        intent = random.choice(INTENTS)
        response = random.choice(TEMPLATES[intent])
        context = f"[NPC:{intent.upper()}] Player approaches."
        samples.append({"prompt": context, "response": response, "intent": intent})
    return samples


def build_dataset(n: int = 2000, save_path: Path | None = None) -> Dataset:
    samples = generate_samples(n)
    ds = Dataset.from_list(samples)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(save_path))
        print(f"Saved {len(ds)} samples → {save_path}")
    return ds


if __name__ == "__main__":
    build_dataset(2000, Path(__file__).parent.parent / "data" / "game_dataset.parquet")
