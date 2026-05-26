import json
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

CLEANED_ROOT = Path("cleaned_data")
EMBEDDINGS_ROOT = Path("embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# LOAD EMBEDDER
# ============================================================

print(f"Loading embedder: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ============================================================
# HELPERS
# ============================================================

def extract_attempt_reasonings(data):
    """
    Extract one concatenated assistant reasoning string
    per attempt.
    """

    attempt_texts = []

    messages = data.get("messages", [])

    for attempt in messages:

        assistant_parts = []

        for msg in attempt:

            if msg.get("role") == "assistant":

                content = msg.get("content", "")

                if content:
                    assistant_parts.append(content)

        full_reasoning = "\n\n".join(assistant_parts).strip()

        attempt_texts.append(full_reasoning)

    return attempt_texts


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# FIND ALL JSON FILES
# ============================================================

json_files = sorted(
    CLEANED_ROOT.rglob("*.json")
)

print(f"Found {len(json_files)} JSON files")

# ============================================================
# MAIN LOOP
# ============================================================

for json_path in tqdm(json_files):

    try:

        # ----------------------------------------------------
        # READ JSON
        # ----------------------------------------------------

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ----------------------------------------------------
        # EXTRACT REASONINGS
        # ----------------------------------------------------

        attempt_texts = extract_attempt_reasonings(data)

        if len(attempt_texts) == 0:
            print(f"Skipping empty file: {json_path}")
            continue

        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        embeddings = model.encode(
            attempt_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # ----------------------------------------------------
        # MIRROR DIRECTORY STRUCTURE
        # ----------------------------------------------------

        relative_path = json_path.relative_to(
            CLEANED_ROOT
        )

        output_path = (
            EMBEDDINGS_ROOT /
            relative_path.with_suffix(".npy")
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # SAVE EMBEDDINGS
        # ----------------------------------------------------

        np.save(output_path, embeddings)

        # ----------------------------------------------------
        # SAVE METADATA
        # ----------------------------------------------------

        parts = relative_path.parts

        benchmark = "/".join(parts[:-2])
        model_id = parts[-2]
        question_idx = json_path.stem

        meta = {
            "source_path": str(json_path),
            "embedding_path": str(output_path),
            "benchmark": benchmark,
            "model_id": model_id,
            "question_idx": question_idx,
            "embedder": MODEL_NAME,
            "embedding_shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "normalized": True,
            "attempts": [],
        }

        for i, text in enumerate(attempt_texts):

            meta["attempts"].append({
                "attempt_idx": i,
                "text_length": len(text),
                "sha256": sha256_text(text),
            })

        meta_path = output_path.with_suffix(
            ".meta.json"
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    except Exception as e:

        print(f"\nERROR: {json_path}")
        print(e)

print("\nDone.")