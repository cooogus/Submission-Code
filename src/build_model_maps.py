import csv
import json
from pathlib import Path

VERIFICATION_CSVS = [
    Path("agieval_lsat_ar_verification.csv"),
    Path("aime_2025_verification.csv"),
    Path("gpqa_diamond_verification.csv"),
    Path("mmmu_account_verification.csv"),
    Path("mmmu_clinical_medicine_verification.csv"),
    Path("mmmu_finance_merged_verification.csv"),
]

IDENTITY_OUT = Path("model_identity_map.json")
FAMILY_OUT = Path("model_family_lookup.json")


def canonicalize_model_name(raw_name):
    name = raw_name.strip()

    mapping = {
        "claude-37-sonnet": "Claude 3.7 Sonnet",
        "claude-4-opus": "Claude 4 Opus",
        "claude-45-sonnet": "Claude 4.5 Sonnet",
        "deepseek_r1": "DeepSeek-R1",
        "deepseek-v31-terminus": "DeepSeek-v3.1-Terminus",
        "gpt-5": "GPT-5",
        "gpt-5-mini": "GPT-5-mini",
        "gpt-51": "GPT-5.1",
        "grok-4": "Grok 4",
        "grok-4-fast-reasoning": "Grok 4 Fast Reasoning",
        "gemini-pro-2.5": "Gemini 2.5 Pro",
        "gemini-3": "Gemini 3 Preview",
        "llama-4-maverick": "Llama-4-Maverick",
        "llama-4-scout": "Llama-4-Scout",
        "minimax_m2": "MiniMax M2",
        "o4-mini": "o4-mini",
        "oss-120b": "GPT OSS 120B",
        "qwen3_235b_a22b" : "Qwen3-235B",
        "qwen3_30b_a3b" : "Qwen3-30B"
    }

    return mapping.get(name, name)


def infer_family(canonical_name):
    lower = canonical_name.lower()

    if "claude" in lower:
        return "Anthropic"
    if "gpt" in lower or "o4" in lower or "oss" in lower:
        return "OpenAI"
    if "gemini" in lower:
        return "Google"
    if "grok" in lower:
        return "xAI"
    if "deepseek" in lower:
        return "DeepSeek"
    if "llama" in lower:
        return "Meta"
    if "minimax" in lower:
        return "MiniMax"
    if "qwen" in lower:
        return "Qwen"
    if "arcee" in lower:
        return "Arcee"

    return "Unknown"


raw_models = set()

for csv_path in VERIFICATION_CSVS:
    if not csv_path.exists():
        print(f"Missing CSV, skipping: {csv_path}")
        continue

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            model = row.get("model")
            if model:
                raw_models.add(model.strip())


model_identity_map = {}
model_family_lookup = {}

for raw_model in sorted(raw_models):
    canonical = canonicalize_model_name(raw_model)
    family = infer_family(canonical)

    model_identity_map[raw_model] = canonical
    model_family_lookup[canonical] = family


with open(IDENTITY_OUT, "w", encoding="utf-8") as f:
    json.dump(model_identity_map, f, indent=2, ensure_ascii=False)

with open(FAMILY_OUT, "w", encoding="utf-8") as f:
    json.dump(model_family_lookup, f, indent=2, ensure_ascii=False)


print(f"Saved: {IDENTITY_OUT}")
print(f"Saved: {FAMILY_OUT}")
print(f"Unique raw models found: {len(raw_models)}")

print("\nModel identity map:")
for raw, canonical in model_identity_map.items():
    print(f"{raw} -> {canonical}")

print("\nFamily lookup:")
for canonical, family in model_family_lookup.items():
    print(f"{canonical} -> {family}")