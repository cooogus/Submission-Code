import json
import csv
import re
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

CLEANED_ROOT = Path("cleaned_data")

BENCHMARKS = [
    {
        "benchmark": "agieval_lsat_ar",
        "root": CLEANED_ROOT / "agieval" / "lsat_ar",
    },
    {
        "benchmark": "aime_2025",
        "root": CLEANED_ROOT / "aime" / "aime_2025",
    },
    {
        "benchmark": "gpqa_diamond",
        "root": CLEANED_ROOT / "gpqa" / "gpqa_diamond",
    },
    {
        "benchmark": "mmmu_accounting",
        "root": CLEANED_ROOT / "mmmu" / "mmmu_accounting",
    },
    {
        "benchmark": "mmmu_clinical_medicine",
        "root": CLEANED_ROOT / "mmmu" / "mmmu_clinical_medicine",
    },
    {
        "benchmark": "mmmu_finance_merged",
        "root": CLEANED_ROOT / "mmmu" / "mmmu_finance_merged",
    },
]

OUTPUT_CSV = Path("rpc_s_structural_scores.csv")

FIELDNAMES = [
    "benchmark",
    "model",
    "question_id",
    "json_path",
    "rpc_s",
    "pair_1_2",
    "pair_1_3",
    "pair_1_4",
    "pair_2_3",
    "pair_2_4",
    "pair_3_4",
    "seq_1",
    "seq_2",
    "seq_3",
    "seq_4",
    "correct",
    "answers",
    "gold_answer",
]

STEP_PATTERNS = {
    "define": [
        r"\blet\b",
        r"\bdenote\b",
        r"\bdefine\b",
        r"\bset\b",
        r"\bcall\b",
        r"\bassign\b",
        r"\bvariables?\b",
        r"\bconstraints?\b",
        r"\bgiven\b",
    ],
    "calculate": [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\bsimplif",
        r"\bevaluate\b",
        r"\bcount\b",
        r"\bsum\b",
        r"\bsubtract\b",
        r"\bmultiply\b",
        r"\bdivide\b",
        r"\bmod\b",
        r"\bmodulo\b",
        r"\bcase\b",
        r"\bif\b",
        r"\bwhen\b",
        r"\d+\s*[+\-*/=]\s*\d+",
        r"=",
        r"\$.*?\$",
    ],
    "verify": [
        r"\bcheck\b",
        r"\bverify\b",
        r"\bconfirm\b",
        r"\bdouble-check\b",
        r"\brecheck\b",
        r"\bvalidate\b",
        r"\bmake sure\b",
        r"\bworks\b",
        r"\bsatisfies\b",
        r"\bimpossible\b",
        r"\bnot allowed\b",
    ],
    "conclude": [
        r"\btherefore\b",
        r"\bthus\b",
        r"\bso\b",
        r"\bhence\b",
        r"\banswer\b",
        r"\bfinal\b",
        r"\bboxed\b",
        r"\bconclude\b",
        r"\bresult\b",
        r"\bremainder\b",
    ],
}


def load_existing_json_paths():
    """
    Prevents overwriting/recomputing rows already saved.
    Uses json_path as the unique key.
    """
    existing = set()

    if not OUTPUT_CSV.exists():
        return existing

    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            json_path = row.get("json_path")
            if json_path:
                existing.add(json_path)

    return existing


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_cot_from_attempt(attempt_messages):
    if not isinstance(attempt_messages, list):
        return ""

    for msg in attempt_messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("type") == "cot"
        ):
            return msg.get("content", "")

    return ""


def split_into_units(text):
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    units = []

    for para in paragraphs:
        pieces = re.split(r"(?<=[.!?])\s+", para)

        for piece in pieces:
            piece = piece.strip()
            if len(piece) >= 10:
                units.append(piece)

    return units


def tag_unit(unit):
    lower = unit.lower()

    scores = {}

    for step_type, patterns in STEP_PATTERNS.items():
        score = 0

        for pattern in patterns:
            if re.search(pattern, lower):
                score += 1

        scores[step_type] = score

    best_step = max(scores, key=scores.get)

    if scores[best_step] == 0:
        return "other"

    return best_step


def reasoning_to_step_sequence(reasoning_text):
    units = split_into_units(reasoning_text)
    sequence = []

    for unit in units:
        tag = tag_unit(unit)

        if tag == "other":
            continue

        if not sequence or sequence[-1] != tag:
            sequence.append(tag)

    return sequence


def lcs_length(a, b):
    m = len(a)
    n = len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(
                    dp[i][j + 1],
                    dp[i + 1][j],
                )

    return dp[m][n]


def structural_similarity(seq_a, seq_b):
    if not seq_a and not seq_b:
        return None

    if not seq_a or not seq_b:
        return 0.0

    lcs = lcs_length(seq_a, seq_b)
    return lcs / max(len(seq_a), len(seq_b))


def compute_rpc_s_for_record(record):
    messages = record.get("messages", [])

    if len(messages) != 4:
        return None

    reasoning_paths = [
        extract_cot_from_attempt(attempt)
        for attempt in messages
    ]

    if any(not path.strip() for path in reasoning_paths):
        return None

    step_sequences = [
        reasoning_to_step_sequence(path)
        for path in reasoning_paths
    ]

    pair_scores = []

    for i, j in combinations(range(4), 2):
        score = structural_similarity(
            step_sequences[i],
            step_sequences[j]
        )

        if score is not None:
            pair_scores.append(score)

    if len(pair_scores) != 6:
        return None

    return {
        "rpc_s": sum(pair_scores) / len(pair_scores),
        "pair_scores": pair_scores,
        "step_sequences": step_sequences,
    }


def main():
    existing_paths = load_existing_json_paths()
    file_exists = OUTPUT_CSV.exists()

    rows_written = 0
    rows_skipped_existing = 0
    rows_skipped_invalid = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        for bench in BENCHMARKS:
            benchmark_name = bench["benchmark"]
            root = bench["root"]

            if not root.exists():
                print(f"Missing benchmark folder, skipping: {root}")
                continue

            print(f"\nProcessing {benchmark_name}")

            model_dirs = [
                d for d in root.iterdir()
                if d.is_dir()
            ]

            for model_dir in tqdm(
                model_dirs,
                desc=f"{benchmark_name} models"
            ):
                model_name = model_dir.name
                json_files = list(model_dir.rglob("*.json"))

                for json_file in tqdm(
                    json_files,
                    desc=model_name,
                    leave=False
                ):
                    json_path_str = str(json_file)

                    if json_path_str in existing_paths:
                        rows_skipped_existing += 1
                        continue

                    record = read_json(json_file)

                    if not isinstance(record, dict):
                        rows_skipped_invalid += 1
                        continue

                    result = compute_rpc_s_for_record(record)

                    if result is None:
                        rows_skipped_invalid += 1
                        continue

                    row = {
                        "benchmark": benchmark_name,
                        "model": model_name,
                        "question_id": record.get("idx"),
                        "json_path": json_path_str,
                        "rpc_s": result["rpc_s"],
                        "pair_1_2": result["pair_scores"][0],
                        "pair_1_3": result["pair_scores"][1],
                        "pair_1_4": result["pair_scores"][2],
                        "pair_2_3": result["pair_scores"][3],
                        "pair_2_4": result["pair_scores"][4],
                        "pair_3_4": result["pair_scores"][5],
                        "seq_1": " > ".join(result["step_sequences"][0]),
                        "seq_2": " > ".join(result["step_sequences"][1]),
                        "seq_3": " > ".join(result["step_sequences"][2]),
                        "seq_4": " > ".join(result["step_sequences"][3]),
                        "correct": json.dumps(record.get("correct")),
                        "answers": json.dumps(record.get("answers")),
                        "gold_answer": record.get("gold_answer"),
                    }

                    writer.writerow(row)
                    f.flush()

                    existing_paths.add(json_path_str)
                    rows_written += 1

    print(f"\nSaved/appended RPC-S scores to: {OUTPUT_CSV}")
    print(f"New rows written: {rows_written}")
    print(f"Skipped already-existing rows: {rows_skipped_existing}")
    print(f"Skipped invalid/incomplete records: {rows_skipped_invalid}")


if __name__ == "__main__":
    main()