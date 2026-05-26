import csv
import shutil
from pathlib import Path

INPUT_CSVS = [
    {
        "csv_path": Path("agieval_lsat_ar_verification.csv"),
        "output_root": Path("cleaned_data/agieval/lsat_ar"),
    },
    {
        "csv_path": Path("aime_2025_verification.csv"),
        "output_root": Path("cleaned_data/aime/aime_2025"),
    },
    {
    "csv_path": Path("gpqa_diamond_verification.csv"),
    "output_root": Path("cleaned_data/gpqa/gpqa_diamond"),
    }, 
    {
        "csv_path": Path("mmmu_accounting_verification.csv"),
        "output_root": Path("cleaned_data/mmmu/mmmu_accounting"),
    },
    {
        "csv_path": Path("mmmu_clinical_medicine_verification.csv"),
        "output_root": Path("cleaned_data/mmmu/mmmu_clinical_medicine"),
    },
    {
        "csv_path": Path("mmmu_finance_merged_verification.csv"),
        "output_root": Path("cleaned_data/mmmu/mmmu_finance_merged"),
    }, 
]


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


for config in INPUT_CSVS:
    csv_path = config["csv_path"]
    output_root = config["output_root"]

    copied = 0
    skipped_no_reasoning = 0
    missing_files = 0

    print(f"\nProcessing: {csv_path}")

    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}")
        continue

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not is_true(row.get("has_reasoning")):
                skipped_no_reasoning += 1
                continue

            json_path = Path(row["json_path"])
            model_name = row["model"]

            if not json_path.exists():
                print(f"Missing JSON file: {json_path}")
                missing_files += 1
                continue

            output_dir = output_root / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / json_path.name

            shutil.copy2(json_path, output_path)
            copied += 1

    print(f"Copied JSONs with reasoning: {copied}")
    print(f"Skipped without reasoning: {skipped_no_reasoning}")
    print(f"Missing files: {missing_files}")