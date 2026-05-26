from pathlib import Path
import json
import re


ROOT_DIR = "cleaned/cleaned_data" # replace ROOT_DIR with the cleaned data path.
OUTPUT_FILE = "output.jsonl"


# Custom processing function - modify it
def process_item(data: dict) -> dict:
    # dummy return
    return {
        "benchmark": data["benchmark"],
        "benchmark_type": data["benchmark_type"],
        "model" : data["model"]
    }

# add the benchmark and model fields to the item 
def add_fields(item: dict, path: str) -> dict:
    parts = re.findall(r"[^/]+", path)
    
    item["benchmark_type"] = parts[-3]
    item["model"] = parts[-2]

    if(parts[-3].startswith("mmmu")):
        item["benchmark"] = "mmmu"
    if(parts[-3].startswith("gpqa")):
        item["benchmark"] = "gpqa"
    if(parts[-3].startswith("lsat")):
        item["benchmark"] = "agieval"
    if(parts[-3].startswith("aime")):
        item["benchmark"] = "aime"

    return item

# run the full pipeline

def run_pipeline():

    for path in Path(ROOT_DIR).rglob("*.json"):
        if not path.is_file():
            continue

        try:
            # load json
            item = json.loads(path.read_text(encoding="utf-8"))
            item = add_fields(item, str(path))

            # process
            result = process_item(item)

            # write output (jsonl)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            print(f"[ERROR] {path}: {repr(e)}")

if __name__ == "__main__":
    run_pipeline()



