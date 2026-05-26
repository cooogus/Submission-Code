from pathlib import Path
import json
import re

from nltk.util import ngrams


ROOT_DIR = "cleaned/cleaned_data" # replace ROOT_DIR with the cleaned data path.
OUTPUT_FILE = "n_grams.jsonl"


def get_ngram_overlap(text1, text2, n=3):
    # Tokenize and create n-grams for both texts
    grams1 = set(ngrams(text1.lower().split(), n))
    grams2 = set(ngrams(text2.lower().split(), n))
    
    # Compute intersection (the overlap)
    overlap = grams1.intersection(grams2)
    
    return list(overlap), len(overlap)

def get_type(data, msg_type):
    for d in data:

        if("type" in d and d["type"] == msg_type):
            return d
    return {}


# compute n-gram overlap for all the cot pairs and response pairs
def process_item(data: dict):
    messages = data["messages"]

    results = []

    for i, m1 in enumerate(messages):
        for j, m2 in enumerate(messages):

            cot_result = {}
            response_result = {}

            if(i <= j):
                continue

            # cot n gram

            cot1 = get_type(m1, "cot")
            cot2 = get_type(m2, "cot")

            if(cot1 == {} or cot2 == {}):
                # skip the entries when any of the messages are missing cot.
                print(f"Missing cot in idx {data["idx"]}, {data["benchmark"]}, {data["benchmark_type"]}, {data["model"]}")
                return []

            cot_overlap, cot_len_overlap = get_ngram_overlap(cot1["content"], cot2["content"])

            cot_result["idx"] = data["idx"]
            cot_result["message_index_pairs"] = (i,j)
            cot_result["len_overlap"] = cot_len_overlap
            cot_result["num_grams"] = 3
            cot_result["type"] =  "cot"
            cot_result["benchmark"] =  data["benchmark"]
            cot_result["benchmark_type"] = data["benchmark_type"]
            cot_result["model"] = data["model"]
            cot_result["overlap"] = cot_overlap
            

          
            results.append(cot_result)
           

    return results

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
import traceback

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

            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for r in result:
                    f.write(json.dumps(r,  ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[ERROR] {path}: {repr(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()



