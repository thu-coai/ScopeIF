"""Convert ScopeInstruct into a verl training parquet, one per reward variant.

    python3 scopeif_process.py --name {scopeif,scopeif_wo_hra,rl_ila,rl_cla}
"""

import argparse
import json
import os

import datasets

REWARD_VARIANTS = ("scopeif", "scopeif_wo_hra", "rl_ila", "rl_cla")


def load_dataset(data_paths):
    data = []
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for i, item in enumerate(items):
            row = dict(item)
            row["prompt_id"] = i
            data.append(row)
    return data


def get_checklist(constraints):
    return [constraint["constraint"] for constraint in constraints]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="../ScopeInstruct",
        help="Directory holding the ScopeInstruct json and receiving the parquet.",
    )
    parser.add_argument("--file_name", default="train.json")
    parser.add_argument("--name", default="scopeif", choices=REWARD_VARIANTS)
    args = parser.parse_args()

    data_list = load_dataset([os.path.join(args.local_dir, args.file_name)])
    dataset = datasets.Dataset.from_list(data_list)

    def process_fn(example, idx):
        prompt = example.pop("prompt")
        constraints = example.pop("constraints")
        targets = example.pop("targets")
        language = example.pop("language", "en")
        prompt_id = example.pop("prompt_id")
        avg_reasoning_length = example.pop("avg_reasoning_length", None)

        ground_truth = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "constraints": constraints,
            "checklist": get_checklist(constraints),
            "targets": targets,
            "language": language,
        }
        if avg_reasoning_length is not None:
            ground_truth["avg_reasoning_length"] = avg_reasoning_length
            ground_truth["reasoning_length_unit"] = "tokens"

        return {
            "data_source": args.name,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": args.name,
            "reward_model": {
                "style": "rm",
                "ground_truth": json.dumps(ground_truth),
            },
            "extra_info": {
                "split": "train",
                "index": idx,
            },
        }

    train_dataset = dataset.map(function=process_fn, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, f"train_{args.name}.parquet"))


if __name__ == "__main__":
    main()
