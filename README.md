# ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling

This repository is the official implementation of ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling.

```
submit/
├── ScopeInstruct/       16,968 train / 1,000 test instructions
├── data_construction/   pipeline that builds ScopeInstruct
├── ScopeIF/             RL training (based on verl)
└── evaluation/          ScopeIF-Test harness
```

## ⚙️ Requirements

To install requirements:

```shell
pip install -r requirements.txt
```

## 📦 Data

ScopeInstruct is placed in `ScopeInstruct/train.json` and `ScopeInstruct/test.json`. The format of an instance is as follows:

- `id` (integer): A unique identifier for the instance.
- `prompt` (string): The instruction.
- `constraints` (list): The constraints contained in `prompt`. Each constraint has a `constraint` field (string) holding its text, and a `category` field describing it along four dimensions: `scope`, `primary_target`, `secondary_target` and `range`.
- `targets` (list): `targets[i]` holds the counting objects of `constraints[i]`, given as a JSON array of `{"counting_object", "type"}` in a fenced code block.

## 🔨 Data Construction

ScopeInstruct is built in three stages. Each stage has a generation script and an LLM-judge filter in its `constraint_validation/` subdirectory, reads `input/` and writes `output/`. Some examples of the inputs are placed in each stage's `input/`. You could preprocess your own data referring to these examples and pass it with `--input_file`.

The generation scripts call an OpenAI-compatible endpoint, while the filters serve a local judge model with vLLM:

```shell
export GENERATION_API_BASE_URL=<api_base_url>
export GENERATION_API_KEY=<api_key>
export GPT_OSS_MODEL_PATH=<path_to_gpt_oss_120b>
```

### Step 1: Atomic Constraint Generation

Run this command to generate the atomic constraints:

```shell
cd data_construction/atomic_constraint_generation
python atom_constraint_generation.py
cd constraint_validation
python validate_atom_constraints.py
```

### Step 2: Constraint Crossover

Run this command to crossover constraints, where `--constraint_file` takes the accepted output of Step 1:

```shell
cd data_construction/constraint_crossover
python crossover.py --constraint_file <path_to_step1_output>
cd constraint_validation
python validate_crossover_constraints.py
```

### Step 3: Target Extraction

Run this command to extract targets of each constraint:

```shell
cd data_construction/target_extraction
python extract_targets.py
```

## 🔥 Training

Our training codes are modified from [verl](https://github.com/volcengine/verl). First build the training parquet from `ScopeInstruct/`, where `--name` selects the reward variant (`scopeif`, `scopeif_wo_hra`, `rl_ila`, `rl_cla`) implemented in `verl/utils/reward_score/`:

```shell
cd ScopeIF
python3 examples/data_preprocess/scopeif_process.py --name scopeif
```

The reward queries a judge model served over an OpenAI-compatible endpoint, so deploy it first and point these variables at it:

```shell
export SCOPEIF_JUDGE_BASE_URL=http://<host>:8080/v1
export SCOPEIF_JUDGE_MODEL=<judge_model_name>
bash scripts/qwen3_4b_scopeif.sh
```

## 🚀 Evaluation

Evaluation on ScopeIF-Test comprises two steps:

- Generating the responses of the evaluated model for the test instructions.
- Scoring each constraint with a code-assisted LLM judge.

```shell
cd evaluation/ScopeIF-Test
python run_scopeif.py --stage generate --model_path <path_to_policy> --model_name <name>
python run_scopeif.py --stage score --judge_model_path <path_to_gpt_oss_120b> --model_name <name>
```

Results land in `results/<name>/`. The `responses.jsonl` file holds the generations, `responses_scored.json` holds the per-constraint judgements, and `metrics.json` holds the ISR and CSR scores along with the per-category breakdown.
