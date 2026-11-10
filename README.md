# reasoning-in-llms
Repository for master thesis about reasoning in Large Language Models. Human reasoning vs. Logical reasoning.
Prerequisites: transformers needs to be installed
# Generate seeds
python -m src.cli gen \
  --n 150 \
  --templates configs/syllogism_templates.json \
  --domains configs/domains.json \
  --out data/raw/seed.jsonl

# Expand
python -m src.cli expand \
  --in data/raw/seed.jsonl \
  --out data/processed/dataset_v1.jsonl

# Predict
python -m src.cli predict \
  --model google/flan-t5-base \
  --data data/processed/dataset_v1.jsonl \
  --out data/predictions/flan_t5_base2_predictions.jsonl

# Eval
python -m src.cli eval \
  --gold data/processed/dataset_v1.jsonl \
  --pred data/predictions/flan_t5_base2_predictions.jsonl \
  --out outputs/flan_t5_base2_outputs_log.csv

# Reproduce the thesis tables and figures
The final results use the full dataset (`data/processed/dataset_all_forms.jsonl`) and the saved model
predictions, so no model needs to be run. Requires Python 3.10 or newer and matplotlib
(`pip install matplotlib`).

Inputs, all included in the repository:
- `data/predictions/final/`: chain-of-thought predictions for seven models, five sampled runs each
- `data/predictions/all_forms/`: forced-choice predictions for the three FLAN-T5 models
- `outputs/final/`: evaluation logs for the chain-of-thought runs, used as a cross-check

Run the scripts from the repository root, in this order. The first one writes a CSV that the others
check their numbers against.

python scripts/final2_disagreement_table.py
python scripts/final2_gold_tables.py
python scripts/final2_neither_alignment.py
python scripts/final2_cd_tables.py
python scripts/final2_alignment_by_form.py
python scripts/final2_role_domain_alignment.py
python scripts/final2_response_bias.py

The last script shuffles each run 5,000 times and takes a few minutes. Tables and figures are written to
`tables/final2_*`, and the detailed numbers to `outputs/report/final2_*.csv`.
