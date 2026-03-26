# reasoning-in-llms
Repository for master thesis about reasoning in Large Language Models. Human reasoning vs. Logical reasoning.
Prerquisities: transformer needs to be installed
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