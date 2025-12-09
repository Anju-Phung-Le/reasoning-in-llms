from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import json
from .prompts import build_prompt

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def predict(model_name: str, data_fp: str, out_fp: str, max_new_tokens: int = 2):
    """
    Predicts using seq2seq model from (e.g. flan-t5 model.

    Args:
        model_name (str): Model id from Hugging Face (e.g., 'google/flan-t5-small')
        data_fp (str): Path to processed dataset (dataset_v1.jsonl)
        out_fp (str): Path to save predictions (flan_t5_predictions.jsonl)
        max_new_tokens (int): Limit on output length (default: 2)
    """
     # load config to see what kind of model this is
    config = AutoConfig.from_pretrained(model_name)

    tok = AutoTokenizer.from_pretrained(model_name)

    if getattr(config, "is_encoder_decoder", False):
        # T5 / Flan / other seq2seq
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        # decoder-only LMs like Mistral / LLaMA
        mdl = AutoModelForCausalLM.from_pretrained(model_name)   

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

    with open(data_fp) as fin, open(out_fp, "w") as fout:
        for line in fin:
            item = json.loads(line)
            prompt = build_prompt(item)

            # Encode text for the model to read
            enc = tok(prompt, return_tensors="pt")

            # Generate prediction
            out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)

            # Decode prediction back to natural text
            text = tok.decode(out[0], skip_special_tokens=True).strip()

            # Extract predicted letter (A, B, C)
            letter = text[:1].upper()
            idx = LETTERS.find(letter)
            pred_index = idx if 0 <= idx < len(item["options"]) else -1

            fout.write(json.dumps({
                "id": item["id"],
                "pred_text": text,
                "pred_letter": letter,
                "pred_index": pred_index
            }) + "\n")

    print(f"Saved predictions to {out_fp}")