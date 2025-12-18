import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM
from .prompts import build_prompt

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# tok.encode() returns a list of token ids(numbers)
def _single_token_id(tok, s: str):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def predict(model_name: str, data_fp: str, out_fp: str, max_new_tokens: int = 2):
    """
    Generate multiple-choice predictions (A/B/C) for a dataset and write them to JSONL.

    This function supports both encoder–decoder (seq2seq) models such as Flan-T5 models and
    decoder-only (causal) instruction-tuned models such as Mistral.

    Prediction methods
    1) Encoder–decoder models (config.is_encoder_decoder == True)
       - Builds a prompt for each example using `build_prompt(item)`.
       - Tokenizes the prompt and runs `model.generate(...)`.
       - Decodes the generated text and extracts the first character as the predicted letter.
       - Forces at least one generated token via `min_new_tokens=1` to reduce empty outputs.

    2) Decoder-only models (config.is_encoder_decoder == False)
       - Optionally wraps the prompt using the tokenizer chat template (if available),
         which is the foundation for instruction-tuned models.
       - Tokenizes the prompt and runs a forward pass (no generation).
       - Computes next-token logits and selects the best label among {A, B, C} by comparing
         the logits of the corresponding token ids (trying both "A" and " A" to handle
         tokenizer whitespace behavior).
       - Produces a single-letter output.
    """
    config = AutoConfig.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(config, "is_encoder_decoder", False):
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    mdl.eval()

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

    with open(data_fp) as fin, open(out_fp, "w") as fout, torch.inference_mode():
        for line in fin:
            item = json.loads(line)
            prompt = build_prompt(item)

            # if is_encoder_decoder: use generate()
            if getattr(config, "is_encoder_decoder", False):
                enc = tok(prompt, return_tensors="pt").to(device)

                out = mdl.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
                text = tok.decode(out[0], skip_special_tokens=True).strip()

                letter = text[:1].upper() if text else ""
                idx = LETTERS.find(letter) if letter else -1
                pred_index = idx if 0 <= idx < len(item["options"]) else -1

            # else: use next-token logits (chat-templates)
            else:
                if hasattr(tok, "apply_chat_template"):
                    prompt = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
                input_device = mdl.get_input_embeddings().weight.device
                enc = {k: v.to(input_device) for k, v in enc.items()}

                # Score A/B/C directly from next-token logits
                logits = mdl(**enc).logits[0, -1]  

                # Try both "A" and " A" to get token ids
                cand = {}
                for L in ["A", "B", "C"]:
                    tid = _single_token_id(tok, L)
                    if tid is None:
                        tid = _single_token_id(tok, " " + L)
                    if tid is not None:
                        cand[L] = tid

                if not cand:
                    # very unlikely, but fallback just in case
                    letter = ""
                    pred_index = -1
                    text = ""
                else:
                    scores = {L: logits[tid].item() for L, tid in cand.items()}
                    letter = max(scores, key=scores.get)
                    pred_index = "ABC".find(letter)
                    text = letter  # keep pred_text as single letter

            fout.write(json.dumps({
                "id": item["id"],
                "pred_text": text,
                "pred_letter": letter,
                "pred_index": pred_index
            }) + "\n")

    print(f"Saved predictions to {out_fp}")
