import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM
from .prompts import build_prompt

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OPTIONS = ["Yes", "No", "Cannot determine"]

# Matches a <think>...</think> block (including newlines).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Fallback: any stray opening/closing think tags (e.g. output truncated).
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
# Matches a standalone A/B/C, optionally in parens or followed by ')' '.' ':'.
_ANSWER_RE = re.compile(r"\b([ABC])\b")

# tok.encode() returns a list of token ids(numbers)
def _single_token_id(tok, s: str):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _split_think(text: str):
    """
    Split a raw generation into (scratchpad, cleaned) where scratchpad is the
    concatenation of any <think>...</think> content and cleaned is everything
    else (with stray think tags removed). For non-reasoning models both are
    just ("", text).
    """
    scratchpad_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_TAG_RE.sub("", cleaned).strip()
    scratchpad = "\n".join(p.strip() for p in scratchpad_parts).strip()
    return scratchpad, cleaned


def _extract_letter(cleaned: str) -> str:
    """
    Return the model's final answer letter from a cleaned (post-<think>) output.
    Strategy: pick the LAST standalone A/B/C — this is robust to intros like
    'Let me consider A vs B... The final answer is C.'
    """
    matches = _ANSWER_RE.findall(cleaned.upper())
    return matches[-1] if matches else ""


def predict(model_name: str, data_fp: str, out_fp: str, max_new_tokens: int = 2,
            log_fp: str | None = None, cot: bool = False,
            cot_max_new_tokens: int = 256):
    """
    Generate multiple-choice predictions (A/B/C) for a dataset and write them to JSONL.

    This function supports both encoder–decoder (seq2seq) models such as Flan-T5 models and
    decoder-only (causal) instruction-tuned models such as Mistral.

    If `log_fp` is None, a conversation log is written next to `out_fp` at
    `<out_fp without .jsonl>_conversations.jsonl`. Pass `log_fp=""` to disable
    logging entirely. Each log line contains the prompt sent to the model and
    the raw output for one (example, conclusion) pair.

    Modes
    -----
    - `cot=False` (default, letter-only forced choice):
        * Encoder-decoder: greedy generation with `max_new_tokens=2`, first
          character is the answer.
        * Decoder-only: one forward pass, argmax next-token logit over {A,B,C}.

    - `cot=True` (chain-of-thought):
        * Both branches: greedy generation with `cot_max_new_tokens` tokens.
        * Prompt gets a "think step by step" suffix.
        * Reasoning-model `<think>...</think>` blocks are stripped before letter
          extraction and stored separately in the log as `scratchpad`.
        * Letter extracted as the LAST standalone A/B/C in the cleaned output.
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

    # Set up conversation log (default: alongside out_fp; pass log_fp="" to disable)
    if log_fp is None:
        out_path = Path(out_fp)
        log_fp = str(out_path.with_name(out_path.stem + "_conversations.jsonl"))
    if log_fp:
        Path(log_fp).parent.mkdir(parents=True, exist_ok=True)
    # buffering=1 => line-buffered, flush on every '\n'. Critical for cluster jobs
    # where we want to see progress and inspect predictions while running.
    flog = open(log_fp, "w", buffering=1) if log_fp else None

    with open(data_fp) as fin, open(out_fp, "w", buffering=1) as fout, torch.inference_mode():
        n_preds = 0
        for item_idx, line in enumerate(fin):
            item = json.loads(line)

            for conc_code, conc in item["conclusions"].items():
                prompt = build_prompt(item, conc)
                if cot:
                    # Swap the "letter only" suffix for a CoT suffix.
                    prompt = prompt.replace(
                        "Answer with the letter only.",
                        "Let's think step by step. Then answer with a single letter A, B, or C.",
                    )

                scratchpad = ""

                # ── Encoder-decoder branch (Flan-T5) ────────────────────────
                if getattr(config, "is_encoder_decoder", False):
                    enc = tok(prompt, return_tensors="pt").to(device)

                    out = mdl.generate(
                        **enc,
                        max_new_tokens=cot_max_new_tokens if cot else max_new_tokens,
                        min_new_tokens=1,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                    text = tok.decode(out[0], skip_special_tokens=True).strip()

                    if cot:
                        scratchpad, cleaned = _split_think(text)
                        letter = _extract_letter(cleaned)
                    else:
                        cleaned = text
                        letter = text[:1].upper() if text else ""

                    idx = LETTERS.find(letter) if letter else -1
                    pred_index = idx if 0 <= idx < len(OPTIONS) else -1

                # ── Decoder-only branch (Mistral, DeepSeek, …) ──────────────
                else:
                    if getattr(tok, "chat_template", None):
                        prompt = tok.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
                    input_device = mdl.get_input_embeddings().weight.device
                    enc = {k: v.to(input_device) for k, v in enc.items()}

                    if cot:
                        # Free generation, then extract answer from cleaned text.
                        out = mdl.generate(
                            **enc,
                            max_new_tokens=cot_max_new_tokens,
                            min_new_tokens=1,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                        # Only decode the newly generated tokens, not the prompt.
                        gen_ids = out[0, enc["input_ids"].shape[1]:]
                        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

                        scratchpad, cleaned = _split_think(text)
                        letter = _extract_letter(cleaned)
                        pred_index = "ABC".find(letter) if letter else -1
                    else:
                        # Argmax over {A,B,C} next-token logits (fast, forced-choice).
                        logits = mdl(**enc).logits[0, -1]

                        cand = {}
                        for L in ["A", "B", "C"]:
                            tid = _single_token_id(tok, L)
                            if tid is None:
                                tid = _single_token_id(tok, " " + L)
                            if tid is not None:
                                cand[L] = tid

                        if not cand:
                            letter = ""
                            pred_index = -1
                            text = ""
                            cleaned = ""
                        else:
                            scores = {L: logits[tid].item() for L, tid in cand.items()}
                            letter = max(scores, key=scores.get)
                            pred_index = "ABC".find(letter)
                            text = letter
                            cleaned = letter

                conc["pred_index"] = pred_index

                if flog:
                    log_record = {
                        "id": item.get("id"),
                        "form": item.get("form"),
                        "conclusion_code": conc_code,
                        "prompt": prompt,
                        "raw_output": text,
                        "pred_letter": letter,
                        "pred_index": pred_index,
                    }
                    if cot:
                        log_record["scratchpad"] = scratchpad
                        log_record["cleaned_output"] = cleaned
                    flog.write(json.dumps(log_record, ensure_ascii=False) + "\n")

                n_preds += 1
                # Progress print every 20 predictions so SLURM stdout shows life.
                if n_preds % 20 == 0:
                    print(f"[progress] {n_preds} predictions written", flush=True)

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    if flog:
        flog.close()
        print(f"Saved conversation log to {log_fp}")
    print(f"Saved predictions to {out_fp}")
