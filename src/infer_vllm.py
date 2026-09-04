"""
vLLM backend for `predict`, mirroring src/infer.py's contract exactly.

Not supported here: encoder-decoder models (Flan-T5). 
"""

import contextlib
import json
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from .prompts import build_prompt
from .infer import _split_think, _extract_letter, _single_token_id, OPTIONS

# Suffix used to close an unterminated reasoning block. Kept byte-identical to
# the HF backend (src/infer.py) so budget-forced answers stay comparable
# across the two.
FORCE_SUFFIX = "\n</think>\n\nFinal answer:"
FORCE_ANSWER_TOKENS = 32


def _validate_local_model(model_path: str) -> None:
    """
    Fails fast on a bad checkpoint directory.

    Without this, a typo in the path surfaces either as an opaque transformers
    stack trace or -- worse, when the path merely looks like a hub id -- as a
    network call that hangs for minutes on a compute node with no internet.
    """
    entries = os.listdir(model_path)

    if "config.json" not in entries:
        raise ValueError(f"no config.json in {model_path}\n"
                         f"       found: {sorted(entries)[:20]}")

    if not any(f.endswith((".safetensors", ".bin")) for f in entries):
        raise ValueError(f"no .safetensors or .bin weight files in {model_path}")

    if not any(f in ("tokenizer.json", "tokenizer.model",
                     "tokenizer_config.json", "vocab.json") for f in entries):
        raise ValueError(f"no tokenizer files in {model_path}")


def _hf_config_overrides(model_name: str, local_files_only: bool = False) -> dict:
    """
    Fields transformers declares as None and vLLM then does arithmetic on.

    vLLM reads these with getattr(config, name, <sensible default>), so an
    attribute that exists and is None defeats the default: the engine dies in
    llama.py with "unsupported operand type(s) for *: ... and 'NoneType'"
    before a single weight is loaded. Recent transformers declares both on the
    config class, so on vLLM 0.8.2 this fires for Mistral on every machine.

    Supplied as overrides rather than written into config.json, because the
    weights are not ours to edit and a re-download would undo it anyway. Only
    fields that are actually None are touched, so a model that states its own
    head_dim keeps it.
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True,
                                     local_files_only=local_files_only)
    overrides = {}

    if getattr(cfg, "head_dim", None) is None:
        hidden = getattr(cfg, "hidden_size", None)
        heads = getattr(cfg, "num_attention_heads", None)
        if hidden and heads:
            overrides["head_dim"] = hidden // heads

    if getattr(cfg, "partial_rotary_factor", None) is None:
        overrides["partial_rotary_factor"] = 1.0   # full rotary

    return overrides


def _load_llm(model_name: str, tensor_parallel_size: int, max_model_len: int,
              gpu_memory_utilization: float, dtype: str,
              local_files_only: bool = False):
    """Construct the engine. Must run before anything touches CUDA."""
    from vllm import LLM

    kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    overrides = _hf_config_overrides(model_name, local_files_only)
    if overrides:
        print(f"Model config overrides: {overrides}", flush=True)
        kwargs["hf_overrides"] = overrides

    try:
        return LLM(**kwargs)
    except TypeError as exc:
        if "hf_overrides" not in str(exc):
            raise
        # Older vLLM without the override hook. Say exactly what to do rather
        # than let it die further down with a TypeError about NoneType.
        print(f"WARNING: this vLLM does not accept hf_overrides ({exc}). "
              f"If it now fails in llama.py, add these to the model's "
              f"config.json by hand: {overrides}", flush=True)
        kwargs.pop("hf_overrides")
        return LLM(**kwargs)


def _apply_chat_template(tok, prompt: str) -> str:
    """
    Same treatment the HF decoder-only branch gives a prompt (src/infer.py):
    wrap in the model's chat template when it has one, otherwise pass through.
    """
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _candidate_token_ids(tok) -> dict:
    """
    Map each answer letter to the single token id that represents it.

    Ported from src/infer.py's forced-choice branch, including the " A"
    fallback: under a chat template the letter is preceded by whitespace, and
    for some tokenizers only the space-prefixed form is a single token.
    """
    cand = {}
    for letter in ["A", "B", "C"]:
        tid = _single_token_id(tok, letter)
        if tid is None:
            tid = _single_token_id(tok, " " + letter)
        if tid is not None:
            cand[letter] = tid
    return cand


def _sampling_params(do_sample: bool, temperature: float, top_p: float,
                     max_tokens: int, seed=None, n: int = 1):
    """
    Greedy unless sampling was asked for, matching the HF backend.

    `n` draws that many independent continuations per prompt in one pass. It
    only makes sense under sampling: greedy decoding would return the same
    continuation n times, so the greedy branch ignores it.
    """
    from vllm import SamplingParams

    if do_sample:
        return SamplingParams(n=n, temperature=temperature, top_p=top_p,
                              max_tokens=max_tokens, min_tokens=1, seed=seed)
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=1)


def _sample_path(path: str, idx: int, n_samples: int) -> str:
    """
    Per-sample output path. A single-sample run keeps its filename exactly as
    given, so nothing about existing runs changes; a multi-sample run gets
    `_sample0`, `_sample1`, ... before the extension.
    """
    if n_samples == 1:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}_sample{idx}{p.suffix}"))


def _generate(llm, token_ids: list, params):
    """
    Run one batch. `params` is either a single SamplingParams applied to every
    prompt, or a list with one per prompt (used for per-request seeds).

    Returns one list per prompt, holding (text, n_generated_tokens) per sample:
    [[(text, n), ...n_samples...], ...one per prompt...]. Prompt order matches
    the input -- vLLM preserves the request list's ordering even though it
    schedules out of order -- and samples are sorted by their own index rather
    than by likelihood, so "sample 3" means the same draw on every re-run.
    """
    if not token_ids:
        return []

    try:
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
        outputs = llm.generate(prompts, params, use_tqdm=False)
    except ImportError:
        outputs = llm.generate(prompt_token_ids=token_ids,
                               sampling_params=params, use_tqdm=False)

    return [[(c.text, len(c.token_ids))
             for c in sorted(o.outputs, key=lambda c: getattr(c, "index", 0))]
            for o in outputs]


def _forced_choice(llm, tok, token_ids: list, cand: dict):
    """
    One-token forced choice over {A, B, C}.

    The HF backend takes argmax over the next-token logits restricted to the
    three letter ids. vLLM does not expose raw logits, but `allowed_token_ids`
    masks everything else out, so a temperature-0 sample from the masked
    distribution IS that argmax -- same decision, same tie-breaking.

    Where `allowed_token_ids` is unavailable, fall back to reading the top-k
    logprobs and picking the best letter among them. That fallback can return
    no letter at all (none in the top 20), which the HF path never does, so it
    reports how often that happened rather than hiding it.
    """
    from vllm import SamplingParams

    ids = list(cand.values())
    letters_by_id = {tid: letter for letter, tid in cand.items()}

    try:
        params = SamplingParams(temperature=0.0, max_tokens=1,
                                allowed_token_ids=ids, logprobs=len(ids))
    except TypeError:
        # vLLM without allowed_token_ids: decide from the top-k instead.
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        return _forced_choice_from_logprobs(llm, token_ids, params,
                                            letters_by_id)

    try:
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=i) for i in token_ids]
        outputs = llm.generate(prompts, params, use_tqdm=False)
    except ImportError:
        outputs = llm.generate(prompt_token_ids=token_ids,
                               sampling_params=params, use_tqdm=False)

    letters = []
    for o in outputs:
        out_ids = o.outputs[0].token_ids
        letters.append(letters_by_id.get(out_ids[0], "") if out_ids else "")
    return letters


def _forced_choice_from_logprobs(llm, token_ids: list, params, letters_by_id):
    """Fallback for `_forced_choice`: best letter present in the top-k."""
    try:
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=i) for i in token_ids]
        outputs = llm.generate(prompts, params, use_tqdm=False)
    except ImportError:
        outputs = llm.generate(prompt_token_ids=token_ids,
                               sampling_params=params, use_tqdm=False)

    letters, missing = [], 0
    for o in outputs:
        top = (o.outputs[0].logprobs or [{}])[0] or {}
        scored = [(lp.logprob if hasattr(lp, "logprob") else lp, letter)
                  for tid, letter in letters_by_id.items()
                  for lp in [top.get(tid)] if lp is not None]
        if scored:
            letters.append(max(scored)[1])
        else:
            letters.append("")
            missing += 1

    if missing:
        print(f"WARNING: no A/B/C in the top-20 logprobs for {missing} of "
              f"{len(outputs)} prompts; those are recorded as unparsed "
              f"(pred_index=-1). The HF backend always decides, so these "
              f"predictions are not comparable to an HF run.", flush=True)
    return letters


def predict_vllm(model_name: str, data_fp: str, out_fp: str,
                 log_fp: str | None = None, cot: bool = False,
                 thinking: bool = False, cot_max_new_tokens: int = 256,
                 do_sample: bool = False, seed: int = 0,
                 temperature: float = 0.6, top_p: float = 0.95,
                 tensor_parallel_size: int = 1, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.90, dtype: str = "bfloat16",
                 chunk_items: int = 512, n_samples: int = 1):
    """
    vLLM equivalent of `src.infer.predict`, writing the same two files.

    Output is identical in shape to the HF backend: `out_fp` gets one JSON
    line per dataset item with `pred_index` filled into each conclusion, and
    the conversation log gets one line per (item, conclusion).

    Differences from the HF backend, all deliberate
    -----------------------------------------------
    - `thinking` is a fallback, not the switch. Whether budget forcing applies
      is read off the chat template (does it open a <think> block?) and off
      each individual output (does it contain <think>?). src.infer.predict
      requires the flag by hand, so a forgotten `--thinking` silently disables
      forcing on a reasoning model; here the flag only forces the behaviour on.
    - Budget-forced records keep both texts: `raw_output` stays the original
      generation and the continuation lands in `forced_output`. The HF backend
      overwrites `raw_output` with the continuation, losing the reasoning. The
      extracted letter is the same either way.
    - Sampled runs seed each request as `seed * 1_000_003 + <prompt index>`.
      A single seed shared across a batch would make results depend on how the
      engine happened to batch them; per-request seeds make a run reproducible
      independently of `chunk_items`. These will NOT reproduce the HF backend's
      torch.manual_seed runs -- a vLLM seed sweep is its own set of numbers.

    Processing happens in chunks of `chunk_items` dataset items (8 prompts
    each) so a long job writes results as it goes instead of only at the end.
    The chunk is already far larger than needed to saturate the GPU.

    Multiple samples per prompt
    ---------------------------
    `n_samples > 1` draws that many independent continuations per prompt in a
    single pass and writes one prediction file per draw, suffixed `_sample0`,
    `_sample1`, ... A five-sample run therefore replaces five separate jobs
    over the same prompts: one model load instead of five, and the engine
    batches all n x prompts sequences together.

    It requires `cot=True` and `do_sample=True`. The forced-choice path is a
    deterministic argmax and greedy decoding repeats itself, so in either case
    the extra draws would be identical copies.

    Note for a variance analysis: n draws under one seed are NOT the same
    experiment as n runs at seeds 0..n-1. They are iid draws from the same
    sampler state rather than independently seeded runs -- which is the usual
    thing to want for decoding variance, but it is a different quantity, so do
    not mix the two in one table.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}.")
    if n_samples > 1 and not cot:
        raise ValueError(
            "--n-samples > 1 requires --cot. The forced-choice path is a "
            "deterministic argmax over {A,B,C}, so extra draws would be "
            "identical copies of the same answer."
        )
    if n_samples > 1 and not do_sample:
        raise ValueError(
            "--n-samples > 1 requires --do-sample. Greedy decoding returns "
            "the same continuation every time, so extra draws would be "
            "identical copies."
        )

    # A local checkpoint directory is loaded strictly offline: compute nodes
    # generally have no route to huggingface.co, and a hub lookup there hangs
    # rather than failing.
    is_local = os.path.isdir(model_name)
    if is_local:
        _validate_local_model(model_name)
        print(f"Local checkpoint: {model_name}", flush=True)

    config = AutoConfig.from_pretrained(model_name, local_files_only=is_local)
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError(
            f"{model_name} is an encoder-decoder model. vLLM does not support "
            f"T5-family encoder-decoder inference; run it with the HF backend "
            f"(--backend hf) instead."
        )

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Does the chat template end inside a reasoning block? DeepSeek-R1-Distill's
    # does, so its output only becomes an answer after </think>. Mistral's does
    # not: its whole output is the answer, and treating a missing </think> as
    # "ran out of budget" would spend a second pass on every prompt and read
    # the answer off text the model never meant as one.
    reasoning_prompt = "<think>" in _apply_chat_template(tok, "probe")
    print(f"Chat template opens a reasoning block: {reasoning_prompt}",
          flush=True)

    llm = _load_llm(model_name, tensor_parallel_size, max_model_len,
                    gpu_memory_utilization, dtype, local_files_only=is_local)

    cand = _candidate_token_ids(tok) if not cot else {}
    if not cot and not cand:
        raise ValueError(
            f"None of A/B/C is a single token for {model_name}'s tokenizer, "
            f"so forced choice has nothing to score. Use --cot."
        )

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

    out_paths = [_sample_path(out_fp, i, n_samples) for i in range(n_samples)]

    # Set up conversation logs (default: alongside each out path; pass
    # log_fp="" to disable), matching src.infer.predict.
    if log_fp == "":
        log_paths = [None] * n_samples
    elif log_fp is None:
        log_paths = [str(Path(o).with_name(Path(o).stem + "_conversations.jsonl"))
                     for o in out_paths]
    else:
        log_paths = [_sample_path(log_fp, i, n_samples) for i in range(n_samples)]
    for lp in log_paths:
        if lp:
            Path(lp).parent.mkdir(parents=True, exist_ok=True)

    n_preds = 0
    prompt_ordinal = 0   # global index, feeds the per-request seed

    with contextlib.ExitStack() as stack:
        fin = stack.enter_context(open(data_fp))
        fouts = [stack.enter_context(open(o, "w", buffering=1))
                 for o in out_paths]
        flogs = [stack.enter_context(open(lp, "w", buffering=1)) if lp else None
                 for lp in log_paths]

        chunk = []
        for line in fin:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_items:
                n_preds, prompt_ordinal = _run_chunk(
                    llm, tok, chunk, fouts, flogs, cot, thinking,
                    reasoning_prompt, cot_max_new_tokens, do_sample, seed,
                    temperature, top_p, cand, n_preds, prompt_ordinal,
                    n_samples)
                chunk = []

        if chunk:
            n_preds, prompt_ordinal = _run_chunk(
                llm, tok, chunk, fouts, flogs, cot, thinking, reasoning_prompt,
                cot_max_new_tokens, do_sample, seed, temperature, top_p,
                cand, n_preds, prompt_ordinal, n_samples)

    for lp in log_paths:
        if lp:
            print(f"Saved conversation log to {lp}")
    for o in out_paths:
        print(f"Saved predictions to {o}")


def _run_chunk(llm, tok, items, fouts, flogs, cot, thinking, reasoning_prompt,
               cot_max_new_tokens, do_sample, seed, temperature, top_p,
               cand, n_preds, prompt_ordinal, n_samples=1):
    """
    Generate for one chunk of items and write their records.

    Every (item, conclusion) in the chunk becomes one request, and all of them
    go to the engine in a single call; with n_samples > 1 each request also
    carries n draws, so one call covers len(items) x 8 x n_samples sequences.

    Returns the updated (n_preds, prompt_ordinal) counters.
    """
    # Flatten to one request per (item, conclusion), keeping a back-reference
    # so results can be fanned out to the right conclusion dict.
    refs, prompts = [], []
    for item in items:
        for conc_code, conc in item["conclusions"].items():
            prompt = build_prompt(item, conc)
            if cot:
                # Swap the "letter only" suffix for a CoT suffix.
                prompt = prompt.replace(
                    "Answer with the letter only.",
                    "Let's think step by step. Then answer with a single "
                    "letter A, B, or C.",
                )
            refs.append((item, conc_code, conc, prompt))
            prompts.append(_apply_chat_template(tok, prompt))

    # add_special_tokens=False: the chat template already carries BOS.
    token_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

    # records[prompt][sample] -> {"raw", "n_gen", "forced"}
    records = [None] * len(refs)

    if cot:
        seeds = [seed * 1_000_003 + prompt_ordinal + i for i in range(len(refs))]
        params = [_sampling_params(do_sample, temperature, top_p,
                                   cot_max_new_tokens, sd, n_samples)
                  for sd in seeds] \
            if do_sample else _sampling_params(do_sample, temperature, top_p,
                                               cot_max_new_tokens)
        results = _generate(llm, token_ids, params)

        # Budget forcing: an output that never closed its reasoning block ran
        # out of budget, so any letter in it comes from mid-reasoning text.
        # Close </think> by hand, cue a final answer, and generate a few more
        # tokens greedily. Batched across every (prompt, sample) that needs it,
        # so the second pass costs one round trip for the whole chunk.
        forced_slots, forced_ids = [], []
        for i, samples in enumerate(results):
            records[i] = [{"raw": raw, "n_gen": n_gen, "forced": None}
                          for raw, n_gen in samples]
            for j, (raw, _n) in enumerate(samples):
                is_reasoning = reasoning_prompt or thinking or "<think>" in raw
                if is_reasoning and "</think>" not in raw.lower():
                    forced_slots.append((i, j))
                    forced_ids.append(
                        token_ids[i]
                        + tok(raw, add_special_tokens=False)["input_ids"]
                        + tok(FORCE_SUFFIX, add_special_tokens=False)["input_ids"]
                    )

        if forced_ids:
            forced_params = _sampling_params(False, temperature, top_p,
                                             FORCE_ANSWER_TOKENS)
            for (i, j), samples in zip(forced_slots,
                                       _generate(llm, forced_ids, forced_params)):
                text, n_forced = samples[0]
                records[i][j]["forced"] = text
                records[i][j]["n_gen"] += n_forced
    else:
        letters = _forced_choice(llm, tok, token_ids, cand)
        for i, letter in enumerate(letters):
            records[i] = [{"raw": letter, "n_gen": None, "forced": None,
                           "letter": letter}]

    # Fan results back out, one complete pass over the chunk per sample. The
    # items are mutated and serialized within the same pass, so each sample's
    # pred_index reaches its own file before the next sample overwrites it --
    # no copy of the chunk is needed.
    for sample_idx in range(n_samples):
        for (item, conc_code, conc, prompt), rec_samples in zip(refs, records):
            rec = rec_samples[sample_idx]
            if cot:
                # The letter comes from the forced continuation when there is
                # one: that is the text written after the block was closed.
                answer_text = (rec["forced"] if rec["forced"] is not None
                               else rec["raw"])
                scratchpad, cleaned = _split_think(answer_text)
                letter = _extract_letter(cleaned)
            else:
                letter = rec["letter"]
                scratchpad, cleaned = "", letter

            pred_index = "ABC".find(letter) if letter else -1
            if not 0 <= pred_index < len(OPTIONS):
                pred_index = -1
            conc["pred_index"] = pred_index

            flog = flogs[sample_idx]
            if flog:
                log_record = {
                    "id": item.get("id"),
                    "form": item.get("form"),
                    "conclusion_code": conc_code,
                    "prompt": prompt,
                    "raw_output": rec["raw"],
                    "pred_letter": letter,
                    "pred_index": pred_index,
                }
                if n_samples > 1:
                    log_record["sample_index"] = sample_idx
                if cot:
                    log_record["scratchpad"] = scratchpad
                    log_record["cleaned_output"] = cleaned
                    # Budget-usage diagnostics, same fields the HF backend logs.
                    log_record["gen_tokens"] = rec["n_gen"]
                    log_record["budget"] = cot_max_new_tokens
                    log_record["hit_budget"] = (
                        rec["n_gen"] >= cot_max_new_tokens
                        if rec["n_gen"] is not None else None
                    )
                    log_record["has_closing_think"] = (
                        "</think>" in rec["raw"].lower())
                    log_record["budget_forced"] = rec["forced"] is not None
                    log_record["forced_output"] = rec["forced"]
                flog.write(json.dumps(log_record, ensure_ascii=False) + "\n")

            n_preds += 1

        for item in items:
            fouts[sample_idx].write(json.dumps(item, ensure_ascii=False) + "\n")

    prompt_ordinal += len(refs)
    print(f"[progress] {n_preds} predictions written", flush=True)
    return n_preds, prompt_ordinal
