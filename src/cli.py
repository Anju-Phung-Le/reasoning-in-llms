import argparse
from .data import expand_roles
from .infer import predict
from .eval import evaluate
from .generate import make_seed, make_seed_all_forms

def main():
    parser = argparse.ArgumentParser(description="Run data or model tasks.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # expand command - data.py expand_roles()
    p_exp = subparsers.add_parser("expand", help="Expand dataset with roles.")
    p_exp.add_argument("--in", dest="inp", required=True, help="Input JSONL file")
    p_exp.add_argument("--out", dest="out", required=True, help="Output JSONL file")

    # predict command - infer.py predict()
    p_pred = subparsers.add_parser("predict", help="Run inference with a model.")
    p_pred.add_argument("--model", required=True, help="Hugging Face model name")
    p_pred.add_argument("--data", required=True, help="Input dataset JSONL file")
    p_pred.add_argument("--out", required=True, help="Output predictions JSONL file")
    p_pred.add_argument("--log", default=None,
                        help="Path to conversation log JSONL. "
                             "Default: alongside --out. Pass empty string to disable.")
    p_pred.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought mode (free generation + <think> stripping).")
    p_pred.add_argument("--cot-max-new-tokens", dest="cot_max_new_tokens",
                        type=int, default=256,
                        help="Max new tokens when --cot is enabled (default: 256).")
    p_pred.add_argument("--do-sample", action="store_true", dest="do_sample",
                        help="Use nucleus sampling instead of greedy decoding for CoT generation. "
                             "Non-CoT branches remain deterministic.")
    p_pred.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampled decoding (default: 0). "
                             "Ignored when --do-sample is not set.")
    p_pred.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6, DeepSeek-R1 recommended).")
    p_pred.add_argument("--top-p", dest="top_p", type=float, default=0.95,
                        help="Nucleus sampling top-p (default: 0.95).")
    p_pred.add_argument("--thinking", action="store_true", help="Enable thinking mode.")
    p_pred.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                        help="Inference backend (default: hf). vllm batches across "
                             "prompts and is far faster, but does not support "
                             "encoder-decoder models such as Flan-T5.")
    p_pred.add_argument("--tensor-parallel-size", dest="tensor_parallel_size",
                        type=int, default=1,
                        help="vLLM only: GPUs to shard across (default: 1). "
                             "Must be a power of two and divide the head count.")
    p_pred.add_argument("--max-model-len", dest="max_model_len", type=int,
                        default=4096,
                        help="vLLM only: KV cache context cap (default: 4096). "
                             "Must exceed prompt + --cot-max-new-tokens.")
    p_pred.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization",
                        type=float, default=0.90,
                        help="vLLM only: fraction of each GPU vLLM may use "
                             "(default: 0.90).")
    p_pred.add_argument("--chunk-items", dest="chunk_items", type=int, default=512,
                        help="vLLM only: dataset items generated per batch before "
                             "results are flushed to disk (default: 512).")
    p_pred.add_argument("--n-samples", dest="n_samples", type=int, default=1,
                        help="vLLM only: independent continuations to draw per "
                             "prompt in one pass, written to <out>_sample0.jsonl, "
                             "_sample1, ... (default: 1, which keeps the plain "
                             "filename). Replaces a multi-job seed sweep with a "
                             "single model load. Requires --cot and --do-sample.")

    # eval command - eval.py evaluate()
    p_eval = subparsers.add_parser("eval", help="Evaluate predictions.")
    p_eval.add_argument("--gold", required=True, help="Processed dataset JSONL")
    p_eval.add_argument("--pred", required=True, help="Predictions JSONL")
    p_eval.add_argument("--out", required=True, help="Output CSV path")

    # gen command - generate.py make_seed()
    p_gen = subparsers.add_parser("gen", help="Generate raw syllogism seeds.")
    p_gen.add_argument("--n", type=int, help="Number of items to generate (required with --templates)")
    p_gen.add_argument("--templates", help="Syllogism templates JSON file (mutually exclusive with --all-forms)")
    p_gen.add_argument("--all-forms", action="store_true", dest="all_forms",
                       help="Generate all 64 forms × 8 conclusions × all domain triplets")
    p_gen.add_argument("--domains", required=True, help="Domains JSON file")
    p_gen.add_argument("--out", required=True, help="Output JSONL file")

    args = parser.parse_args()
    
    if args.cmd == "gen":
        if args.all_forms:
            make_seed_all_forms(args.domains, args.out)
        elif args.templates and args.n:
            make_seed(args.n, args.templates, args.domains, args.out)
        else:
            parser.error("gen requires either --all-forms or --templates with --n")
    elif args.cmd == "expand":
        expand_roles(args.inp, args.out)
    elif args.cmd == "predict":
        common = dict(
            log_fp=args.log,
            cot=args.cot,
            cot_max_new_tokens=args.cot_max_new_tokens,
            do_sample=args.do_sample,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking=args.thinking,
        )
        if args.backend == "vllm":
            # Imported here, not at module scope: vllm is only installed on the
            # cluster, and `gen`/`expand`/`eval` must keep working without it.
            from .infer_vllm import predict_vllm
            predict_vllm(args.model, args.data, args.out,
                         tensor_parallel_size=args.tensor_parallel_size,
                         max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         chunk_items=args.chunk_items,
                         n_samples=args.n_samples,
                         **common)
        else:
            if args.n_samples != 1:
                parser.error("--n-samples is a vLLM-only flag; the HF backend "
                             "generates one continuation per prompt.")
            predict(args.model, args.data, args.out, **common)
    elif args.cmd == "eval":
        evaluate(args.gold, args.pred, args.out)

if __name__ == "__main__":
    main()
