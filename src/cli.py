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
        predict(args.model, args.data, args.out)
    elif args.cmd == "eval":
        evaluate(args.gold, args.pred, args.out)

if __name__ == "__main__":
    main()
