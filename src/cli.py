import argparse
from .data import expand_roles
from .infer import predict
from .eval import evaluate
from .generate import make_seed

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
    p_gen.add_argument("--n", type=int, required=True, help="Number of items to generate")
    p_gen.add_argument("--templates", required=True, help="Syllogism templates JSON file")
    p_gen.add_argument("--domains", required=True, help="Domains JSON file")
    p_gen.add_argument("--out", required=True, help="Output JSONL file")

    args = parser.parse_args()
    
    if args.cmd == "gen":
        make_seed(args.n, args.templates, args.domains, args.out)
    elif args.cmd == "expand":
        expand_roles(args.inp, args.out)
    elif args.cmd == "predict":
        predict(args.model, args.data, args.out)
    elif args.cmd == "eval":
        evaluate(args.gold, args.pred, args.out)

if __name__ == "__main__":
    main()
