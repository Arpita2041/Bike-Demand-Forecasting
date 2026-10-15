import argparse
import json
from .workflow import train, predict
from .data import obtain_data
from .common import ROOT

def main():
    parser = argparse.ArgumentParser(description="Rolling one-hour-ahead bike demand forecasts")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    commands.add_parser("demo", help="Run the labelled offline simulated demonstration")
    training = commands.add_parser("train")
    training.add_argument("--mode", choices=["quick", "full"], default="full")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--data", help="Optional official-schema hour.csv")
    training.add_argument("--output", default="outputs/full")
    inference = commands.add_parser("predict")
    inference.add_argument("--model", default="outputs/full/models/model.joblib")
    inputs = inference.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Prepared feature JSON")
    inputs.add_argument("--history", help="Observed hourly CSV; forecast the hour after its last row")
    args = parser.parse_args()
    if args.command == "download":
        print(obtain_data())
    elif args.command == "demo":
        train("quick", 42, ROOT / "data/demo/bike_demo.csv", ROOT / "outputs/demo")
    elif args.command == "train":
        train(args.mode, args.seed, args.data, args.output)
    else:
        print(json.dumps(predict(args.model, args.input, args.history), indent=2))
