import argparse
import json
from pathlib import Path
import torch
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description="Plan a tool call with the frozen Ember candidate")
    parser.add_argument("request")
    parser.add_argument("--bundle", type=Path, default=Path("."))
    parser.add_argument("--precision", choices=("full", "int4"), default="int4")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = Runtime(args.bundle, args.precision)
    print(json.dumps(runtime.plan(args.request), ensure_ascii=False))


if __name__ == "__main__":
    main()
