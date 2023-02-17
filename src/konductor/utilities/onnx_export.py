#!/usr/bin/env python3
from pathlib import Path
from typing import List, Any

import torch


def export(
    model: torch.nn.Module, input_shapes: List[Any], savepath: Path | None = None
) -> None:
    """
    Export model to ONNX format given config dictionary and path
    """
    if savepath is None:
        savepath = Path.cwd() / "model.onnx"

    if savepath.exists():
        print(f"warning: overwriting existing onnx model at {savepath}")

    dummy_input = []
    for shape in input_shapes:
        dummy_input.append(torch.randn(shape, device=torch.device("cuda")))

    torch.onnx.export(model, tuple(dummy_input), str(savepath), opset_version=11)


def main() -> None:
    """Run as main script"""
    import argparse
    from ..trainer import get_model_from_experiment

    model = get_model_from_experiment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=int, nargs="+")
    parser.add_argument("--save", "-s", type=str, required=False)
    args = parser.parse_known_args()[0]

    savepath = Path(f"{args.save if args.save else 'model.onnx'}")
    print(f"Exporting onnx model to {savepath}")
    export(model, args.input, savepath=savepath)


if __name__ == "__main__":
    main()
