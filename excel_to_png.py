#!/usr/bin/env python3
"""Convert a spreadsheet of grayscale values in [0, 1] to a PNG image.

Supported input formats are .xlsx/.xlsm (requires openpyxl) and .ods
(requires odfpy). The first worksheet is read with no header row.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def positive_integer(value: str) -> int:
    """Parse a positive integer for an image dimension."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def convert_spreadsheet_to_png(
    input_path: Path,
    output_path: Path | None = None,
    resize: tuple[int, int] | None = None,
) -> Path:
    """Convert the first worksheet to an 8-bit grayscale PNG.

    ``resize``, when supplied, is a ``(width, height)`` tuple. Nearest-neighbor
    resampling is used so resizing does not introduce new grayscale values.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".png")
    else:
        output_path = Path(output_path)

    if output_path.suffix.lower() != ".png":
        raise ValueError("The output file must have a .png extension.")

    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        engine = "openpyxl"
        engine_package = "openpyxl"
    elif suffix == ".ods":
        engine = "odf"
        engine_package = "odfpy"
    else:
        raise ValueError("Input must be an .xlsx, .xlsm, or .ods spreadsheet.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    try:
        values = pd.read_excel(input_path, header=None, engine=engine)
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {suffix} files requires the '{engine_package}' package. "
            f"Install it with: python -m pip install {engine_package}"
        ) from exc

    if values.empty or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("The first worksheet must contain image data.")

    try:
        pixels = values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("All worksheet cells must contain numeric values.") from exc

    if not np.isfinite(pixels).all():
        raise ValueError("All worksheet cells must contain finite numeric values.")
    if (pixels < 0).any() or (pixels > 1).any():
        raise ValueError("Grayscale values must be in the range [0, 1].")

    # The worksheet's own row and column counts determine the image dimensions.
    grayscale = np.rint(pixels * 255).astype(np.uint8)
    image = Image.fromarray(grayscale)

    if resize is not None:
        width, height = resize
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    image.save(output_path, format="PNG")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the first worksheet of an .xlsx, .xlsm, or .ods file "
            "containing grayscale values from 0 to 1 into a PNG."
        )
    )
    parser.add_argument("input", type=Path, help="input spreadsheet")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output PNG path (defaults to the input path with a .png extension)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=positive_integer,
        metavar=("WIDTH", "HEIGHT"),
        help="optionally resize the image to these pixel dimensions",
    )
    args = parser.parse_args()

    try:
        output_path = convert_spreadsheet_to_png(
            args.input,
            args.output,
            tuple(args.resize) if args.resize else None,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()