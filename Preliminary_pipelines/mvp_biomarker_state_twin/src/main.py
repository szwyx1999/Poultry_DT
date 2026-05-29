from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the poultry biomarker/state twin pipeline.")
    parser.add_argument(
        "--config",
        default="mvp_biomarker_state_twin/config/default.yaml",
        help="Path to the pipeline YAML config file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    output_paths = run_pipeline(config)
    for label, path_value in output_paths.items():
        logging.info("%s: %s", label, path_value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
