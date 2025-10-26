"""Command-line entry point for the OpenSeesPy dynamic analysis workflow."""
from __future__ import annotations

import glob
import logging
import os
from typing import List

from analysis import run_time_history
from uncertainty import sample_parameters
from utils import ensure_directory, setup_logging, timestamped_name, write_summary

GROUND_MOTION_DIR = "scaled_data"
RESULTS_DIR = "results"
LOG_DIR = "logs"


def _collect_ground_motions(root_dir: str) -> List[str]:
    """Return all ground motion files within the root directory."""
    patterns = ["*.txt", "*.AT2"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))
    return sorted(files)


def main() -> None:
    """Run the analysis workflow for each available ground motion."""
    ensure_directory(RESULTS_DIR)
    ensure_directory(LOG_DIR)
    log_path = os.path.join(LOG_DIR, f"run_{timestamped_name('log')}.txt")
    setup_logging(log_path)

    gm_files = _collect_ground_motions(GROUND_MOTION_DIR)
    if not gm_files:
        logging.error("No ground motion files found under %s", GROUND_MOTION_DIR)
        return

    logging.info("Processing %d ground motions.", len(gm_files))

    summaries = []
    for gm_file in gm_files:
        params = sample_parameters()
        try:
            summary = run_time_history(gm_file, params, results_dir=RESULTS_DIR)
            summaries.append(summary)
        except Exception as exc:  # pragma: no cover - unexpected failure
            logging.exception("Analysis failed for %s: %s", gm_file, exc)

    if summaries:
        aggregate_path = os.path.join(RESULTS_DIR, "aggregate_summary.json")
        write_summary(aggregate_path, {"analyses": summaries})
        logging.info("Aggregate summary written to %s", aggregate_path)
    else:
        logging.warning("No successful analyses were recorded.")


if __name__ == "__main__":
    main()
