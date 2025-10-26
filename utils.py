"""Utility helpers for OpenSeesPy dynamic time-history analyses."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
from openseespy import opensees as ops


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def timestamped_name(prefix: str) -> str:
    """Return a name with an ISO timestamp suffix."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def setup_logging(log_file: str | None = None) -> None:
    """Configure the logging system for the project."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        ensure_directory(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def create_recorders(
    output_dir: str,
    master_nodes: Iterable[int],
    story_pairs: Iterable[Tuple[int, int]],
    base_nodes: Iterable[int],
) -> Dict[str, List[str]]:
    """Set up OpenSees recorders for displacement, drift, and base shear."""
    ensure_directory(output_dir)

    master_nodes = list(master_nodes)
    story_pairs = list(story_pairs)
    base_nodes = list(base_nodes)

    displacement_path = os.path.join(output_dir, "floor_displacement.out")
    ops.recorder(
        "Node",
        "-file",
        displacement_path,
        "-time",
        "-node",
        *master_nodes,
        "-dof",
        1,
        "disp",
    )

    drift_paths: List[str] = []
    for idx, (i_node, j_node) in enumerate(story_pairs, start=1):
        path = os.path.join(output_dir, f"story_{idx}_drift.out")
        ops.recorder(
            "Drift",
            "-file",
            path,
            "-time",
            "-iNode",
            i_node,
            "-jNode",
            j_node,
            "-dof",
            1,
            "-perpDirn",
            2,
        )
        drift_paths.append(path)

    base_reaction_path = os.path.join(output_dir, "base_reaction.out")
    ops.recorder(
        "Node",
        "-file",
        base_reaction_path,
        "-time",
        "-node",
        *base_nodes,
        "-dof",
        1,
        "reaction",
    )

    return {
        "displacement": [displacement_path],
        "drift": drift_paths,
        "base_reaction": [base_reaction_path],
    }


def load_ground_motion(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column ground motion file (time, acceleration in g)."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Ground motion file {path} must contain at least two columns.")
    time = data[:, 0]
    accel = data[:, 1] * 9.80665  # Convert from g to m/s^2
    return time, accel


def summarize_results(
    recorder_paths: Dict[str, List[str]],
    master_nodes: List[int],
    top_node: int,
) -> Dict[str, float]:
    """Compute key response metrics from recorder outputs."""
    displacement_file = recorder_paths["displacement"][0]
    disp_data = np.loadtxt(displacement_file)
    if disp_data.size == 0:
        return {"max_top_disp": 0.0, "max_drift_ratio": 0.0, "max_base_shear": 0.0}

    # Node displacement columns follow time + node responses.
    master_index = master_nodes.index(top_node)
    top_disp = disp_data[:, master_index + 1]
    max_top_disp = float(np.max(np.abs(top_disp)))

    drift_values: List[float] = []
    for path in recorder_paths["drift"]:
        drift_data = np.loadtxt(path)
        if drift_data.size == 0:
            continue
        drift_values.append(np.max(np.abs(drift_data[:, 1])))
    max_drift_ratio = float(max(drift_values)) if drift_values else 0.0

    base_reaction_file = recorder_paths["base_reaction"][0]
    base_data = np.loadtxt(base_reaction_file)
    if base_data.size:
        reaction_sum = np.sum(base_data[:, 1:], axis=1)
        max_base_shear = float(np.max(np.abs(reaction_sum)))
    else:
        max_base_shear = 0.0

    return {
        "max_top_disp": max_top_disp,
        "max_drift_ratio": max_drift_ratio,
        "max_base_shear": max_base_shear,
    }


def write_summary(path: str, summary: Dict[str, float]) -> None:
    """Persist a summary dictionary to JSON."""
    ensure_directory(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
