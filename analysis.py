"""Dynamic time-history analysis routines for the OpenSeesPy frame model."""
from __future__ import annotations

import logging
import math
import os
from typing import Dict

from openseespy import opensees as ops

from model import build_frame_model
from utils import (
    create_recorders,
    ensure_directory,
    load_ground_motion,
    summarize_results,
    timestamped_name,
    write_summary,
)


def _configure_damping(zeta: float) -> None:
    """Set Rayleigh damping coefficients using the first two modes."""
    try:
        eigenvalues = ops.eigen(2)
    except Exception as exc:  # pragma: no cover - OpenSees internal error
        logging.warning("Eigenvalue analysis failed: %s", exc)
        return

    if len(eigenvalues) == 0:
        logging.warning("Eigenvalue analysis returned no modes; damping not updated.")
        return

    w1 = math.sqrt(max(eigenvalues[0], 1e-8))
    if len(eigenvalues) > 1:
        w2 = math.sqrt(max(eigenvalues[1], w1**2))
    else:
        w2 = 1.2 * w1

    if w1 <= 0.0 or w2 <= 0.0:
        logging.warning("Non-positive frequencies detected; skipping Rayleigh damping setup.")
        return

    alpha_m = 2.0 * zeta * w1 * w2 / (w1 + w2)
    beta_k = 2.0 * zeta / (w1 + w2)
    ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)
    logging.info("Applied Rayleigh damping with zeta=%.3f (alphaM=%.3e, betaK=%.3e).", zeta, alpha_m, beta_k)


def run_time_history(
    ground_motion_path: str,
    params: Dict[str, float],
    n_story: int = 7,
    n_span: int = 3,
    bay_width: float = 6.0,
    story_height: float = 3.0,
    results_dir: str = "results",
) -> Dict[str, float]:
    """Execute a transient analysis for the provided ground motion file."""
    gm_name = os.path.splitext(os.path.basename(ground_motion_path))[0]
    output_dir = os.path.join(results_dir, timestamped_name(gm_name))
    ensure_directory(output_dir)

    logging.info("Starting analysis for %s", gm_name)

    time, accel = load_ground_motion(ground_motion_path)
    dt_series = time[1:] - time[:-1]
    dt = float(dt_series.mean()) if len(dt_series) else 0.01
    if len(dt_series) and not (abs(dt_series - dt) < 1e-6).all():
        logging.warning("Ground motion %s has variable time step; using mean dt=%.6f s", gm_name, dt)

    ops.wipe()
    geom = build_frame_model(
        n_story=n_story,
        n_span=n_span,
        bay_width=bay_width,
        story_height=story_height,
        params=params,
    )

    recorder_paths = create_recorders(output_dir, geom.master_nodes, geom.story_pairs, geom.base_nodes)

    _configure_damping(params.get("zeta", 0.05))

    series_values = accel.tolist()
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *series_values)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Newton")
    ops.test("NormDispIncr", 1e-6, 20)
    ops.analysis("Transient")

    n_steps = len(series_values)
    status = "OK"
    for step in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            logging.warning(
                "Non-convergence at step %d (t=%.3f s) for %s.",
                step,
                step * dt,
                gm_name,
            )
            status = "NON_CONVERGENCE"
            break

    ops.reactions()

    summary_metrics = summarize_results(recorder_paths, geom.master_nodes, geom.top_node)
    summary = {
        "ground_motion": gm_name,
        "status": status,
        "time_step": dt,
        "num_steps": n_steps,
        "parameters": params,
        **summary_metrics,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    write_summary(summary_path, summary)

    logging.info(
        "Completed analysis for %s | status=%s, max drift=%.4f, top disp=%.4f m",
        gm_name,
        status,
        summary_metrics["max_drift_ratio"],
        summary_metrics["max_top_disp"],
    )

    ops.wipe()
    return summary
