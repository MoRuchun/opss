"""Model definition for the three-span, seven-story RC frame using OpenSeesPy."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from openseespy import opensees as ops


@dataclass
class ModelGeometry:
    """Container for key node collections used by recorders and post-processing."""

    master_nodes: List[int]
    story_pairs: List[Tuple[int, int]]
    base_nodes: List[int]
    top_node: int
    node_grid: List[List[int]]
    span_width: float
    story_height: float


def build_frame_model(
    n_story: int = 7,
    n_span: int = 3,
    bay_width: float = 6.0,
    story_height: float = 3.0,
    params: Dict[str, float] | None = None,
) -> ModelGeometry:
    """Create the OpenSeesPy model for the planar RC frame.

    Parameters
    ----------
    n_story:
        Number of stories in the frame.
    n_span:
        Number of bays (spans) in the frame.
    bay_width:
        Reference bay width in meters.
    story_height:
        Reference story height in meters.
    params:
        Dictionary of model parameters sampled from the uncertainty module.

    Returns
    -------
    ModelGeometry
        Data object with node metadata to support recorders and response
        post-processing.
    """
    if params is None:
        raise ValueError("Model parameters must be provided to build the frame model.")

    logging.info("Initializing OpenSees model with %d stories and %d spans.", n_story, n_span)

    # Adjust primary geometric parameters based on sampled values.
    span_width_eff = max(params.get("L", bay_width), 3.0)
    story_height_eff = max(params.get("Hc", story_height), 2.5)

    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Node generation -----------------------------------------------------
    node_grid: List[List[int]] = []
    node_tag = 1
    for level in range(n_story + 1):
        y_coord = story_height_eff * level
        level_nodes: List[int] = []
        for column in range(n_span + 1):
            x_coord = span_width_eff * column
            ops.node(node_tag, x_coord, y_coord)
            level_nodes.append(node_tag)
            node_tag += 1
        node_grid.append(level_nodes)

    # Base fixities -------------------------------------------------------
    for base_node in node_grid[0]:
        ops.fix(base_node, 1, 1, 1)

    # Mass assignment (tributary masses at floor nodes) -------------------
    density_concrete = max(params.get("rho_con", 2500.0), 2000.0)  # kg/m^3
    slab_thickness = 0.20  # m, assumed constant slab thickness
    frame_depth = span_width_eff  # tributary depth assumption equal to bay width
    floor_area = span_width_eff * n_span * frame_depth
    floor_mass_total = density_concrete * floor_area * slab_thickness
    for level_nodes in node_grid[1:]:
        tributary_mass = floor_mass_total / len(level_nodes)
        for node in level_nodes:
            ops.mass(node, tributary_mass, tributary_mass * 0.05, 0.0)

    # Section and material properties ------------------------------------
    rho_s = max(min(params.get("rho_s", 0.02), 0.04), 0.005)

    fc_core = max(params.get("fc_core", 35.0), 10.0)  # MPa
    fcu_core = max(params.get("fcu_core", 20.0), 5.0)  # MPa
    fc_cover = max(params.get("fc_cover", 30.0), 8.0)  # MPa

    eps_c_core = max(params.get("eps_c_core", 0.0022), 0.0015)
    eps_cu_core = max(params.get("eps_cu_core", 0.0040), 0.0025)
    eps_c_cover = max(params.get("eps_c_cover", 0.0020), 0.0012)
    eps_cu_cover = max(params.get("eps_cu_cover", 0.0035), 0.0020)

    fys = max(params.get("fys", 420.0), 300.0)  # MPa
    es_modulus = max(params.get("Es", 200000.0), 150000.0)  # MPa
    bs_ratio = max(params.get("bs", 0.01), 0.002)

    # Convert to SI units.
    ec_core = 4700.0 * (fc_core ** 0.5) * 1e6  # Pa
    ec_cover = 4700.0 * (fc_cover ** 0.5) * 1e6  # Pa
    es_pa = es_modulus * 1e6  # Pa
    fys_pa = fys * 1e6  # Pa

    logging.debug(
        (
            "Material sample | fc_core=%.2f MPa, fcu_core=%.2f MPa, "
            "eps_c_core=%.4f, eps_cu_core=%.4f, fc_cover=%.2f MPa, "
            "eps_c_cover=%.4f, eps_cu_cover=%.4f, fys=%.1f MPa, Es=%.0f MPa, bs=%.4f"
        ),
        fc_core,
        fcu_core,
        eps_c_core,
        eps_cu_core,
        fc_cover,
        eps_c_cover,
        eps_cu_cover,
        fys,
        es_modulus,
        bs_ratio,
    )

    column_depth = max(params.get("Ds", 0.5), 0.3)
    column_width = max(0.35, column_depth * 0.7)
    column_area = column_depth * column_width
    column_inertia = column_width * column_depth**3 / 12.0

    beam_depth = max(column_depth * 0.8, 0.3)
    beam_width = max(0.30, beam_depth * 0.6)
    beam_area = beam_depth * beam_width
    beam_inertia = beam_width * beam_depth**3 / 12.0

    e_column_eff = (1.0 - rho_s) * ec_core + rho_s * es_pa
    e_beam_eff = (1.0 - rho_s) * ec_cover + rho_s * es_pa

    ops.geomTransf("Linear", 1)

    element_tag = 1
    # Columns -------------------------------------------------------------
    for column in range(n_span + 1):
        for level in range(n_story):
            i_node = node_grid[level][column]
            j_node = node_grid[level + 1][column]
            ops.element(
                "elasticBeamColumn",
                element_tag,
                i_node,
                j_node,
                column_area,
                e_column_eff,
                column_inertia,
                1,
            )
            element_tag += 1

    # Beams ---------------------------------------------------------------
    for level in range(1, n_story + 1):
        for column in range(n_span):
            i_node = node_grid[level][column]
            j_node = node_grid[level][column + 1]
            ops.element(
                "elasticBeamColumn",
                element_tag,
                i_node,
                j_node,
                beam_area,
                e_beam_eff,
                beam_inertia,
                1,
            )
            element_tag += 1

    # Master nodes for recorders: choose middle column nodes.
    middle_column = n_span // 2
    master_nodes = [node_grid[level][middle_column] for level in range(1, n_story + 1)]
    story_pairs = [
        (node_grid[level][middle_column], node_grid[level + 1][middle_column])
        for level in range(n_story)
    ]
    top_node = node_grid[-1][middle_column]
    base_nodes = node_grid[0]

    logging.info(
        "Model created with span width %.3f m, story height %.3f m, and %d elements.",
        span_width_eff,
        story_height_eff,
        element_tag - 1,
    )

    return ModelGeometry(
        master_nodes=master_nodes,
        story_pairs=story_pairs,
        base_nodes=base_nodes,
        top_node=top_node,
        node_grid=node_grid,
        span_width=span_width_eff,
        story_height=story_height_eff,
    )
