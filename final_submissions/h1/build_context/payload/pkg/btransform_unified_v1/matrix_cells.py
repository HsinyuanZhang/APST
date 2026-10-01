"""H1 matrix cell registry (MATRIX_H1_L_IDENTITY_V1_20260906.md §1).

9 preregistered cells of the fractional L x identity design (2026-09-06 user
revision: L in {250, 350} only; M-A150 withdrawn; M-F250 added by the user's
2026-09-06 proj-add16 proposal; M-F700 added by the matrix §2 revision
formalized in REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906, blocking item C):

  2x2 primary   M-A250 / M-A350 (joined36-真)  x  M-B250 / M-B350 (concat700)
  conditional   M-C250 (add_tail — run only if (a)/(b) both weak at 250;
                truncation disclosure: L=250 uses only the LAST 250/700 bins
                of the identity template)
  user proposal M-F250 (proj_add — rank-16 bottleneck of the concat first
                layer, L-independent; vs (b): "learned bottleneck vs encoder
                mid-layer")
  baseline      M-F700 (proj_add at the FULL 700 window — the same-recipe
                full-window reference cell of the L axis; with M-F250 it forms
                the same-injection L-axis controlled pair: same CAL-2/M3,
                same LODO holdout, same recipe/seed, only variable L)
  controls      M-D250 (zero — identity total contribution),
                M-E250 (permute — structure-break sanity)

Every cell shares the global contract (matrix doc §0): same skeleton, CAL-1
{7,5,4,3} -> deploy M3, TRN-1 by updates, EMA 0.9995, p=0.10 unit dropout,
bf16 autocast, 20y train / native /20 score (ratio bridge), SEL-2 epoch-pick,
LODO method-selection exam. This module is the single enumeration source for
training scripts; costs are the doc's per-cell GPU-hour estimates (cache-warm
local machine) carried for planning only.

Identity declaration: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import h1_config, plan


@dataclass(frozen=True)
class MatrixCell:
    """One preregistered matrix cell.

    Fields:
      cell_id         matrix-doc cell name (e.g. "M-A250")
      L               input window length (matrix L axis)
      identity_mode   identity usage (matrix letters a-f mapped to
                      concat/joined/add_tail/zero/permute/proj_add)
      conditional     True -> run only under the doc's precondition
                      (M-C250: only if (a)/(b) are both weak at L=250)
      est_cost_hours  doc's estimated single-cell cost (planning only)
      note            doc's description (Chinese, verbatim intent)
    """

    cell_id: str
    L: int
    identity_mode: str
    conditional: bool
    est_cost_hours: float
    note: str


MATRIX_CELLS: tuple[MatrixCell, ...] = (
    MatrixCell("M-A250", 250, "joined", False, 1.2, "主推组合：短窗+低维"),
    MatrixCell("M-A350", 350, "joined", False, 1.6, "保守短窗"),
    MatrixCell("M-B250", 250, "concat", False, 1.7, "喂法轴对照（L=250 下 a vs b）"),
    MatrixCell("M-B350", 350, "concat", False, 2.2, "喂法轴对照（L=350 下 a vs b）"),
    MatrixCell(
        "M-C250", 250, "add_tail", True, 1.7,
        "仅当 (a)/(b) 在 250 都弱时跑；截断披露：L=250 只用 identity 模板末端 250/700",
    ),
    MatrixCell(
        "M-F250", 250, "proj_add", False, 1.6,
        "用户 2026-09-06 提案；rank-16 瓶颈正则（数学上 = concat 首层的低秩约束版），"
        "与 (b) 对比「自学瓶颈 vs 编码器中层」；L 无关",
    ),
    MatrixCell(
        "M-F700", 700, "proj_add", False, 4.0,
        "基线 cell（§2 审核后正式化，REVIEW 阻断项 C）：同配方全窗 L=700 参照系，"
        "与 M-F250 构成同注入 L 轴受控对（同 CAL-2/M3/LODO/配方/seed，唯一变量 L）",
    ),
    MatrixCell("M-D250", 250, "zero", False, 1.6, "分析对照：identity 总贡献"),
    MatrixCell("M-E250", 250, "permute", False, 1.6, "分析对照：结构破坏 sanity（应显著劣于 d 之外的 all）"),
)

# Modes forming the 2x2 primary grid (doc §1 design logic).
PRIMARY_MODES: tuple[str, ...] = ("joined", "concat")

# Baseline full-window cell (matrix doc §2 revision, formalized by
# REVIEW_MATRIX_H1_L_IDENTITY_V1_20260906 blocking item C): L = 700 is the
# ORIGINAL settled H1 window, not a matrix "shortening", so the registry
# exempts exactly this one cell from the MATRIX_L membership and the
# analysis-cells-sit-at-250 checks below.
BASELINE_CELL_ID = "M-F700"


def validate_matrix_cells() -> dict[str, Any]:
    """Structural invariants of the registry (raises on any drift)."""
    plan.require(
        len(MATRIX_CELLS) == 9,
        "the matrix has exactly 9 cells (doc §1 incl. M-F250 + the §2-revision "
        "baseline M-F700)",
    )
    ids = [cell.cell_id for cell in MATRIX_CELLS]
    plan.require(len(set(ids)) == len(ids), "duplicate cell_id in MATRIX_CELLS")
    for cell in MATRIX_CELLS:
        # L-axis gate with the single full-window exemption: every matrix cell
        # shortens to MATRIX_L {250, 350} EXCEPT the baseline M-F700, whose
        # L=700 is the original settled window itself (matrix §2 revision —
        # "700 是原始窗长非缩短"), not an override of the shorten-only rule.
        plan.require(
            cell.L in h1_config.MATRIX_L
            or (cell.cell_id == BASELINE_CELL_ID and cell.L == h1_config.FULL_WINDOW),
            f"{cell.cell_id}: L={cell.L} not in MATRIX_L {h1_config.MATRIX_L} (or the "
            f"baseline {BASELINE_CELL_ID} full window {h1_config.FULL_WINDOW})",
        )
        plan.require(
            cell.identity_mode in h1_config.IDENTITY_MODES,
            f"{cell.cell_id}: unknown identity mode {cell.identity_mode!r}",
        )
        plan.require(cell.est_cost_hours > 0.0, f"{cell.cell_id}: cost must be positive")
    # 2x2 primary = {joined, concat} x {250, 350}, all non-conditional
    primary = {
        (cell.identity_mode, cell.L)
        for cell in MATRIX_CELLS
        if cell.identity_mode in PRIMARY_MODES and not cell.conditional
    }
    expected_primary = {
        (mode, length) for mode in PRIMARY_MODES for length in h1_config.MATRIX_L
    }
    plan.require(
        primary == expected_primary,
        f"primary 2x2 must be concat/joined x {h1_config.MATRIX_L}, got {sorted(primary)}",
    )
    # analysis cells (add_tail/zero/permute/proj_add) all sit at L=250 — the
    # baseline M-F700 (proj_add at the full 700 window) is the ONE exemption
    analysis = [
        cell
        for cell in MATRIX_CELLS
        if cell.identity_mode in ("add_tail", "zero", "permute", "proj_add")
        and cell.cell_id != BASELINE_CELL_ID
    ]
    plan.require(
        all(cell.L == 250 for cell in analysis),
        "the analysis cells (add_tail/zero/permute/proj_add) all sit at L=250 "
        "(doc §1; the full-window baseline M-F700 is exempt)",
    )
    plan.require(
        len(analysis) == 4 and len({cell.identity_mode for cell in analysis}) == 4,
        "exactly one analysis cell per mode at L=250 (add_tail/zero/permute/proj_add)",
    )
    plan.require(
        cell_by_id(BASELINE_CELL_ID).L == h1_config.FULL_WINDOW
        and cell_by_id(BASELINE_CELL_ID).identity_mode == "proj_add"
        and not cell_by_id(BASELINE_CELL_ID).conditional,
        f"the baseline {BASELINE_CELL_ID} is the non-conditional proj_add full-window cell",
    )
    plan.require(
        sum(1 for cell in MATRIX_CELLS if cell.conditional) == 1
        and next(cell for cell in MATRIX_CELLS if cell.conditional).identity_mode == "add_tail",
        "exactly one conditional cell: M-C250 add_tail",
    )
    return {
        "n_cells": len(MATRIX_CELLS),
        "n_primary": 4,
        "n_conditional": 1,
        "n_user_proposal": 1,
        "n_baseline_full_window": 1,
        "n_controls": 2,
        "total_cost_hours_primary": sum(
            cell.est_cost_hours for cell in MATRIX_CELLS if cell.identity_mode in PRIMARY_MODES
        ),
        "total_cost_hours_all": sum(cell.est_cost_hours for cell in MATRIX_CELLS),
    }


def cell_by_id(cell_id: str) -> MatrixCell:
    for cell in MATRIX_CELLS:
        if cell.cell_id == cell_id:
            return cell
    raise plan.BTransformerUnifiedError(
        f"unknown matrix cell {cell_id!r}; expected one of {[c.cell_id for c in MATRIX_CELLS]}"
    )


def primary_cells() -> tuple[MatrixCell, ...]:
    """The 2x2 primary cells ((a)/(b) x 250/350)."""
    return tuple(
        cell for cell in MATRIX_CELLS if cell.identity_mode in PRIMARY_MODES
    )


def control_cells() -> tuple[MatrixCell, ...]:
    """The zero and permute analysis controls."""
    return tuple(
        cell for cell in MATRIX_CELLS if cell.identity_mode in ("zero", "permute")
    )


def baseline_cell() -> MatrixCell:
    """The full-window L-axis baseline cell (M-F700, proj_add at L=700)."""
    return cell_by_id(BASELINE_CELL_ID)


def cell_geometry(cell: MatrixCell) -> dict[str, Any]:
    """Explicit-mapping geometry for a cell (h1_config.h1_matrix_geometry).

    The returned mapping keeps the two PENDING sentinels; instantiate with
    ``BTransformerUnifiedDecoderIdentity(geo, seed, override_prefix=0,
    override_window=cell.L)``.
    """
    return h1_config.h1_matrix_geometry(cell.L, cell.identity_mode)


__all__ = [
    "MatrixCell",
    "MATRIX_CELLS",
    "PRIMARY_MODES",
    "BASELINE_CELL_ID",
    "validate_matrix_cells",
    "cell_by_id",
    "primary_cells",
    "control_cells",
    "baseline_cell",
    "cell_geometry",
]
