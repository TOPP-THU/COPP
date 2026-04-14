//! Shared Clarabel constraint assembly for second-order path-parameterization backends.
//!
//! # Method identity
//! This module is shared by:
//! - **Time-Optimal Path Parameterization (TOPP2)** solvers,
//! - **Convex-Objective Path Parameterization (COPP2)** solvers.
//!
//! # Scope
//! This module contains reusable routines that convert second-order path constraints into
//! Clarabel triplet form `(row, col, val, b, cones)`.
//!
//! # Constraint convention
//! Constraints are assembled in Clarabel form:
//! - `s = b - A*x in K`
//! - equivalent implementation view: `A*x - b = -s`.

use crate::copp::clarabel_backend::ConstraintsClarabel;
use crate::copp::constraints::Constraints;
use crate::copp::copp2::formulation::Topp2Problem;
use crate::diag::{Verboser, Verbosity};
use clarabel::solver::{NonnegativeConeT, SupportedConeT, ZeroConeT};
use core::f64;

/// Assemble the standard TOPP2 feasibility constraints into Clarabel triplet buffers.
///
/// # Included blocks
/// 1. Boundary equalities: `a[0] = a_start`, `a[n] = a_final`.
/// 2. First-order bounds for interior nodes.
/// 3. Second-order acceleration bounds using finite-difference coupling.
/// 4. Final nonnegative cone packing for all inequality rows added after boundary equalities.
///
/// # Logging behavior
/// - `Debug`: emits stage-by-stage matrix/vector/cone sizes.
/// - `Trace`: no extra output in this function yet; Trace currently inherits Debug visibility.
pub(crate) fn clarabel_standard_constraint_topp2(
    problem: &Topp2Problem,
    constraints: ConstraintsClarabel,
    verboser: &impl Verboser,
) {
    let (idx_s_start, idx_s_final) = problem.idx_s_interval;
    let (a_start, a_final) = problem.a_boundary;
    let n = idx_s_final - idx_s_start;
    let (row, col, val, b, cones) = constraints;

    let report_counts = |phase: &str,
                         row: &Vec<usize>,
                         col: &Vec<usize>,
                         val: &Vec<f64>,
                         b: &Vec<f64>,
                         cones: &Vec<SupportedConeT<f64>>| {
        if verboser.is_enabled(Verbosity::Debug) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "clarabel_standard_constraint_topp2[{phase}]: row={}, col={}, val={}, b={}, cones={}",
                row.len(),
                col.len(),
                val.len(),
                b.len(),
                cones.len()
            );
        }
    };

    // Step 1.1 initial constraints (num_val<=2, num_b<=2, num_cone<=1)
    clarabel_a_point_constraint_topp2(a_start, 0, (row, col, val, b, cones));
    report_counts("a_start", row, col, val, b, cones);
    // Step 1.2 terminal constraints (num_val<=2, num_b<=2, num_cone<=1)
    clarabel_a_point_constraint_topp2(a_final, n, (row, col, val, b, cones));
    report_counts("a_final", row, col, val, b, cones);
    let n_rows_old = b.len();
    // Step 1.3 first-order constraints (num_val<=2*(n-1), num_b<=2*(n-1), num_cone<=1 if packed independently)
    // A*x-b = -s = -1*a[k] <= 0
    // A*x-b = -s = 1*a[k] - amax[k] <= 0
    clarabel_1order_constraint_topp2(
        problem.constraints,
        n,
        idx_s_start,
        (row, col, val, b, cones),
        false,
    );
    report_counts("first_order", row, col, val, b, cones);
    // Step 1.4 second-order constraints (num_val<=4*count_acc_constraints, num_b<=2*count_acc_constraints, num_cone<=1 if packed independently)
    // acc_a * a[k] + acc_b * b[k-1] <= acc_max
    // acc_a * a[k] + acc_b * b[k] <= acc_max
    clarabel_2order_constraint_topp2(
        problem.constraints,
        idx_s_start,
        n,
        (row, col, val, b, cones),
        false,
    );
    report_counts("second_order", row, col, val, b, cones);
    cones.push(NonnegativeConeT(b.len() - n_rows_old));
    report_counts("final_nonnegative_cone", row, col, val, b, cones);
}

/// Estimate allocation capacities for standard TOPP2 feasibility constraints.
///
/// Returns `(capacity_val, capacity_b, capacity_cones)` where:
/// - `capacity_val`: estimated nonzero triplets in `A`;
/// - `capacity_b`: estimated row count (`b.len()`);
/// - `capacity_cones`: cone blocks count (boundary zero-cones + inequality nonnegative-cone).
#[inline]
pub(crate) fn clarabel_standard_capacity_topp2(
    constraints: &Constraints,
    idx_s_interval: (usize, usize),
) -> (usize, usize, usize) {
    let (idx_s_start, idx_s_final) = idx_s_interval;
    let n = idx_s_final - idx_s_start;
    let count_acc_constraints = constraints.count_rows_acc(idx_s_start, idx_s_final + 1);
    // Step 1. Boundary constraints (num_val<=4; num_b<=4; num_cone==2)
    // Step 2. first-order constraints
    // num_val<=2*(n-1), num_b<=2*(n-1)
    // Step 3. second-order constraints
    // num_val<=4*count_acc_constraints, num_b<=2*count_acc_constraints
    // Step 4. final inequality cone packing (num_cone==1)
    let capacity_val = 4 * count_acc_constraints + 2 * (n + 1);
    let capacity_b = 2 * count_acc_constraints + 2 * (n + 1);
    (capacity_val, capacity_b, 3)
}

/// Add one boundary equality `a[id_col] = a_boundary` as a Clarabel `ZeroConeT(1)` row.
///
/// Note: this function always appends data and does not perform feasibility checks.
fn clarabel_a_point_constraint_topp2(
    a_boundary: f64,
    id_col: usize,
    constraints: ConstraintsClarabel,
) {
    let (row, col, val, b, cones) = constraints;
    // A*x-b = -s = 1*a[id_col] - a = 0
    row.push(b.len());
    col.push(id_col);
    val.push(1.0);
    b.push(a_boundary);
    cones.push(ZeroConeT(1));
}

/// Append first-order bounds for interior samples `k=1..n-1`.
///
/// Added inequalities:
/// - lower bound: `a[k] >= 0`;
/// - upper bound: `a[k] <= amax[k]` (only when `amax[k]` is finite).
///
/// Upper-bound estimate: `num_val<=2*(n-1)`, `num_b<=2*(n-1)`, `num_cone<=1`.
///
/// If `modify_cones=true`, a single `NonnegativeConeT` is appended for all added rows.
fn clarabel_1order_constraint_topp2(
    constraints: &Constraints,
    n: usize,
    idx_s_start: usize,
    input: ConstraintsClarabel,
    modify_cones: bool,
) {
    let (row, col, val, b, cones) = input;
    // A*x-b = -s = -1*a[k] <= 0
    row.extend((b.len() + 1)..(b.len() + n));
    col.extend(1..n);
    val.resize(val.len() + n - 1, -1.0);
    b.resize(b.len() + n - 1, 0.0);
    // A*x-b = -s = 1*a[k] - amax[k] <= 0
    let amax_constraint: Vec<(usize, f64)> = (1..n)
        .filter_map(|k| {
            let amax_curr = constraints.amax_unchecked(idx_s_start + k);
            if amax_curr.is_finite() {
                Some((k, amax_curr))
            } else {
                None
            }
        })
        .collect();
    row.extend(b.len()..(b.len() + amax_constraint.len()));
    col.extend(amax_constraint.iter().map(|(col_, _)| *col_));
    val.resize(val.len() + amax_constraint.len(), 1.0);
    b.extend(amax_constraint.iter().map(|(_, b_)| *b_));
    if modify_cones {
        cones.push(NonnegativeConeT(n - 1 + amax_constraint.len()));
    }
}

/// Append second-order acceleration constraints across all stages `k=0..n`.
///
/// This routine converts constraints involving `(a[k], b[k-1])` / `(a[k], b[k])` into
/// linear inequalities over adjacent `a` entries via finite-difference substitutions.
///
/// Upper-bound estimate: `num_val<=4*count_acc_constraints`,
/// `num_b<=2*count_acc_constraints`, `num_cone<=1`.
///
/// If `modify_cones=true`, a single `NonnegativeConeT` is appended for all added rows.
fn clarabel_2order_constraint_topp2(
    constraints: &Constraints,
    idx_s_start: usize,
    n: usize,
    input: ConstraintsClarabel,
    modify_cones: bool,
) {
    let (row, col, val, b, cones) = input;
    let n_b_old = b.len();
    for k in 0..=n {
        let (acc_a, acc_b, acc_max) = constraints.acc_constraints_unchecked(idx_s_start + k);
        let nrows = acc_a.nrows();
        if k > 0 {
            // acc_a * a[k] + acc_b * b[k-1] <= acc_max
            // acc_a * a[k] + acc_b * (a[k] - a[k-1]) / ds_double <= acc_max
            // A*x-b = -s = (acc_a + acc_b / ds_double) * a[k] - acc_b / ds_double * a[k-1] - acc_max <= 0
            let ds_double_reciprocal = 0.5
                / (constraints.s_unchecked(idx_s_start + k)
                    - constraints.s_unchecked(idx_s_start + k - 1));
            row.extend(b.len()..(b.len() + nrows));
            row.extend(b.len()..(b.len() + nrows));
            // row.extend((b.len()..(b.len() + ncols)).flat_map(|r| [r, r]));
            col.resize(col.len() + nrows, k - 1);
            col.resize(col.len() + nrows, k);
            // col.extend(std::iter::repeat([k - 1, k]).take(ncols).flatten());
            val.extend(acc_b.iter().map(|&b_coeff| -b_coeff * ds_double_reciprocal));
            val.extend(
                acc_a
                    .iter()
                    .zip(acc_b.iter())
                    .map(|(&a_coeff, &b_coeff)| a_coeff + b_coeff * ds_double_reciprocal),
            );
            b.extend(acc_max.iter());
        }
        if k < n {
            // acc_a * a[k] + acc_b * b[k] <= acc_max
            // acc_a * a[k] + acc_b * (a[k+1] - a[k]) / ds_double <= acc_max
            // A*x-b = -s = (acc_a - acc_b / ds_double) * a[k] + acc_b / ds_double * a[k+1] - acc_max <= 0
            let ds_double_reciprocal = 0.5
                / (constraints.s_unchecked(idx_s_start + k + 1)
                    - constraints.s_unchecked(idx_s_start + k));
            row.extend(b.len()..(b.len() + nrows));
            row.extend(b.len()..(b.len() + nrows));
            // row.extend((b.len()..(b.len() + ncols)).flat_map(|r| [r, r]));
            col.resize(col.len() + nrows, k);
            col.resize(col.len() + nrows, k + 1);
            // col.extend(std::iter::repeat([k, k + 1]).take(ncols).flatten());
            val.extend(
                acc_a
                    .iter()
                    .zip(acc_b.iter())
                    .map(|(&a_coeff, &b_coeff)| a_coeff - b_coeff * ds_double_reciprocal),
            );
            val.extend(acc_b.iter().map(|&b_coeff| b_coeff * ds_double_reciprocal));
            b.extend(acc_max.iter());
        }
    }
    if modify_cones {
        cones.push(NonnegativeConeT(b.len() - n_b_old));
    }
}
