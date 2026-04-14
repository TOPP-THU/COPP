//! Shared Clarabel constraint assembly for third-order path-parameterization backends.
//!
//! # Method identity
//! This module is shared by:
//! - **Time-Optimal Path Parameterization (TOPP3)** solvers,
//! - **Convex-Objective Path Parameterization (COPP3)** solvers.
//!
//! # Scope
//! This module contains reusable routines that convert third-order path constraints into
//! Clarabel triplet form `(row, col, val, b, cones)`.
//!
//! # Constraint convention
//! Constraints are assembled in Clarabel form:
//! - `s = b - A*x in K`
//! - equivalent implementation view: `A*x - b = -s`.

use crate::copp::clarabel_backend::ConstraintsClarabel;
use crate::copp::constraints::Constraints;
use crate::copp::copp3::formulation::Topp3Problem;
use crate::diag::{CoppError, Verboser, Verbosity};
use clarabel::solver::{NonnegativeConeT, SupportedConeT, ZeroConeT};
use core::f64;

/// Assemble the standard TOPP3 feasibility constraints into Clarabel triplet buffers.
///
/// # Included blocks
/// 1. Boundary equalities for `a` and `b` at start/end samples.
/// 2. Dynamic equalities for stationary and moving intervals.
/// 3. First-order bounds (including stationary-endpoint handling).
/// 4. Second-order acceleration bounds.
/// 5. Third-order jerk-linear bounds.
/// 6. Final nonnegative cone packing for all inequality rows added after dynamic equalities.
///
/// # Logging behavior
/// - `Debug`: emits stage-by-stage matrix/vector/cone sizes.
/// - `Trace`: no extra output in this function yet; Trace currently inherits Debug visibility.
pub(crate) fn clarabel_standard_constraint_topp3(
    problem: &Topp3Problem,
    s: &[f64],
    constraints: ConstraintsClarabel,
    num_stationary: (usize, usize),
    verboser: &impl Verboser,
) -> Result<(), CoppError> {
    let n = problem.a_linearization.len() - 1;
    // s=b-A*x \in cone, where A[row[i],col[i]]=val[i], A \in R^{m*(n+1)}, b \in R^m, s \in R^m
    // -s=-b+A*x
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
                "clarabel_standard_constraint_topp3[{phase}]: row={}, col={}, val={}, b={}, cones={}",
                row.len(),
                col.len(),
                val.len(),
                b.len(),
                cones.len()
            );
        }
    };
    // Step 1. Boundary constraints (num_val==4; num_b==4; num_cone==0 here)
    clarabel_boundary_constraint_topp3(
        n,
        problem.a_boundary,
        problem.b_boundary,
        (row, col, val, b, cones),
        false,
    );
    report_counts("boundary", row, col, val, b, cones);
    // Step 2. Dynamic constraints (num_val<=4*n; num_b<=n; num_cone==1 after dynamic cone packing)
    clarabel_dynamic_constraint_topp3(s, num_stationary, (row, col, val, b, cones), false);
    cones.push(ZeroConeT(b.len()));
    report_counts("dynamic+zero_cone", row, col, val, b, cones);
    let n_b_old = b.len();
    // Step 3. first-order constraints
    // num_val<=2*(n-1), num_b<=2*(n-1), num_cone<=1 (if packed independently)
    if !clarabel_1order_constraint_topp3(
        problem,
        s,
        num_stationary,
        (row, col, val, b, cones),
        false,
    ) {
        return Err(CoppError::Infeasible(
            "clarabel_standard_constraint_topp3".into(),
            "Failed to add first-order constraints.".into(),
        ));
    }
    report_counts("first_order", row, col, val, b, cones);
    // Step 4. second-order constraints
    // num_val<=2*count_acc_constraints, num_b<=count_acc_constraints, num_cone<=1 (if packed independently)
    clarabel_2order_constraint_topp3(problem, s, num_stationary, (row, col, val, b, cones), false);
    report_counts("second_order", row, col, val, b, cones);
    // Step 5. third-order constraints
    // num_val<=6*count_jerk_constraints, num_b<=2*count_jerk_constraints, num_cone<=1 (if packed independently)
    clarabel_3order_constraint_topp3(problem, s, num_stationary, (row, col, val, b, cones), false);
    cones.push(NonnegativeConeT(b.len() - n_b_old));
    report_counts("third_order+final_nonnegative", row, col, val, b, cones);
    Ok(())
}

/// Append third-order jerk-linear constraints.
///
/// Added inequalities are generated per stage from
/// `jerk_a_linear * a + jerk_b * b + jerk_c * c <= jerk_max_linear`
/// using the finite-difference coupling between `c` and adjacent `b`.
///
/// Upper-bound estimate: `num_val<=6*count_jerk_constraints`,
/// `num_b<=2*count_jerk_constraints`, `num_cone<=1`.
///
/// If `modify_cones=true`, a single `NonnegativeConeT` is appended for all added rows.
fn clarabel_3order_constraint_topp3(
    problem: &Topp3Problem,
    s: &[f64],
    num_stationary: (usize, usize),
    constraints: ConstraintsClarabel,
    modify_cones: bool,
) {
    let (row, col, val, b, cones) = constraints;
    let n_b_old = b.len();
    let n = s.len() - 1;

    for k in (num_stationary.0)..=(n - num_stationary.1) {
        let (jerk_a_linear, jerk_b, jerk_c, jerk_max_linear) = problem
            .constraints
            .jerk_linear_constraints_unchecked(problem.idx_s_start + k);
        let nrows = jerk_a_linear.nrows();

        if k > num_stationary.0 {
            // jerk_a_linear * a[k] + jerk_b * b[k] + jerk_c * c[k-1] <= jerk_max_linear
            // jerk_a_linear * a[k] + jerk_b * b[k] + jerk_c / ds * (b[k] - b[k-1]) <= jerk_max_linear
            // jerk_a_linear * a[k] - jerk_c / ds * b[k-1] + (jerk_b + jerk_c / ds) * b[k] <= jerk_max_linear
            // A*x-b = -s = jerk_a_linear * x[k] - jerk_ds_down * x[n+k] + (jerk_b + jerk_ds_down) * b[n+k+1] - jerk_max_linear <= 0
            let ds_down = 1.0 / (s[k] - s[k - 1]);
            row.extend(b.len()..(b.len() + nrows));
            row.extend(b.len()..(b.len() + nrows));
            row.extend(b.len()..(b.len() + nrows));
            col.resize(col.len() + nrows, k);
            col.resize(col.len() + nrows, n + k);
            col.resize(col.len() + nrows, n + k + 1);
            val.extend(jerk_a_linear.iter());
            val.extend(jerk_c.iter().map(|&c_coeff| -c_coeff * ds_down));
            val.extend(
                jerk_b
                    .iter()
                    .zip(jerk_c.iter())
                    .map(|(&b_coeff, &c_coeff)| b_coeff + c_coeff * ds_down),
            );
            b.extend(jerk_max_linear.iter());
        }

        if k < n - num_stationary.1 {
            // jerk_a_linear * a[k] + jerk_b * b[k] + jerk_c * c[k] <= jerk_max_linear
            // jerk_a_linear * a[k] + jerk_b * b[k] + jerk_c / ds * (b[k+1] - b[k]) <= jerk_max_linear
            // jerk_a_linear * a[k] + (jerk_b - jerk_c / ds) * b[k] + jerk_c / ds * b[k+1] <= jerk_max_linear
            // A*x-b = -s = jerk_a_linear * x[k] + (jerk_b - jerk_c / ds) * x[n+k+1] + jerk_c / ds * x[n+k+2] - jerk_max_linear <= 0
            let ds_down = 1.0 / (s[k + 1] - s[k]);
            row.extend(b.len()..(b.len() + nrows));
            row.extend(b.len()..(b.len() + nrows));
            row.extend(b.len()..(b.len() + nrows));
            col.resize(col.len() + nrows, k);
            col.resize(col.len() + nrows, n + k + 1);
            col.resize(col.len() + nrows, n + k + 2);
            val.extend(jerk_a_linear.iter());
            val.extend(
                jerk_b
                    .iter()
                    .zip(jerk_c.iter())
                    .map(|(&b_coeff, &c_coeff)| b_coeff - c_coeff * ds_down),
            );
            val.extend(jerk_c.iter().map(|&c_coeff| c_coeff * ds_down));
            b.extend(jerk_max_linear.iter());
        }
    }

    if modify_cones {
        cones.push(NonnegativeConeT(b.len() - n_b_old));
    }
}

/// Append second-order acceleration constraints on moving intervals.
///
/// Added inequalities follow
/// `acc_a * a[k] + acc_b * b[k] <= acc_max`.
///
/// Upper-bound estimate: `num_val<=2*count_acc_constraints`,
/// `num_b<=count_acc_constraints`, `num_cone<=1`.
///
/// If `modify_cones=true`, a single `NonnegativeConeT` is appended for all added rows.
fn clarabel_2order_constraint_topp3(
    problem: &Topp3Problem,
    s: &[f64],
    num_stationary: (usize, usize),
    constraints: ConstraintsClarabel,
    modify_cones: bool,
) {
    let (row, col, val, b, cones) = constraints;
    let n_b_old = b.len();
    let n = s.len() - 1;
    for k in (num_stationary.0 + 1)..(n - num_stationary.1) {
        let (acc_a, acc_b, acc_max) = problem
            .constraints
            .acc_constraints_unchecked(problem.idx_s_start + k);
        let nrows = acc_a.nrows();
        // acc_a * a[k] + acc_b * b[k] <= acc_max
        // A*x-b = -s = acc_a * a[k] + acc_b * b[k] - acc_max = acc_a * x[k] + acc_b * x[n+k+1] - acc_max <= 0
        row.extend(b.len()..(b.len() + nrows));
        row.extend(b.len()..(b.len() + nrows));
        col.resize(col.len() + nrows, k);
        col.resize(col.len() + nrows, n + k + 1);
        val.extend(acc_a.iter());
        val.extend(acc_b.iter());
        b.extend(acc_max.iter());
    }
    if modify_cones {
        cones.push(NonnegativeConeT(b.len() - n_b_old));
    }
}

/// Append first-order bounds with stationary-endpoint handling.
///
/// Added inequalities include:
/// - stationary-endpoint bounds (when stationary segments exist at start/end),
/// - interior lower bounds `a[k] >= 0`,
/// - interior upper bounds `a[k] <= amax[k]` (only when `amax[k]` is finite).
///
/// Upper-bound estimate: `num_val<=2*(n-1)`, `num_b<=2*(n-1)`, `num_cone<=1`.
///
/// Returns `false` when stationary-endpoint bounds are infeasible.
///
/// If `modify_cones=true`, a single `NonnegativeConeT` is appended for all added rows.
fn clarabel_1order_constraint_topp3(
    problem: &Topp3Problem,
    s: &[f64],
    num_stationary: (usize, usize),
    constraints: ConstraintsClarabel,
    modify_cones: bool,
) -> bool {
    let (row, col, val, b, cones) = constraints;
    let n_b_old = b.len();
    let n = s.len() - 1;
    // num_val<=2*num_stationary.0; num_b<=2*num_stationary.0
    if num_stationary.0 > 0 {
        let (amax_stationary, amin_stationary) =
            problem.constraints.stationary_constraint_topp3::<false>(
                problem.a_linearization,
                problem.idx_s_start,
                num_stationary.0,
            );
        // amin_stationary <= a[num_stationary.0] <= amax_stationary
        if amax_stationary < amin_stationary {
            // Infeasible
            return false;
        }
        if amax_stationary.is_finite() {
            // A*x-b = -s = 1*a[num_stationary.0] - amax_stationary = 1*x[num_stationary.0] - amax_stationary <= 0
            row.push(b.len());
            col.push(num_stationary.0);
            val.push(1.0);
            b.push(amax_stationary);
        }
        // A*x-b = -s = -1*a[num_stationary.0] + amin_stationary = -1*x[num_stationary.0] + amin_stationary <= 0
        row.push(b.len());
        col.push(num_stationary.0);
        val.push(-1.0);
        b.push(-amin_stationary);
    }
    // num_val<=2*num_stationary.0; num_b<=2*num_stationary.0
    if num_stationary.1 > 0 {
        let (amax_stationary, amin_stationary) =
            problem.constraints.stationary_constraint_topp3::<true>(
                problem.a_linearization,
                problem.idx_s_start,
                num_stationary.1,
            );
        // amin_stationary <= a[n - num_stationary.1] <= amax_stationary
        if amax_stationary < amin_stationary {
            // Infeasible
            return false;
        }
        let id_a = n - num_stationary.1;
        if amax_stationary.is_finite() {
            // A*x-b = -s = 1*a[n - num_stationary.1] - amax_stationary = 1*x[id_a] - amax_stationary <= 0
            row.push(b.len());
            col.push(id_a);
            val.push(1.0);
            b.push(amax_stationary);
        }
        // A*x-b = -s = -1*a[n - num_stationary.1] + amin_stationary = -1*x[id_a] + amin_stationary <= 0
        row.push(b.len());
        col.push(id_a);
        val.push(-1.0);
        b.push(-amin_stationary);
    }
    // 0<= a[k] <= amax[k] for k in (num_stationary.0+1) .. (n - num_stationary.1)
    // A*x-b = -s = 1*a[k] - amax[k] = 1*x[k] - amax[k] <= 0
    // num_val<=2*(n - num_stationary.0 - num_stationary.1 - 1)
    // num_b<=2*(n - num_stationary.0 - num_stationary.1 - 1)
    // A*x-b = -s = -1*a[k] <= 0
    let n_b_here = n - num_stationary.1 - num_stationary.0 - 1;
    row.extend(b.len()..(b.len() + n_b_here));
    col.extend((num_stationary.0 + 1)..(n - num_stationary.1));
    val.resize(val.len() + n_b_here, -1.0);
    b.resize(b.len() + n_b_here, 0.0);
    // A*x-b = -s = 1*a[k] - amax[k] <= 0
    let amax_constraint: Vec<(usize, f64)> = ((num_stationary.0 + 1)..(n - num_stationary.1))
        .filter_map(|k| {
            let amax_curr = problem.constraints.amax_unchecked(problem.idx_s_start + k);
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
    // clarabel_1order_constraint_topp2(
    //     constraints,
    //     idx_s_start + num_stationary.0,
    //     n - num_stationary.0 - num_stationary.1,
    //     (row, col, val, b, cones),
    //     false,
    // );

    if modify_cones {
        cones.push(NonnegativeConeT(b.len() - n_b_old));
    }
    true
}

/// Append boundary equalities for `a` and `b` as Clarabel zero-cone rows.
///
/// Upper-bound estimate: `num_val==4`, `num_b==4`, `num_cone<=1`.
///
/// If `modify_cones=true`, appends `ZeroConeT(4)` for the newly added rows.
fn clarabel_boundary_constraint_topp3(
    n: usize,
    a_boundary: (f64, f64),
    b_boundary: (f64, f64),
    constraints: ConstraintsClarabel,
    modify_cones: bool,
) {
    let (row, col, val, b, cones) = constraints;
    // A*x-b = -s = 1*a[0]-a_boundary.0 = 1*x[0]-a_boundary.0 = 0
    // A*x-b = -s = 1*a[n]-a_boundary.1 = 1*x[n]-a_boundary.1 = 0
    // A*x-b = -s = 1*b[0]-b_boundary.0 = 1*x[n+1]-b_boundary.0 = 0
    // A*x-b = -s = 1*b[n]-b_boundary.1 = 1*x[2*n+1]-b_boundary.1 = 0
    row.extend(b.len()..(b.len() + 4));
    col.extend([0, n, n + 1, 2 * n + 1]);
    val.resize(val.len() + 4, 1.0);
    b.extend([a_boundary.0, a_boundary.1, b_boundary.0, b_boundary.1]);
    if modify_cones {
        cones.push(ZeroConeT(4));
    }
}

/// Append dynamic equalities for both stationary and moving intervals.
///
/// Upper-bound estimate: `num_val<=4*n`, `num_b<=n`, `num_cone<=1`.
///
/// If `modify_cones=true`, appends one `ZeroConeT` that covers all rows added by this routine.
fn clarabel_dynamic_constraint_topp3(
    s: &[f64],
    num_stationary: (usize, usize),
    constraints: ConstraintsClarabel,
    modify_cones: bool,
) {
    let (row, col, val, b, cones) = constraints;
    let n_b_old = b.len();
    if num_stationary.0 > 0 {
        // num_val==2; num_b==1
        clarabel_dynamic_stationary_topp3(s, num_stationary.0, false, (row, col, val, b, cones));
    }
    if num_stationary.1 > 0 {
        // num_val==2; num_b==1
        clarabel_dynamic_stationary_topp3(s, num_stationary.1, true, (row, col, val, b, cones));
    }
    // num_val==4*(n-num_stationary.0-num_stationary.1)
    // num_b==(n - num_stationary.0 - num_stationary.1)
    clarabel_dynamic_moving_topp3(s, num_stationary, (row, col, val, b, cones));
    if modify_cones {
        cones.push(ZeroConeT(b.len() - n_b_old));
    }
}

/// Append dynamic equality for one stationary boundary interval.
///
/// Upper-bound estimate: `num_val==2`, `num_b==1`, `num_cone==0`.
fn clarabel_dynamic_stationary_topp3(
    s: &[f64],
    num_stationary: usize,
    reverse: bool,
    constraints: ConstraintsClarabel,
) {
    let (row, col, val, b, _) = constraints;
    let n = s.len() - 1;
    // 3*(s[idx_s_start - num_stationary] - s[idx_s_start])* b[n - num_stationary] == 2* a[n - num_stationary] (reverse==true)
    // 3*(s[idx_s_start + num_stationary] - s[idx_s_start])* b[num_stationary] == 2* a[num_stationary] (reverse==false)
    let (id_a, s_start) = if reverse {
        (n - num_stationary, s.last().unwrap())
    } else {
        (num_stationary, s.first().unwrap())
    };
    // a[num_stationary] - 1.5*(s[id_a] - s_start)* b[num_stationary] == 0
    let coeff = -1.5 * (s[id_a] - s_start);
    // A*x-b = -s = 1*a[id_a] + coeff*b[id_a] = 1*x[id_a] + coeff*x[n+id_a+1] = 0
    row.resize(row.len() + 2, b.len());
    col.extend([id_a, n + 1 + id_a]);
    val.extend([1.0, coeff]);
    b.push(0.0);
}

/// Append dynamic equalities for moving intervals.
///
/// This routine encodes the finite-difference identity
/// `a[k] - a[k+1] + ds[k]*b[k] + ds[k]*b[k+1] = 0`.
///
/// Upper-bound estimate: `num_val==4*(n-num_stationary.0-num_stationary.1)`,
/// `num_b==(n-num_stationary.0-num_stationary.1)`, `num_cone==0`.
fn clarabel_dynamic_moving_topp3(
    s: &[f64],
    num_stationary: (usize, usize),
    constraints: ConstraintsClarabel,
) {
    let (row, col, val, b, _) = constraints;
    // b[k+1] - b[k] = c[k] * ds[k]
    // a[k+1] - a[k] = 2.0 * b[k] * ds[k] + c[k] * ds[k]^2 = 2.0 * b[k] * ds[k] + (b[k+1]-b[k]) * ds[k]
    // a[k] - a[k+1] + ds[k]*b[k] + ds[k]*b[k+1] = 0
    // A*x-b = -s = x[k] - x[k+1] + ds[k]*x[n+k+1] + ds[k]*b[n+k+2] = 0
    // for k in num_stationary.0 .. n - num_stationary.1
    let n = s.len() - 1;
    let n_b_here = n - num_stationary.0 - num_stationary.1;
    row.extend(b.len()..(b.len() + n_b_here));
    row.extend(b.len()..(b.len() + n_b_here));
    row.extend(b.len()..(b.len() + n_b_here));
    row.extend(b.len()..(b.len() + n_b_here));
    col.extend(num_stationary.0..(num_stationary.0 + n_b_here)); // < n - num_stationary.1
    col.extend((num_stationary.0 + 1)..(num_stationary.0 + n_b_here + 1)); // < n - num_stationary.1 + 1
    col.extend((n + num_stationary.0 + 1)..(n + num_stationary.0 + n_b_here + 1)); // < 2 * n - num_stationary.1
    col.extend((n + num_stationary.0 + 2)..(n + num_stationary.0 + n_b_here + 2)); // < 2 * n - num_stationary.1 + 1
    val.resize(val.len() + n_b_here, 1.0);
    val.resize(val.len() + n_b_here, -1.0);
    val.extend(
        s.windows(2)
            .skip(num_stationary.0)
            .take(n_b_here)
            .map(|s_pair| s_pair[1] - s_pair[0]),
    );
    val.extend(
        s.windows(2)
            .skip(num_stationary.0)
            .take(n_b_here)
            .map(|s_pair| s_pair[1] - s_pair[0]),
    );
    b.resize(b.len() + n_b_here, 0.0);
}

/// Estimate allocation capacities for standard TOPP3 feasibility constraints.
///
/// Returns `(capacity_val, capacity_b, capacity_cones)` where:
/// - `capacity_val`: estimated nonzero triplets in `A`;
/// - `capacity_b`: estimated row count (`b.len()`);
/// - `capacity_cones`: estimated cone-block count.
#[inline]
pub(crate) fn clarabel_standard_capacity_topp3(
    constraints: &Constraints,
    idx_s_interval: (usize, usize),
) -> (usize, usize, usize) {
    let (idx_s_start, idx_s_final) = idx_s_interval;
    let n = idx_s_final - idx_s_start;
    let count_acc_constraints = constraints.count_rows_acc(idx_s_start, idx_s_final + 1);
    let count_jerk_constraints = constraints.count_rows_jerk(idx_s_start, idx_s_final + 1);
    // Step 1. Boundary constraints (num_val==4; num_b==4)
    // Step 2. Dynamic constraints (num_val<=4*n; num_b<=n)
    // Step 3. first-order constraints
    // num_val<=2*(n-1), num_b<=2*(n-1)
    // Step 4. second-order constraints
    // num_val<=2*count_acc_constraints, num_b<=count_acc_constraints
    // Step 5. third-order constraints
    // num_val<=6*count_jerk_constraints, num_b<=2*count_jerk_constraints
    // Step 6. cone packing
    // num_cone==2 (1 x ZeroCone for equalities + 1 x NonnegativeCone for inequalities)
    let capacity_val = 2 + 6 * n + 2 * count_acc_constraints + 6 * count_jerk_constraints;
    let capacity_b = 2 + 3 * n + count_acc_constraints + 2 * count_jerk_constraints;
    (capacity_val, capacity_b, 2)
}
