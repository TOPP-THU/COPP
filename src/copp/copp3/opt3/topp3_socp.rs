//! 3rd-order Time-Optimal Path Parameterization (TOPP3) based on second-order cone programming (SOCP).
//!
//! # Method identity
//! This module implements the **optimization backend** for TOPP3-QP by transforming
//! third-order path-parameterization constraints/objective into Clarabel-compatible
//! conic form and solving with SOCP.
//!
//! # Discrete variables (local notation)
//! On a path grid `s[0..=n]`:
//! - `a[k]` denotes $\dot{s}_k^2$;
//! - `b[k]` denotes $\ddot{s}_k$;
//! - auxiliary variables `xi[k]` and `eta[k]` satisfy reciprocal-SOC coupling for
//!   the time objective in QP form.
//! - decision vector is organized as
//!   `x = [a[0..=n], b[0..=n], xi[0..len_xi), eta[0..len_xi)]`.
//!
//! # High-level pipeline
//! 1. Validate boundary/index contracts.
//! 2. Assemble standard TOPP3 conic constraints.
//! 3. Add QP-specific SOC constraints for `(xi, eta)` and reciprocal coupling.
//! 4. Build sparse matrices `A`, `P`, vector `q`, and solve by Clarabel.
//! 5. Apply status acceptance policy (`ClarabelOptions::is_allow`) and extract
//!    `(a,b,num_stationary)` only when accepted.
//!
//! # API layering
//! - `topp3_socp`: strict/normal API, returns only accepted `(a,b,num_stationary)`.
//! - `topp3_socp_expert`: expert API returning `(Option<Copp3Result>, DefaultSolution<f64>)`.

use crate::copp::clarabel_backend::ConstraintsClarabel;
use crate::copp::copp3::Copp3Result;
use crate::copp::copp3::formulation::{Topp3Problem, get_weight_a_topp3};
use crate::copp::copp3::opt3::clarabel_constraints::{
    clarabel_standard_capacity_topp3, clarabel_standard_constraint_topp3,
};
use crate::copp::{ClarabelOptions, clarabel_to_copp3_solution};
use crate::diag::{
    CoppError, DebugVerboser, SilentVerboser, SummaryVerboser, TraceVerboser, Verboser, Verbosity,
    check_boundary_state_copp3_valid, check_s_interval_valid, format_duration_human,
};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{NonnegativeConeT, SecondOrderConeT};
use clarabel::solver::{DefaultSolution, DefaultSolver, IPSolver, SupportedConeT};

/// Strict TOPP3-SOCP API for production use.
///
/// # Purpose
/// Use this entry when caller only needs a valid profile `(a,b,num_stationary)` and treats
/// non-accepted solver statuses as hard failures.
///
/// # Contract
/// - Internally calls [`topp3_socp_expert`].
/// - Returns `Ok((a,b,num_stationary))` **iff** `options.is_allow(solution.status)` is `true`.
/// - Returns `Err(CoppError::ClarabelSolverStatus(...))` when status is not accepted.
///
/// # Returns
/// Returns accepted TOPP3 profile `(a, b, num_stationary)`.
///
/// # Errors
/// Returns [`CoppError`] on conic-model/solver failures and non-accepted solver status.
///
/// More details are provided in the documentation of [`topp3_socp_expert`].
pub fn topp3_socp(
    problem: &Topp3Problem,
    options: &ClarabelOptions,
) -> Result<Copp3Result, CoppError> {
    let (result, solution) = topp3_socp_expert(problem, options)?;
    result.ok_or_else(|| CoppError::ClarabelSolverStatus("topp3_socp".into(), solution.status))
}

/// Expert TOPP3-SOCP API with full Clarabel solution exposure.
///
/// # Return contract
/// - `Ok((Some(result), solution))`: status accepted by `options.is_allow(solution.status)`.
/// - `Ok((None, solution))`: solve finished but status not accepted.
/// - `Err(...)`: input/model/solver-construction runtime failures.
///
/// # Returns
/// Returns tuple `(Option<Copp3Result>, DefaultSolution<f64>)` for diagnostic pipelines.
///
/// # Errors
/// Returns [`CoppError`] only for true build/runtime failures.
///
/// # Contract
/// - caller handles `None` profile when status is not accepted;
/// - acceptance policy is controlled by `options.is_allow`.
///
/// # Verbosity behavior
/// Logging is layered by `options.verbosity()`:
/// - [`Silent`](Verbosity::Silent): no algorithm logs;
/// - [`Summary`](Verbosity::Summary): lifecycle milestones and elapsed time;
/// - [`Debug`](Verbosity::Debug): assembly-level counters and stage summaries;
/// - [`Trace`](Verbosity::Trace): fine-grained stage deltas and solver snapshot diagnostics.
pub fn topp3_socp_expert(
    problem: &Topp3Problem,
    options: &ClarabelOptions,
) -> Result<(Option<Copp3Result>, DefaultSolution<f64>), CoppError> {
    match options.verbosity() {
        Verbosity::Silent => topp3_socp_core(problem, (options, SilentVerboser)),
        Verbosity::Summary => topp3_socp_core(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => topp3_socp_core(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => topp3_socp_core(problem, (options, TraceVerboser::new())),
    }
}

/// Core implementation for TOPP3-SOCP expert flow.
///
/// # Internal contract
/// `options_verboser` packs:
/// - `options`: acceptance policy and Clarabel numerical settings;
/// - `verboser`: concrete logger implementation chosen by external verbosity dispatch.
///
/// # Invariants
/// - decision-variable layout always starts with contiguous `a[0..=n]` and `b[0..=n]`;
/// - auxiliary block `[xi, eta]` has shared length `length_xi_eta(n, num_stationary)`;
/// - extracted `(a,b)` is produced only through `clarabel_to_copp3_solution` when status is accepted.
fn topp3_socp_core(
    problem: &Topp3Problem,
    options_verboser: (&ClarabelOptions, impl Verboser),
) -> Result<(Option<Copp3Result>, DefaultSolution<f64>), CoppError> {
    let (options, mut verboser) = options_verboser;
    let idx_s_start = problem.idx_s_start;
    let a_boundary = problem.a_boundary;
    let b_boundary = problem.b_boundary;
    let num_stationary = problem.num_stationary;
    if verboser.is_enabled(Verbosity::Summary) {
        verboser.record_start_time();
    }
    if verboser.is_enabled(Verbosity::Trace) {
        let settings = options.clarabel_settings();
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: options snapshot -> allow(almost={}, max_iter={}, max_time={}, callback_term={}, insufficient_progress={}), tol_gap_rel={}, tol_feas={}, max_iter={}, verbose={}",
            options.is_allow(clarabel::solver::SolverStatus::AlmostSolved),
            options.is_allow(clarabel::solver::SolverStatus::MaxIterations),
            options.is_allow(clarabel::solver::SolverStatus::MaxTime),
            options.is_allow(clarabel::solver::SolverStatus::CallbackTerminated),
            options.is_allow(clarabel::solver::SolverStatus::InsufficientProgress),
            settings.tol_gap_rel,
            settings.tol_feas,
            settings.max_iter,
            settings.verbose
        );
    }

    // Check input validity
    check_boundary_state_copp3_valid(a_boundary, b_boundary)?;
    let n = problem.a_linearization.len() - 1;
    let idx_s_final = idx_s_start + n;
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "\ntopp3_socp started: {} <= idx_s <= {}, s_len = {}, num_stationary={:?}.",
            idx_s_start,
            idx_s_final,
            problem.a_linearization.len(),
            num_stationary
        );
    }
    check_s_interval_valid("topp3_socp", idx_s_start, idx_s_final)?;
    let len_xi = length_xi_eta(n, num_stationary);
    let id_xi_start = 2 * (n + 1);
    let id_eta_start = id_xi_start + len_xi;
    // Let x = [a[0,1,...,n],
    //          b[0,1,...,n],
    //          xi[0,1,...,len_xi-1],
    //          eta[0,1,...,len_xi-1]]
    //       \in R^{2*(n+1)+2*len_xi}.
    // Step 1. Deal with constraints
    // s=b-A*x \in cone, where A[row[i],col[i]]=val[i], A \in R^{m*(n+1)}, b \in R^m, s \in R^m
    // -s=-b+A*x
    // Step 1.1 create constraints
    let (cap_val_lp, cap_b_lp, cap_cone_lp) =
        clarabel_standard_capacity_topp3(problem.constraints, (idx_s_start, idx_s_final));
    let (cap_val_qp, cap_b_qp, cap_cone_qp) = clarabel_capacity_topp3_qp(n);
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: capacity estimate lp(val={cap_val_lp}, b={cap_b_lp}, cone={cap_cone_lp}), qp(val={cap_val_qp}, b={cap_b_qp}, cone={cap_cone_qp}), n_var={}",
            id_eta_start + len_xi
        );
    }
    let mut cones = Vec::<SupportedConeT<f64>>::with_capacity(cap_cone_lp + cap_cone_qp);
    let mut row = Vec::<usize>::with_capacity(cap_val_lp + cap_val_qp);
    let mut col = Vec::<usize>::with_capacity(cap_val_lp + cap_val_qp);
    let mut val = Vec::<f64>::with_capacity(cap_val_lp + cap_val_qp);
    let mut b = Vec::<f64>::with_capacity(cap_b_lp + cap_b_qp);
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: allocated capacities row/col/val/b/cones <= {}/{}/{}/{}/{}",
            cap_val_lp + cap_val_qp,
            cap_val_lp + cap_val_qp,
            cap_val_lp + cap_val_qp,
            cap_b_lp + cap_b_qp,
            cap_cone_lp + cap_cone_qp
        );
    }

    // Step 1.2 deal with standard constraints
    let s = problem.constraints.s_vec(idx_s_start, idx_s_final + 1)?;
    let row_before_std = row.len();
    let col_before_std = col.len();
    let val_before_std = val.len();
    let b_before_std = b.len();
    let cones_before_std = cones.len();
    clarabel_standard_constraint_topp3(
        problem,
        &s,
        (&mut row, &mut col, &mut val, &mut b, &mut cones),
        num_stationary,
        &verboser,
    )?;
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: standard-constraints delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}",
            row.len() - row_before_std,
            col.len() - col_before_std,
            val.len() - val_before_std,
            b.len() - b_before_std,
            cones.len() - cones_before_std
        );
    }
    // Step 1.3 deal with additional constraints for QP
    let row_before_qp = row.len();
    let col_before_qp = col.len();
    let val_before_qp = val.len();
    let b_before_qp = b.len();
    let cones_before_qp = cones.len();
    clarabel_constraint_topp3_qp(
        (&mut row, &mut col, &mut val, &mut b, &mut cones),
        (idx_s_start, idx_s_final),
        num_stationary,
        id_xi_start,
        id_eta_start,
    );
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: qp-aux delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}",
            row.len() - row_before_qp,
            col.len() - col_before_qp,
            val.len() - val_before_qp,
            b.len() - b_before_qp,
            cones.len() - cones_before_qp
        );
    }

    // Step 1.4 build the constraints
    let n_var = id_eta_start + len_xi;
    let row_len = row.len();
    let col_len = col.len();
    let val_len = val.len();
    let b_len = b.len();
    let cones_len = cones.len();
    let a_csc = CscMatrix::new_from_triplets(b.len(), n_var, row, col, val);
    // Step 2. objective function (time QP surrogate): min \sum w[k] * eta[k]
    let p_object = CscMatrix::<f64>::zeros((n_var, n_var));
    let q_object = clarabel_q_object_topp3_qp(&s, num_stationary, n_var, id_eta_start);
    if verboser.is_enabled(Verbosity::Trace) {
        let (q_min, q_max) = q_object
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: matrix built with m={}, n={}, A.nnz={}, P.nnz={}, q_range=[{}, {}]",
            b_len,
            n_var,
            a_csc.nnz(),
            p_object.nnz(),
            q_min,
            q_max
        );
    }
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: ready to solve with row/col/val/b/cones = {row_len}/{col_len}/{val_len}/{b_len}/{cones_len} and n_var = {n_var}.",
        );
    }
    // Step 3. solve the SOCP problem
    let settings = options.clarabel_settings().clone();
    let mut solver = DefaultSolver::<f64>::new(&p_object, &q_object, &a_csc, &b, &cones, settings)
        .map_err(|e| CoppError::ClarabelSolverError("topp3_socp".into(), e))?;
    solver.solve();
    let solution = solver.solution;
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: solve done, status = {:?}, elapsed = {}.",
            solution.status,
            format_duration_human(verboser.elapsed())
        );
    }
    if verboser.is_enabled(Verbosity::Trace) {
        let show = solution.x.len().min(3);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: solution x_len={}, head={:?}",
            solution.x.len(),
            &solution.x[0..show]
        );
    }
    let result = if options.is_allow(solution.status) {
        let (a, b) =
            clarabel_to_copp3_solution(&solution.x.as_slice()[0..2 * (n + 1)], &s, num_stationary);
        Some((a, b, num_stationary))
    } else {
        None
    };
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_socp: allow(status)={}, extracted_profile={}",
            options.is_allow(solution.status),
            if result.is_some() {
                "Some((a,b,num_stationary))"
            } else {
                "None"
            }
        );
    }
    Ok((result, solution))
}

/// Determine the length of `xi` and `eta` in the decision variable `x`.
#[inline(always)]
fn length_xi_eta(n: usize, num_stationary: (usize, usize)) -> usize {
    n + 1 - num_stationary.0.max(1) - num_stationary.1.max(1)
}

/// Return `k_skip`, where `eta[k] = 1/sqrt(a[k + k_skip])`.
#[inline(always)]
fn skip_a_for_xi(num_stationary_start: usize) -> usize {
    num_stationary_start.max(1)
}

/// Create the constraints for clarabel TOPP3-QP.  
/// `idx_s_interval`: (idx_s_start, idx_s_final), the interval of s for which we want to compute the time-optimal profile.  
/// `num_stationary`: (num_stationary_start, num_stationary_final), the number of stationary points at the start and final of the interval.  
/// `id_xi_start`: the starting index of xi in the decision variable x.
/// `id_eta_start`: the starting index of eta in the decision variable x.
fn clarabel_constraint_topp3_qp(
    constraints: ConstraintsClarabel,
    idx_s_interval: (usize, usize),
    num_stationary: (usize, usize),
    id_xi_start: usize,
    id_eta_start: usize,
) {
    let (idx_s_start, idx_s_final) = idx_s_interval;
    let n = idx_s_final - idx_s_start;
    // s=b-A*x \in cone, where A[row[i],col[i]]=val[i]
    // -s=-b+A*x
    let (row, col, val, b, cones) = constraints;
    // Add constraints for xi and eta
    // xi[k] >= 0, eta[k] >= 0
    // norm2([2, xi[k] - eta[k]]) <= xi[k] + eta[k]
    // xi[k] * xi[k] <= a[k + k_skip]
    let len_xi = length_xi_eta(n, num_stationary);
    let k_skip = skip_a_for_xi(num_stationary.0);
    // Step 1. xi[i] >= 0
    // A*x-b = -s = -1*xi[k] <= 0
    row.extend(b.len()..(b.len() + len_xi));
    col.extend(id_xi_start..(id_xi_start + len_xi));
    val.resize(val.len() + len_xi, -1.0);
    b.resize(b.len() + len_xi, 0.0);
    // Step 2. eta[i] >= 0
    // A*x-b = -s = -1*eta[k] <= 0
    row.extend(b.len()..(b.len() + len_xi));
    col.extend(id_eta_start..(id_eta_start + len_xi));
    val.resize(val.len() + len_xi, -1.0);
    b.resize(b.len() + len_xi, 0.0);
    cones.push(NonnegativeConeT(2 * len_xi));
    // Step 3. norm2([2, xi[k] - eta[k]]) <= xi[k] + eta[k]
    // -A*x+b = s = [xi[k] + eta[k], xi[k] - eta[k], 2] \in SOC
    for k in 0..len_xi {
        // xi[k] + eta[k]
        row.resize(row.len() + 2, b.len());
        col.extend([id_xi_start + k, id_eta_start + k]);
        val.extend([-1.0, -1.0]);
        b.push(0.0);
        // xi[k] - eta[k]
        row.resize(row.len() + 2, b.len());
        col.extend([id_xi_start + k, id_eta_start + k]);
        val.extend([-1.0, 1.0]);
        b.push(0.0);
        // 2
        b.push(2.0);
    }
    // Step 4. xi[k] * xi[k] <= a[k_skip + k]
    // norm2([2*xi[k], a[k_skip + k] - 1]) <= a[k_skip + k] + 1
    // -A*x+b = s = [a[k_skip + k] + 1, a[k_skip + k] - 1, 2*xi[k]] \in SOC
    for k in 0..len_xi {
        // a[k_skip + k] + 1
        row.push(b.len());
        col.push(k_skip + k);
        val.push(-1.0);
        b.push(1.0);
        // a[k_skip + k] - 1
        row.push(b.len());
        col.push(k_skip + k);
        val.push(-1.0);
        b.push(-1.0);
        // 2*xi[k]
        row.push(b.len());
        col.push(id_xi_start + k);
        val.push(-2.0);
        b.push(0.0);
    }
    cones.resize(cones.len() + 2 * len_xi, SecondOrderConeT(3));
}

/// Build the linear objective coefficient `q` for TOPP3-QP.
#[inline(always)]
fn clarabel_q_object_topp3_qp(
    s: &[f64],
    num_stationary: (usize, usize),
    n_var: usize,
    id_eta_start: usize,
) -> Vec<f64> {
    let mut q_object = Vec::<f64>::with_capacity(n_var);
    let weight = get_weight_a_topp3(s, num_stationary);
    let len_eta = length_xi_eta(s.len() - 1, num_stationary);
    let k_skip = skip_a_for_xi(num_stationary.0);
    q_object.resize(id_eta_start, 0.0);
    q_object.extend(weight[k_skip..(k_skip + len_eta)].iter());
    q_object.resize(n_var, 0.0);
    q_object
}

/// Determine Clarabel pre-allocation capacity for TOPP3-QP auxiliary constraints.
///
/// Returns `(capacity_val, capacity_b, capacity_cones)` as upper bounds.
#[inline(always)]
fn clarabel_capacity_topp3_qp(n: usize) -> (usize, usize, usize) {
    // Step 1. xi[k] >= 0, eta[k] >= 0
    //         (num_val==2*len_xi; num_b==2*len_xi, num_cone==1)
    // Step 2. [2, xi[k] - eta[k], xi[k] + eta[k]] \in SOC
    //         (num_val==4*len_xi; num_b==3*len_xi, num_cone==len_xi)
    // Step 3. [2*xi[k], a[num_stationary.0 + k] - 1, a[num_stationary.0 + k] + 1] \in SOC
    //         (num_val==3*len_xi; num_b==3*len_xi, num_cone==len_xi)
    // len_xi = n + 1 - num_stationary.0 - num_stationary.1 <= n + 1
    let len_xi_upper_bound = n + 1;
    (
        9 * len_xi_upper_bound,
        8 * len_xi_upper_bound,
        2 * len_xi_upper_bound + 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::copp2::stable::basic::{Topp2ProblemBuilder, s_to_t_topp2};
    use crate::copp::copp2::stable::reach_set2::{ReachSet2Options, ReachSet2OptionsBuilder};
    use crate::copp::copp2::stable::topp2_ra::topp2_ra;
    use crate::copp::copp3::stable::basic::{Topp3ProblemBuilder, s_to_t_topp3};
    use crate::copp::{ClarabelOptions, ClarabelOptionsBuilder};
    use crate::path::add_symmetric_axial_limits_for_test;
    use crate::robot::robot_core::Robot;
    use crate::solver::topp3_lp::topp3_lp;
    use core::f64;
    use nalgebra::DMatrix;
    use rand::RngExt;
    use std::time::Instant;

    #[test]
    fn test_topp3_lp_qp() -> Result<(), CoppError> {
        run_test_topp3_lp_qp_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// AAverage (fail 1): tc_ra = 0.3294 ms, tc_lp = 257.8678 ms, tc_qp = 327.9065 ms, tf_ra = 6.1838, tf_lp = 7.1079, tf_qp = 7.1079
    #[test]
    #[ignore = "slow"]
    fn test_topp3_lp_qp_robust() -> Result<(), CoppError> {
        run_test_topp3_lp_qp_repeated(100, true)
    }

    fn run_one_topp3_lp_qp_case(
        options_ra: &ReachSet2Options,
        options_lp: &ClarabelOptions,
        options_qp: &ClarabelOptions,
    ) -> Result<(f64, f64, f64, f64, f64, f64), CoppError> {
        let n: usize = 1000;
        let dim = 7;
        let mut rng = rand::rng();
        let omega = (0..dim)
            .map(|_| rng.random_range(0.1..(2.0 * f64::consts::PI)))
            .collect::<Vec<f64>>();
        let phi = (0..dim)
            .map(|_| rng.random_range(0.0..(2.0 * f64::consts::PI)))
            .collect::<Vec<f64>>();

        let mut robot = Robot::with_capacity(dim, n);
        let s = DMatrix::<f64>::from_fn(1, n, |_, j| {
            (j as f64
                + (if 0 < j && 2 * j < n { 0.5 } else { 0.0 }
                    + if n > j && 2 * j > n { 0.5 } else { 0.0 })
                    * j as f64
                    / n as f64)
                * (1.0 / (n - 1) as f64)
        });
        let q = DMatrix::<f64>::from_fn(dim, n, |i, j| (omega[i] * s[j] + phi[i]).sin());
        let dq =
            DMatrix::<f64>::from_fn(dim, n, |i, j| omega[i] * (omega[i] * s[j] + phi[i]).cos());
        let ddq = DMatrix::<f64>::from_fn(dim, n, |i, j| {
            -omega[i] * omega[i] * (omega[i] * s[j] + phi[i]).sin()
        });
        let dddq = DMatrix::<f64>::from_fn(dim, n, |i, j| {
            -omega[i] * omega[i] * omega[i] * (omega[i] * s[j] + phi[i]).cos()
        });
        robot.with_s(&s.as_view())?;
        robot.with_q(
            &q.as_view(),
            &dq.as_view(),
            &ddq.as_view(),
            Some(&dddq.as_view()),
            0,
        )?;
        add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, Some(5.0))?;

        let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
        let start = Instant::now();
        let a_ra = topp2_ra(&topp2_problem, options_ra)?;
        let tc_ra = start.elapsed().as_secs_f64() * 1E3;
        let (tf_ra, _) = s_to_t_topp2(s.as_slice(), &a_ra, 0.0);

        robot.constraints.amax_substitute(&a_ra, 0)?;
        let topp3_problem = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, (0.0, 0.0), (0.0, 0.0))
            .with_num_stationary_max(2)
            .build_with_linearization()?;

        let start = Instant::now();
        let (a_lp, b_lp, num_stationary) = topp3_lp(&topp3_problem, options_lp)?;
        let tc_lp = start.elapsed().as_secs_f64() * 1E3;
        let (tf_lp, _) = s_to_t_topp3(s.as_slice(), &a_lp, &b_lp, num_stationary, 0.0);

        let start = Instant::now();
        let (a_qp, b_qp, num_stationary_qp) = topp3_socp(&topp3_problem, options_qp)?;
        let tc_qp = start.elapsed().as_secs_f64() * 1E3;
        let (tf_qp, _) = s_to_t_topp3(s.as_slice(), &a_qp, &b_qp, num_stationary_qp, 0.0);

        Ok((tc_ra, tc_lp, tc_qp, tf_ra, tf_lp, tf_qp))
    }

    fn run_test_topp3_lp_qp_repeated(n_exp: usize, flag_print_step: bool) -> Result<(), CoppError> {
        let options_ra = ReachSet2OptionsBuilder::new()
            .lp_feas_tol(1E-9)
            .a_cmp_abs_tol(1E-9)
            .a_cmp_rel_tol(1E-9)
            .build()?;
        let options_lp = ClarabelOptionsBuilder::new()
            .allow_almost_solved(true)
            .build()?;
        let options_qp = ClarabelOptionsBuilder::new()
            .allow_almost_solved(true)
            .build()?;

        let mut tc_sum_ra = 0.0;
        let mut tc_sum_lp = 0.0;
        let mut tc_sum_qp = 0.0;
        let mut tf_sum_ra = 0.0;
        let mut tf_sum_lp = 0.0;
        let mut tf_sum_qp = 0.0;
        let mut succeed = 0;

        for i_exp in 0..n_exp {
            if let Ok((tc_ra, tc_lp, tc_qp, tf_ra, tf_lp, tf_qp)) =
                run_one_topp3_lp_qp_case(&options_ra, &options_lp, &options_qp)
            {
                if flag_print_step {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "Exp #{}: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_qp = {:.4} ms, tf_ra = {:.4}, tf_lp = {:.4}, tf_qp = {:.4}",
                        i_exp + 1,
                        tc_ra,
                        tc_lp,
                        tc_qp,
                        tf_ra,
                        tf_lp,
                        tf_qp,
                    );
                }
                tc_sum_ra += tc_ra;
                tc_sum_lp += tc_lp;
                tc_sum_qp += tc_qp;
                tf_sum_ra += tf_ra;
                tf_sum_lp += tf_lp;
                tf_sum_qp += tf_qp;
                succeed += 1;
            }
        }

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average (fail {}): tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_qp = {:.4} ms, tf_ra = {:.4}, tf_lp = {:.4}, tf_qp = {:.4}",
            n_exp - succeed,
            tc_sum_ra / succeed as f64,
            tc_sum_lp / succeed as f64,
            tc_sum_qp / succeed as f64,
            tf_sum_ra / succeed as f64,
            tf_sum_lp / succeed as f64,
            tf_sum_qp / succeed as f64,
        );

        Ok(())
    }
}
