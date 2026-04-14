//! 3rd-order Time-Optimal Path Parameterization (TOPP3) based on linear programming (LP).
//!
//! # Method identity
//! This module implements the **optimization backend** for TOPP3-LP by transforming
//! third-order path-parameterization constraints/objective into Clarabel-compatible
//! conic form and solving with LP.
//!
//! # Discrete variables (local notation)
//! On a path grid `s[0..=n]`:
//! - `a[k]` denotes $\dot{s}_k^2$;
//! - `b[k]` denotes $\ddot{s}_k$;
//! - decision vector is organized as `x = [a[0..=n], b[0..=n]]`.
//!
//! # High-level pipeline
//! 1. Validate boundary/index contracts.
//! 2. Assemble standard TOPP3 conic constraints.
//! 3. Build sparse matrices `A`, `P`, vector `q`, and solve by Clarabel.
//! 4. Apply status acceptance policy (`ClarabelOptions::is_allow`) and extract
//!    `(a,b,num_stationary)` only when accepted.
//!
//! # API layering
//! - [`topp3_lp`]: strict/normal API, returns only accepted `(a,b,num_stationary)`.
//! - [`topp3_lp_expert`]: expert API returning `(Option<Copp3Result>, DefaultSolution<f64>)`.

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
use clarabel::solver::{DefaultSolution, DefaultSolver, IPSolver, SupportedConeT};
use core::f64;

/// Strict TOPP3-LP API for production use.
///
/// # Purpose
/// Use this entry when caller only needs a valid profile `(a,b,num_stationary)` and treats
/// non-accepted solver statuses as hard failures.
///
/// # Contract
/// - Internally calls [`topp3_lp_expert`].
/// - Returns `Ok((a,b,num_stationary))` **iff** `options.is_allow(solution.status)` is `true`.
/// - Returns `Err(CoppError::ClarabelSolverStatus(...))` when status is not accepted.
///
/// # Returns
/// Returns accepted TOPP3 profile `(a, b, num_stationary)`.
///
/// # Errors
/// Returns [`CoppError`] on model/solver failures and non-accepted solver status.
///
/// More details are provided in the documentation of [`topp3_lp_expert`].
pub fn topp3_lp(
    problem: &Topp3Problem,
    options: &ClarabelOptions,
) -> Result<Copp3Result, CoppError> {
    let (result, solution) = topp3_lp_expert(problem, options)?;
    result.ok_or_else(|| CoppError::ClarabelSolverStatus("topp3_lp".into(), solution.status))
}

/// Expert TOPP3-LP API with full Clarabel solution exposure.
///
/// # Return contract
/// - `Ok((Some(result), solution))`: status accepted by `options.is_allow(solution.status)`.
/// - `Ok((None, solution))`: solve finished but status not accepted.
/// - `Err(...)`: input/model/solver-construction runtime failures.
///
/// # Returns
/// Returns tuple `(Option<Copp3Result>, DefaultSolution<f64>)` for diagnostics.
///
/// # Errors
/// Returns [`CoppError`] only for real build/runtime failures.
///
/// # Contract
/// - caller handles `None` profile when status is not accepted;
/// - acceptance policy is fully defined by `options.is_allow`.
///
/// # Verbosity behavior
/// Logging is layered by `options.verbosity()`:
/// - [`Silent`](Verbosity::Silent): no algorithm logs;
/// - [`Summary`](Verbosity::Summary): lifecycle milestones and elapsed time;
/// - [`Debug`](Verbosity::Debug): assembly-level counters and stage summaries;
/// - [`Trace`](Verbosity::Trace): fine-grained stage deltas and solver snapshot diagnostics.
pub fn topp3_lp_expert(
    problem: &Topp3Problem,
    options: &ClarabelOptions,
) -> Result<(Option<Copp3Result>, DefaultSolution<f64>), CoppError> {
    match options.verbosity() {
        Verbosity::Silent => topp3_lp_core(problem, (options, SilentVerboser)),
        Verbosity::Summary => topp3_lp_core(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => topp3_lp_core(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => topp3_lp_core(problem, (options, TraceVerboser::new())),
    }
}

/// Core implementation for TOPP3-LP expert flow.
///
/// # Internal contract
/// `options_verboser` packs:
/// - `options`: acceptance policy and Clarabel numerical settings;
/// - `verboser`: concrete logger implementation chosen by external verbosity dispatch.
///
/// # Invariants
/// - decision-variable layout is always `x = [a[0..=n], b[0..=n]]`;
/// - extracted `(a,b)` is produced only through `clarabel_to_copp3_solution` when status is accepted.
fn topp3_lp_core(
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
            "topp3_lp: options snapshot -> allow(almost={}, max_iter={}, max_time={}, callback_term={}, insufficient_progress={}), tol_gap_rel={}, tol_feas={}, max_iter={}, verbose={}",
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
            "\ntopp3_lp started: {} <= idx_s <= {}, s_len = {}, num_stationary={:?}.",
            idx_s_start,
            idx_s_final,
            problem.a_linearization.len(),
            num_stationary
        );
    }
    check_s_interval_valid("topp3_lp", idx_s_start, idx_s_final)?;
    // Let x = [a[0,1,...,n], b[0,1,...,n]] \in R^{2*(n+1)}.
    // Step 1. Deal with constraints
    // s=b-A*x \in cone, where A[row[i],col[i]]=val[i], A \in R^{m*(n+1)}, b \in R^m, s \in R^m
    // -s=-b+A*x
    // Step 1.1 create constraints
    let (capacity_val, capacity_b, capacity_cones) =
        clarabel_standard_capacity_topp3(problem.constraints, (idx_s_start, idx_s_final));
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_lp: capacity estimate standard(val={capacity_val}, b={capacity_b}, cone={capacity_cones}), n_var={}",
            2 * (n + 1)
        );
    }
    let mut cones = Vec::<SupportedConeT<f64>>::with_capacity(capacity_cones);
    let mut row = Vec::<usize>::with_capacity(capacity_val);
    let mut col = Vec::<usize>::with_capacity(capacity_val);
    let mut val = Vec::<f64>::with_capacity(capacity_val);
    let mut b = Vec::<f64>::with_capacity(capacity_b);
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_lp: allocated capacities row/col/val/b/cones <= {capacity_val}/{capacity_val}/{capacity_val}/{capacity_b}/{capacity_cones}",
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
            "topp3_lp: standard-constraints delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}",
            row.len() - row_before_std,
            col.len() - col_before_std,
            val.len() - val_before_std,
            b.len() - b_before_std,
            cones.len() - cones_before_std
        );
    }

    // Step 1.3 build the constraints
    let n_var = 2 * (n + 1);
    let row_len = row.len();
    let col_len = col.len();
    let val_len = val.len();
    let b_len = b.len();
    let cones_len = cones.len();
    let a_csc = CscMatrix::new_from_triplets(b.len(), n_var, row, col, val);
    // Step 2. objective function. max: \int a(s) ds
    let p_object = CscMatrix::<f64>::zeros((n_var, n_var));
    let q_object = clarabel_q_object_topp3_lp(&s, num_stationary, n_var);
    if verboser.is_enabled(Verbosity::Trace) {
        let (q_min, q_max) = q_object
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_lp: matrix built with m={}, n={}, A.nnz={}, P.nnz={}, q_range=[{}, {}]",
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
            "topp3_lp: ready to solve with row/col/val/b/cones = {row_len}/{col_len}/{val_len}/{b_len}/{cones_len} and n_var = {n_var}.",
        );
    }
    // Step 3. solve the LP problem
    let settings = options.clarabel_settings().clone();
    let mut solver = DefaultSolver::<f64>::new(&p_object, &q_object, &a_csc, &b, &cones, settings)
        .map_err(|e| CoppError::ClarabelSolverError("topp3_lp".into(), e))?;
    solver.solve();
    let solution = solver.solution;
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_lp: solve done, status = {:?}, elapsed = {}.",
            solution.status,
            format_duration_human(verboser.elapsed())
        );
    }
    if verboser.is_enabled(Verbosity::Trace) {
        let show = solution.x.len().min(3);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "topp3_lp: solution x_len={}, head={:?}",
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
            "topp3_lp: allow(status)={}, extracted_profile={}",
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

/// Build LP objective vector for TOPP3-LP in Clarabel form.
///
/// # Definition
/// The primal objective is `max \int a(s) ds`, converted to minimization as
/// `min \int -a(s) ds`.
///
/// # Layout
/// - first block (`a`) gets negated quadrature weights;
/// - second block (`b`) is zero-padded.
#[inline(always)]
fn clarabel_q_object_topp3_lp(s: &[f64], num_stationary: (usize, usize), n_var: usize) -> Vec<f64> {
    let mut q_object = get_weight_a_topp3(s, num_stationary);
    // max \int a(s) ds <=> min \int -a(s) ds
    q_object.iter_mut().for_each(|q_i| *q_i = -*q_i);
    q_object.resize(n_var, 0.0);
    q_object
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::ClarabelOptionsBuilder;
    use crate::copp::InterpolationMode;
    use crate::copp::copp2::stable::basic::{Topp2ProblemBuilder, s_to_t_topp2};
    use crate::copp::copp2::stable::reach_set2::ReachSet2OptionsBuilder;
    use crate::copp::copp2::stable::topp2_ra::topp2_ra;
    use crate::copp::copp3::stable::basic::{Topp3ProblemBuilder, s_to_t_topp3, t_to_s_topp3};
    use crate::path::add_symmetric_axial_limits_for_test;
    use crate::robot::robot_core::Robot;
    use nalgebra::DMatrix;
    use rand::RngExt;
    use std::time::Instant;

    #[test]
    fn test_topp3_lp() -> Result<(), CoppError> {
        run_test_topp3_lp_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average over 100 experiments: tc_ra = 0.3417 ms, tc_lp = 261.8795 ms, tf_ra = 6.138643, tf_lp = 7.051755
    #[test]
    #[ignore = "slow"]
    fn test_topp3_lp_robust() -> Result<(), CoppError> {
        run_test_topp3_lp_repeated(100, true)
    }

    fn run_one_topp3_lp_case(
        options_lp: &ClarabelOptions,
    ) -> Result<(f64, f64, f64, f64, f64, usize), CoppError> {
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
        let options_ra = ReachSet2OptionsBuilder::new()
            .lp_feas_tol(1E-9)
            .a_cmp_abs_tol(1E-9)
            .a_cmp_rel_tol(1E-9)
            .build()?;
        let a_profile_ra = topp2_ra(&topp2_problem, &options_ra)?;
        let time_topp_ra = start.elapsed().as_secs_f64() * 1E3;
        let (t_motion_ra, _) = s_to_t_topp2(s.as_slice(), &a_profile_ra, 0.0);

        let start = Instant::now();
        robot.constraints.amax_substitute(&a_profile_ra, 0)?;
        let topp3_problem =
            Topp3ProblemBuilder::new(&mut robot, 0, &a_profile_ra, (0.0, 0.0), (0.0, 0.0))
                .with_num_stationary_max(2)
                .build_with_linearization()?;
        let (a_profile, b_profile, num_stationary) = topp3_lp(&topp3_problem, options_lp)?;
        let time_topp3_lp = start.elapsed().as_secs_f64() * 1E3;
        let start = Instant::now();
        let (t_motion_lp, t_s) =
            s_to_t_topp3(s.as_slice(), &a_profile, &b_profile, num_stationary, 0.0);
        let s_t = t_to_s_topp3(
            s.as_slice(),
            &a_profile,
            &b_profile,
            num_stationary,
            &t_s,
            InterpolationMode::UniformTimeGrid(0.0, 1E-3, true),
        );
        let time_interpolation = start.elapsed().as_secs_f64() * 1E3;
        Ok((
            time_topp_ra,
            time_topp3_lp,
            time_interpolation,
            t_motion_ra,
            t_motion_lp,
            s_t.len(),
        ))
    }

    fn run_test_topp3_lp_repeated(n_exp: usize, flag_print_step: bool) -> Result<(), CoppError> {
        let options_lp = ClarabelOptionsBuilder::new()
            .allow_almost_solved(true)
            .build()?;

        let mut tc_sum_ra = 0.0;
        let mut tc_sum_lp = 0.0;
        let mut tf_sum_ra = 0.0;
        let mut tf_sum_lp = 0.0;
        for i_exp in 0..n_exp {
            let (
                time_topp_ra,
                time_topp3_lp,
                time_interpolation,
                t_motion_ra,
                t_motion_lp,
                s_t_len,
            ) = run_one_topp3_lp_case(&options_lp)?;

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_interpolation = {:.4} ms, tf_ra = {:.6}, tf_lp = {:.6}, s_t.len() = {}",
                    i_exp + 1,
                    time_topp_ra,
                    time_topp3_lp,
                    time_interpolation,
                    t_motion_ra,
                    t_motion_lp,
                    s_t_len,
                );
            }

            tc_sum_ra += time_topp_ra;
            tc_sum_lp += time_topp3_lp;
            tf_sum_ra += t_motion_ra;
            tf_sum_lp += t_motion_lp;
        }

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average over {} experiments: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tf_ra = {:.6}, tf_lp = {:.6}",
            n_exp,
            tc_sum_ra / n_exp as f64,
            tc_sum_lp / n_exp as f64,
            tf_sum_ra / n_exp as f64,
            tf_sum_lp / n_exp as f64,
        );

        Ok(())
    }
}
