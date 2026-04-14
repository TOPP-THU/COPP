//! Reachability-analysis solver for second-order time-optimal path parameterization.
//!
//! # Method identity
//! This module implements **Reachability Analysis (RA)** for
//! **Time-Optimal Path Parameterization (TOPP2)** and shares the same state-space
//! conventions used by **Convex-Objective Path Parameterization (COPP2)** components.
//!
//! # Discrete variables (local notation)
//! On a path grid `s[0..=n]`:
//! - `a[k]` denotes $\dot{s}_k^2$;
//! - backward intervals are `[a_min[k], a_max[k]]` from `reach_set2`;
//! - forward pass selects one feasible state per station, yielding the final profile `a`.
//!
//! # High-level pipeline
//! 1. Build backward reachable intervals by calling `reach_set2_backward`.
//! 2. Run forward clipping against local constraints and backward intervals.
//! 3. Select maximal feasible `a[k]` at each stage to recover the time-optimal profile.

use super::reach_set2::{ReachSet2Options, reach_set2_backward};
use crate::copp::copp2::formulation::Topp2Problem;
use crate::copp::{ApproxOrdering, approx_order};
use crate::diag::{
    CoppError, DebugVerboser, SilentVerboser, SummaryVerboser, TraceVerboser, Verboser, Verbosity,
    format_duration_human,
};
use crate::math::numerical::{LpToleranceOptions, lp_1d};
use core::f64;
use itertools::izip;

/// Solve TOPP2 with RA and return the profile $a(s)=\dot{s}^2$.
///
/// # Returns
/// Returns `a` such that:
/// - `a[0] = a_start`, `a[n] = a_final`, where `n = idx_s_final - idx_s_start`;
/// - `a` is time-optimal under configured first-/second-order constraints.
///
/// The returned profile can be mapped to `t(s)` by `s_to_t_topp2`, then to sampled
/// `s(t)` by `t_to_s_topp2`.
///
/// # Errors
/// Returns `CoppError` when backward reachable-set construction fails or when
/// forward pass cannot maintain feasibility under constraints.
///
/// # Contract
/// - station interval and boundary states must be valid for the given constraints;
/// - `options` must contain valid tolerance settings.
pub fn topp2_ra(problem: &Topp2Problem, options: &ReachSet2Options) -> Result<Vec<f64>, CoppError> {
    match options.verbosity {
        Verbosity::Silent => topp2_ra_core(problem, (options, SilentVerboser)),
        Verbosity::Summary => topp2_ra_core(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => topp2_ra_core(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => topp2_ra_core(problem, (options, TraceVerboser::new())),
    }
}

/// Core implementation of Reachability Analysis for TOPP2 with layered verbosity logging.
fn topp2_ra_core(
    problem: &Topp2Problem,
    options_verboser: (&ReachSet2Options, impl Verboser),
) -> Result<Vec<f64>, CoppError> {
    let (options, mut verboser) = options_verboser;
    if verboser.is_enabled(Verbosity::Summary) {
        verboser.record_start_time();
        crate::verbosity_log!(
            Verbosity::Summary,
            "\ntopp2_ra started: {} <= idx_s <= {}, a_start = {}, a_final = {}.",
            problem.idx_s_interval.0,
            problem.idx_s_interval.1,
            problem.a_boundary.0,
            problem.a_boundary.1,
        );
    }

    // Step 1. Compute the backward reachable set.
    let reach_set = reach_set2_backward(problem, options).map_err(|e| {
        if verboser.is_enabled(Verbosity::Debug) {
            crate::verbosity_log!(Verbosity::Debug, "{e:?}");
        } else if verboser.is_enabled(Verbosity::Summary) {
            crate::verbosity_log!(
                Verbosity::Summary,
                "topp2_ra: failed while computing backward reachable set."
            );
        }
        e
    })?;
    let a_max = &reach_set.a_max;
    let a_min = &reach_set.a_min;

    // Step 2. Forward pass to select the maximal feasible state at each grid point.
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(Verbosity::Debug, "Forward pass started.");
    }

    let (idx_s_start, idx_s_final) = problem.idx_s_interval;
    let n = idx_s_final - idx_s_start;
    let mut a = vec![0.0; n + 1];
    let mut a_prev = problem.a_boundary.0;
    *a.first_mut().unwrap() = a_prev;

    let mut a_b = Vec::<(f64, f64, f64)>::with_capacity(2 * problem.constraints.acc_rows());
    for (k, (a_curr, &a_max_curr_, &a_min_curr_)) in
        izip!(a.iter_mut(), a_max, a_min).enumerate().skip(1)
    {
        let idx_s = idx_s_start + k;
        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                Verbosity::Trace,
                "\tForward pass at k = {k} (idx_s = {idx_s}): backward interval {a_min_curr_} <= a[k] <= {a_max_curr_}, a_prev = {a_prev}."
            );
        }

        a_b.clear();
        problem
            .constraints
            .fill_acc_topp2::<true>(&mut a_b, idx_s - 1);
        // a_b.0 * a[k] + a_b.1 * a[k-1] <= a_b.2
        let (mut a_max_curr, mut a_min_curr) = lp_1d::<true>(
            a_b.iter().map(|&coeffs| {
                // coeffs.0 * a_curr  + coeffs.1* a_prev <= coeffs.2
                // coeffs.0 * a_curr <= coeffs.2 - coeffs.1 * a_prev
                (coeffs.0, coeffs.2 - coeffs.1 * a_prev)
            }),
            &LpToleranceOptions::with_feas_tol(options.lp_feas_tol),
        );

        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                Verbosity::Trace,
                "\t\tForward LP result before clipping: {a_min_curr} <= a[k] <= {a_max_curr}."
            );
        }

        a_max_curr = a_max_curr.min(a_max_curr_);
        a_min_curr = a_min_curr.max(a_min_curr_);
        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                Verbosity::Trace,
                "\t\tAfter clipping with backward reachable set: {a_min_curr} <= a[k] <= {a_max_curr}."
            );
        }

        if a_max_curr.is_nan()
            || a_min_curr.is_nan()
            || matches!(
                approx_order(
                    a_max_curr,
                    a_min_curr,
                    options.a_cmp_abs_tol,
                    options.a_cmp_rel_tol,
                ),
                ApproxOrdering::Less
            )
        {
            let err = CoppError::Infeasible(
                "topp2_ra".into(),
                format!(
                    "The reachable set is empty at index {} during the forward pass where a_max = {}, a_min = {}",
                    idx_s_start + k,
                    a_max_curr,
                    a_min_curr
                ),
            );
            if verboser.is_enabled(Verbosity::Debug) {
                crate::verbosity_log!(Verbosity::Debug, "{err:?}");
            } else if verboser.is_enabled(Verbosity::Summary) {
                crate::verbosity_log!(
                    Verbosity::Summary,
                    "topp2_ra: the forward pass failed at index {idx_s} due to infeasibility."
                );
            }
            return Err(err);
        }

        if a_max_curr.is_infinite() {
            let err = CoppError::Unbounded(
                "topp2_ra".into(),
                format!(
                    "The reachable set is unbounded at index {} during the forward pass where a_max = {}",
                    idx_s_start + k,
                    a_max_curr
                ),
            );
            if verboser.is_enabled(Verbosity::Debug) {
                crate::verbosity_log!(Verbosity::Debug, "{err:?}");
            } else if verboser.is_enabled(Verbosity::Summary) {
                crate::verbosity_log!(
                    Verbosity::Summary,
                    "topp2_ra: the forward pass failed at index {idx_s} due to unboundedness."
                );
            }
            return Err(err);
        }

        if verboser.is_enabled(Verbosity::Debug)
            && matches!(
                approx_order(
                    a_max_curr,
                    a_min_curr,
                    options.a_cmp_abs_tol,
                    options.a_cmp_rel_tol,
                ),
                ApproxOrdering::Equal
            )
        {
            crate::verbosity_log!(
                Verbosity::Debug,
                "The forward reachable set at k = {k} (idx_s = {idx_s}) is degenerate since a_max and a_min are approximately equal at {}.",
                0.5 * (a_max_curr + a_min_curr)
            );
        }

        a_prev = a_max_curr;
        *a_curr = a_prev;

        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                Verbosity::Trace,
                "\t\tSelected maximal feasible state: a[k] = {a_prev}."
            );
        }
    }

    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            Verbosity::Summary,
            "topp2_ra: total elapsed time = {}.\n",
            format_duration_human(verboser.elapsed())
        );
    }

    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::InterpolationMode;
    use crate::copp::copp2::stable::basic::{
        Topp2ProblemBuilder, a_to_b_topp2, s_to_t_topp2, t_to_s_topp2,
    };
    use crate::copp::copp2::stable::reach_set2::ReachSet2OptionsBuilder;
    use crate::path::{add_symmetric_axial_limits_for_test, lissajous_path_for_test};
    use crate::robot::robot_core::Robot;
    use std::time::{Duration, Instant};

    #[test]
    fn test_topp2_ra() -> Result<(), CoppError> {
        run_test_topp2_ra_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average over 10000 experiments: tc_topp2_ra = 0.286978 ms, tc_interpolation = 0.002001 ms, t_final = 6.168578
    #[test]
    #[ignore = "slow"]
    fn test_topp2_ra_robust() -> Result<(), CoppError> {
        run_test_topp2_ra_repeated(10000, true)
    }

    fn run_test_topp2_ra_repeated(n_exp: usize, flag_print_step: bool) -> Result<(), CoppError> {
        let mut tc_sum_ra = Duration::ZERO;
        let mut tc_sum_interpolation = Duration::ZERO;
        let mut t_final_sum = 0.0;

        let options = ReachSet2OptionsBuilder::new()
            .lp_feas_tol(1E-9)
            .a_cmp_abs_tol(1E-9)
            .a_cmp_rel_tol(1E-9)
            .verbosity(Verbosity::Summary)
            .build()?;

        for i_exp in 0..n_exp {
            let dim = 7;
            let n: usize = 1000;
            let mut robot = Robot::with_capacity(dim, n);

            let mut rng = rand::rng();
            let (s, derivs, _, _) = lissajous_path_for_test(dim, n, &mut rng).map_err(|e| {
                CoppError::InvalidInput("lissajous_path_for_test".into(), e.to_string())
            })?;
            robot.with_s(&s.as_view())?;
            robot.with_q(
                &derivs.q.as_view(),
                &derivs.dq.as_ref().unwrap().as_view(),
                &derivs.ddq.as_ref().unwrap().as_view(),
                derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
                0,
            )?;
            add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, None)?;

            let start = Instant::now();
            let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
            let a_profile = topp2_ra(&topp2_problem, &options)?;
            let tc_topp2_ra = start.elapsed();

            let b_profile = a_to_b_topp2(s.as_slice(), &a_profile);
            assert!(
                !izip!(
                    s.as_slice().windows(2),
                    a_profile.windows(2),
                    b_profile.iter()
                )
                .any(|(s_pair, a_pair, b)| {
                    let ds_double = 2.0 * (s_pair[1] - s_pair[0]);
                    let db = (a_pair[1] - a_pair[0]) / ds_double;
                    (*b - db).abs() > 1e-3
                }),
                "b_profile generation failed!"
            );

            let start = Instant::now();
            let (t_final, t_s) = s_to_t_topp2(s.as_slice(), &a_profile, 0.0);
            assert_eq!(t_s.len(), s.ncols());
            let tc_interpolation = start.elapsed();
            let s_t = t_to_s_topp2(
                s.as_slice(),
                &a_profile,
                &t_s,
                InterpolationMode::UniformTimeGrid(0.0, 1E-3, true),
            );

            tc_sum_ra += tc_topp2_ra;
            tc_sum_interpolation += tc_interpolation;
            t_final_sum += t_final;

            if flag_print_step && ((i_exp + 1) % 100 == 0) {
                crate::verbosity_log!(
                    Verbosity::Summary,
                    "Exp #{}: tc_topp2_ra = {:.4} ms, tc_interpolation = {:.4} ms, t_final = {:.4} s, s_t.len() = {}",
                    i_exp + 1,
                    tc_topp2_ra.as_secs_f64() * 1E3,
                    tc_interpolation.as_secs_f64() * 1E3,
                    t_final,
                    s_t.len()
                );
            }
        }

        crate::verbosity_log!(
            Verbosity::Summary,
            "Average over {} experiments: tc_topp2_ra = {:.6} ms, tc_interpolation = {:.6} ms, t_final = {:.6}",
            n_exp,
            tc_sum_ra.as_secs_f64() * 1E3 / n_exp as f64,
            tc_sum_interpolation.as_secs_f64() * 1E3 / n_exp as f64,
            t_final_sum / n_exp as f64
        );

        Ok(())
    }
}
