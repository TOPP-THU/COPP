//! Reachable-set construction for second-order path parameterization.
//!
//! # Method identity
//! This module implements the reachable-set stage via **Reachability Analysis (RA)** in
//! a **Dynamic Programming (DP)** compatible form, which can be used by:
//! - **Time-Optimal Path Parameterization (TOPP2)**,
//! - **Convex-Objective Path Parameterization (COPP2)**.
//!
//! # Discrete variables (local notation)
//! On a path grid `s[0..=n]`:
//! - `a[k]` denotes $\dot{s}_k^2$ (nonnegative scalar state);
//! - reachable interval at station `k` is `[a_min[k], a_max[k]]`.
//!
//! # High-level pipeline
//! 1. Validate boundary feasibility at both interval ends.
//! 2. Backward pass from terminal boundary to construct feasible intervals.
//! 3. Optional forward clipping (when bidirectional mode is enabled) to enforce start boundary.
//! 4. Return interval arrays `a_min` / `a_max` for downstream solvers.

use crate::copp::copp2::formulation::Topp2Problem;
use crate::copp::{ApproxOrdering, approx_order};
use crate::diag::{
    CoppError, DebugVerboser, SilentVerboser, SummaryVerboser, TraceVerboser, Verboser, Verbosity,
    check_abs_rel_tol, check_strictly_positive, format_duration_human,
};
use crate::math::numerical::{
    LP_BOUND, Lp2dWarmStart, LpToleranceOptions, lp_1d, lp_2d_incre_max_y, normalize_lp2d,
};
use core::f64;

/// Reachable intervals of $a(s)=\dot{s}^2$ for TOPP2/COPP2.  
/// For each point `s[k]`, the reachable set is `a_min[k] <= a[k] <= a_max[k]`.
pub struct ReachSet2 {
    pub a_max: Vec<f64>,
    pub a_min: Vec<f64>,
}

/// Compute backward-only reachable intervals of $a(s)=\dot{s}^2$ using Reachability Analysis.
///
/// Performs a single backward pass from the terminal boundary; the start boundary
/// is **not** enforced.  Use this when only the terminal state is constrained, or
/// as an intermediate step before bidirectional analysis.
///
/// # Returns
/// [`ReachSet2`] with `a_min[k] <= a[k] <= a_max[k]` for every station.
///
/// # Errors
/// Returns `CoppError` when boundary states are infeasible, LP subproblems fail,
/// or numerical comparisons violate configured tolerances.
///
/// # Contract
/// - `problem` indices and boundaries must be consistent with the constraints domain.
/// - `options` tolerances must be positive and numerically meaningful.
#[inline]
pub fn reach_set2_backward(
    problem: &Topp2Problem,
    options: &ReachSet2Options,
) -> Result<ReachSet2, CoppError> {
    match options.verbosity {
        Verbosity::Silent => reach_set2_core::<false>(problem, (options, SilentVerboser)),
        Verbosity::Summary => reach_set2_core::<false>(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => reach_set2_core::<false>(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => reach_set2_core::<false>(problem, (options, TraceVerboser::new())),
    }
}

/// Compute bidirectional reachable intervals of $a(s)=\dot{s}^2$.
///
/// Performs a backward pass then clips the result with a forward pass to enforce
/// **both** the start and terminal boundary constraints simultaneously.
///
/// # Returns
/// [`ReachSet2`] with `a_min[k] <= a[k] <= a_max[k]` for every station.
///
/// # Errors
/// Returns `CoppError` when boundary states are infeasible, LP subproblems fail,
/// or numerical comparisons violate configured tolerances.
///
/// # Contract
/// - `problem` indices and boundaries must be consistent with the constraints domain.
/// - `options` tolerances must be positive and numerically meaningful.
#[inline]
pub fn reach_set2_bidirectional(
    problem: &Topp2Problem,
    options: &ReachSet2Options,
) -> Result<ReachSet2, CoppError> {
    match options.verbosity {
        Verbosity::Silent => reach_set2_core::<true>(problem, (options, SilentVerboser)),
        Verbosity::Summary => reach_set2_core::<true>(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => reach_set2_core::<true>(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => reach_set2_core::<true>(problem, (options, TraceVerboser::new())),
    }
}

/// Core RA implementation with layered verbosity.
/// # Mode
/// - `BIDIRECTION = false`: backward reachable set only (terminal boundary constrained).
/// - `BIDIRECTION = true`: bidirectional reachable set (both start and terminal boundaries constrained).
fn reach_set2_core<const BIDIRECTION: bool>(
    problem: &Topp2Problem,
    options_verboser: (&ReachSet2Options, impl Verboser),
) -> Result<ReachSet2, CoppError> {
    let (options, mut verboser) = options_verboser;
    if verboser.is_enabled(Verbosity::Summary) {
        verboser.record_start_time();
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "reach_set2 started: {} <= idx_s <= {}, a_start = {}, a_final = {}.",
            problem.idx_s_interval.0,
            problem.idx_s_interval.1,
            problem.a_boundary.0,
            problem.a_boundary.1,
        );
        if BIDIRECTION {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "Bidirectional reachable set will be computed."
            );
        } else {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "Backward reachable set will be computed."
            );
        }
    }

    let (idx_s_start, idx_s_final) = problem.idx_s_interval;
    let a_max_0 = problem.constraints.amax_unchecked(problem.idx_s_interval.0);
    if matches!(
        approx_order(
            problem.a_boundary.0,
            a_max_0,
            options.a_cmp_abs_tol,
            options.a_cmp_rel_tol,
        ),
        ApproxOrdering::Greater
    ) {
        let err = CoppError::Infeasible(
            "reach_set2".into(),
            format!(
                "The initial state a_start = {} cannot be greater than the maximum feasible state a_max[0] = {a_max_0} at idx_s_start = {idx_s_start}",
                problem.a_boundary.0,
            ),
        );
        if verboser.is_enabled(Verbosity::Debug) {
            crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
        } else if verboser.is_enabled(Verbosity::Summary) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Debug,
                "reach_set2: the backward pass failed at index {idx_s_start} due to infeasibility of the initial state."
            );
        }
        return Err(err);
    }
    let a_max_f = problem.constraints.amax_unchecked(problem.idx_s_interval.1);
    if matches!(
        approx_order(
            problem.a_boundary.1,
            a_max_f,
            options.a_cmp_abs_tol,
            options.a_cmp_rel_tol,
        ),
        ApproxOrdering::Greater
    ) {
        let err = CoppError::Infeasible(
            "reach_set2".into(),
            format!(
                "The final state a_final = {} cannot be greater than the maximum feasible state a_max[n] = {a_max_f} at idx_s_final = {idx_s_final}",
                problem.a_boundary.1,
            ),
        );
        if verboser.is_enabled(Verbosity::Debug) {
            crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
        } else if verboser.is_enabled(Verbosity::Summary) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Debug,
                "reach_set2: the backward pass failed at index {idx_s_final} due to infeasibility of the final state."
            );
        }
        return Err(err);
    }

    // Step 1. Initialize a_max and a_min at s_final
    let n = idx_s_final - idx_s_start;
    let mut a_max = vec![f64::INFINITY; n + 1];
    let mut a_min = vec![0.0; n + 1];
    *a_max.last_mut().unwrap() = problem.a_boundary.1;
    *a_min.last_mut().unwrap() = problem.a_boundary.1;
    // Step 2. Backward pass
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(crate::diag::Verbosity::Summary, "Backward pass started.");
    }

    let mut a_b = Vec::<(f64, f64, f64)>::with_capacity(2 + 2 * problem.constraints.acc_rows());
    let mut a_max_next = problem.a_boundary.1;
    let mut a_min_next = problem.a_boundary.1;
    for (k, (a_max_k, a_min_k)) in a_max
        .iter_mut()
        .zip(a_min.iter_mut())
        .take(n)
        .enumerate()
        .rev()
    {
        let idx_s = idx_s_start + k;
        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "\tBackward pass at k = {k} (idx_s = {idx_s}) to compute a[k]: {a_min_next} <= a[k+1] <= {a_max_next}."
            );
        }

        a_b.clear();
        a_b.push((1.0, 0.0, a_max_next));
        a_b.push((-1.0, 0.0, -a_min_next));
        problem.constraints.fill_acc_topp2::<true>(&mut a_b, idx_s);
        // a_b.0 * a[k+1] + a_b.1 * a[k] <= a_b.2

        let a_next_mid = 0.5 * (a_max_next + a_min_next);
        let (a_max_curr, a_min_curr) = match approx_order(
            a_max_next,
            a_min_next,
            options.a_cmp_abs_tol,
            options.a_cmp_rel_tol,
        ) {
            ApproxOrdering::Equal => {
                if verboser.is_enabled(Verbosity::Trace) {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "\t\ta[k+1] = {a_next_mid}"
                    );
                }
                lp_1d::<true>(
                    a_b.iter().skip(2).map(|&coeffs| {
                        // coeffs.0 * a_next  + coeffs.1* a_curr <= coeffs.2
                        // coeffs.1 * a_curr <= coeffs.2 - coeffs.0 * a_next
                        (coeffs.1, coeffs.2 - coeffs.0 * a_next_mid)
                    }),
                    &LpToleranceOptions::with_feas_tol(options.lp_feas_tol),
                )
            }
            ApproxOrdering::Greater => {
                if verboser.is_enabled(Verbosity::Trace) {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "\t\t{a_min_next} <= a[k+1] <= {a_max_next}"
                    );
                }
                // Check whether the forward pass of `amin_curr` can be skipped
                let (a_test_max, a_test_min) = lp_1d::<true>(
                    a_b.iter().skip(2).map(|&coeffs| {
                        // coeffs.0 * a_next  + coeffs.1* a_curr <= coeffs.2
                        // coeffs.0 * a_next <= coeffs.2 - coeffs.1 * a_curr
                        (coeffs.0, coeffs.2 - coeffs.1 * *a_min_k)
                    }),
                    &LpToleranceOptions::with_feas_tol(options.lp_feas_tol),
                );
                let flag_need_min = a_test_max.is_nan()
                    || a_test_min.is_nan()
                    || a_test_max < a_min_next
                    || a_test_max > a_max_next;
                if verboser.is_enabled(Verbosity::Trace) {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "\t\tBackward skip checks at idx_s = {idx_s}: need_max = true, need_min = {flag_need_min}."
                    );
                }
                if flag_need_min {
                    backward_bound_a_next::<true, true>(&mut a_b, a_next_mid, options.lp_feas_tol)
                } else {
                    (
                        backward_bound_a_next::<true, false>(
                            &mut a_b,
                            a_next_mid,
                            options.lp_feas_tol,
                        )
                        .0,
                        *a_min_k,
                    )
                }
            }
            ApproxOrdering::Less => {
                let err = CoppError::Infeasible(
                    "reach_set2".into(),
                    format!(
                        "The reachable set is empty at index {idx_s} during the backward pass where a_max_next = {a_max_next}, a_min_next = {a_min_next}"
                    ),
                );
                if verboser.is_enabled(Verbosity::Debug) {
                    crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
                } else if verboser.is_enabled(Verbosity::Summary) {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "reach_set2: the backward pass failed at index {idx_s} due to infeasibility."
                    );
                }
                return Err(err);
            }
        };

        if a_max_curr.is_nan() || a_min_curr.is_nan() {
            let err = CoppError::Infeasible(
                "reach_set2".into(),
                format!(
                    "The reachable set is empty at index {idx_s} during the backward pass where a_max = {a_max_curr}, a_min = {a_min_curr}"
                ),
            );
            if verboser.is_enabled(Verbosity::Debug) {
                crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
            } else if verboser.is_enabled(Verbosity::Summary) {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Debug,
                    "reach_set2: the backward pass failed at index {idx_s} due to infeasibility."
                );
            }
            return Err(err);
        }
        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "\t\tBackward LP result at k = {k}: {a_min_curr} <= a[k] <= {a_max_curr}."
            );
        }

        *a_max_k = a_max_curr.min(problem.constraints.amax_unchecked(idx_s));
        *a_min_k = a_min_curr.max(0.0);

        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "\t\tAfter clipping with path constraints at k = {k}: {} <= a[k] <= {}.",
                *a_min_k,
                *a_max_k
            );
        }

        match approx_order(
            *a_max_k,
            *a_min_k,
            options.a_cmp_abs_tol,
            options.a_cmp_rel_tol,
        ) {
            ApproxOrdering::Less => {
                let err = CoppError::Infeasible(
                    "reach_set2".into(),
                    format!(
                        "The reachable set is empty at k = {k} (idx_s = {idx_s}) during the backward pass where a_max = {a_max_k}, a_min = {a_min_k}"
                    ),
                );
                if verboser.is_enabled(Verbosity::Debug) {
                    crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
                } else if verboser.is_enabled(Verbosity::Summary) {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "reach_set2: the backward pass failed at k = {k} (idx_s = {idx_s}) due to infeasibility."
                    );
                }
                return Err(err);
            }
            ApproxOrdering::Equal => {
                if verboser.is_enabled(Verbosity::Debug) && k < n - 1 {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "The backward reachable set at k = {k} (idx_s = {idx_s}) is degenerate since a_max = {a_max_k} and a_min = {a_min_k} are approximately equal."
                    );
                }
                *a_max_k = 0.5 * (*a_max_k + *a_min_k);
                *a_min_k = *a_max_k;
            }
            ApproxOrdering::Greater => {}
        }
        a_max_next = *a_max_k;
        a_min_next = *a_min_k;

        if verboser.is_enabled(Verbosity::Trace) {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "\t\tBackward propagated interval to k-1: {a_min_next} <= a[k] <= {a_max_next}."
            );
        }
    }

    if BIDIRECTION {
        if verboser.is_enabled(Verbosity::Debug) {
            crate::verbosity_log!(crate::diag::Verbosity::Summary, "Forward pass started.");
        }

        *a_max.first_mut().unwrap() = problem.a_boundary.0;
        *a_min.first_mut().unwrap() = problem.a_boundary.0;
        let mut a_max_prev = problem.a_boundary.0;
        let mut a_min_prev = problem.a_boundary.0;
        for (k, (a_max_k, a_min_k)) in a_max.iter_mut().zip(a_min.iter_mut()).enumerate().skip(1) {
            let idx_s = idx_s_start + k;
            if verboser.is_enabled(Verbosity::Trace) {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "\tForward pass at k = {k} (idx_s = {idx_s}): prev interval {} <= a[k-1] <= {}, backward interval {} <= a[k] <= {}.",
                    a_min_prev,
                    a_max_prev,
                    *a_min_k,
                    *a_max_k
                );
            }

            a_b.clear();
            a_b.push((1.0, 0.0, a_max_prev));
            a_b.push((-1.0, 0.0, -a_min_prev));
            problem
                .constraints
                .fill_acc_topp2::<false>(&mut a_b, idx_s_start + k - 1);
            // a_b.0 * a[k-1] + a_b.1 * a[k] <= a_b.2

            let a_prev_mid = 0.5 * (a_max_prev + a_min_prev);
            let (a_max_curr, a_min_curr) = match approx_order(
                a_max_prev,
                a_min_prev,
                options.a_cmp_abs_tol,
                options.a_cmp_rel_tol,
            ) {
                ApproxOrdering::Equal => {
                    lp_1d::<true>(
                        a_b.iter().skip(2).map(|&coeffs| {
                            // coeffs.0 * a_prev  + coeffs.1* a_curr <= coeffs.2
                            // coeffs.1 * a_curr <= coeffs.2 - coeffs.0 * a_prev
                            (coeffs.1, coeffs.2 - coeffs.0 * a_prev_mid)
                        }),
                        &LpToleranceOptions::with_feas_tol(options.lp_feas_tol),
                    )
                }
                ApproxOrdering::Greater => {
                    // Check whether the forward pass of `amax_curr` can be skipped
                    let (a_test_max, a_test_min) = lp_1d::<true>(
                        a_b.iter().skip(2).map(|&coeffs| {
                            // coeffs.0 * a_prev  + coeffs.1* a_curr <= coeffs.2
                            // coeffs.0 * a_prev <= coeffs.2 - coeffs.1 * a_curr
                            (coeffs.0, coeffs.2 - coeffs.1 * *a_max_k)
                        }),
                        &LpToleranceOptions::with_feas_tol(options.lp_feas_tol),
                    );
                    let flag_need_max = a_test_max.is_nan()
                        || a_test_min.is_nan()
                        || a_test_max < a_min_prev
                        || a_test_max > a_max_prev;
                    // Check whether the forward pass of `amin_curr` can be skipped
                    let (a_test_max, a_test_min) = lp_1d::<true>(
                        a_b.iter().skip(2).map(|&coeffs| {
                            // coeffs.0 * a_prev  + coeffs.1* a_curr <= coeffs.2
                            // coeffs.0 * a_prev <= coeffs.2 - coeffs.1 * a_curr
                            (coeffs.0, coeffs.2 - coeffs.1 * *a_min_k)
                        }),
                        &LpToleranceOptions::with_feas_tol(options.lp_feas_tol),
                    );
                    let flag_need_min = a_test_max.is_nan()
                        || a_test_min.is_nan()
                        || a_test_max < a_min_prev
                        || a_test_max > a_max_prev;

                    if verboser.is_enabled(Verbosity::Trace) {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "\t\tForward skip checks at idx_s = {idx_s}: need_max = {flag_need_max}, need_min = {flag_need_min}."
                        );
                    }

                    // forward and backward is the same
                    match (flag_need_max, flag_need_min) {
                        (true, true) => backward_bound_a_next::<true, true>(
                            &mut a_b,
                            a_prev_mid,
                            options.lp_feas_tol,
                        ),
                        (true, false) => (
                            backward_bound_a_next::<true, false>(
                                &mut a_b,
                                a_prev_mid,
                                options.lp_feas_tol,
                            )
                            .0,
                            *a_min_k,
                        ),
                        (false, true) => (
                            *a_max_k,
                            backward_bound_a_next::<false, true>(
                                &mut a_b,
                                a_prev_mid,
                                options.lp_feas_tol,
                            )
                            .1,
                        ),
                        (false, false) => (*a_max_k, *a_min_k),
                    }
                }
                ApproxOrdering::Less => {
                    let err = CoppError::Infeasible(
                        "reach_set2".into(),
                        format!(
                            "The reachable set is empty at index {idx_s} during the forward pass where a_max_prev = {a_max_prev}, a_min_prev = {a_min_prev}"
                        ),
                    );
                    if verboser.is_enabled(Verbosity::Debug) {
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
                    } else if verboser.is_enabled(Verbosity::Summary) {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "reach_set2: the forward pass failed at index {idx_s} due to infeasibility."
                        );
                    }
                    return Err(err);
                }
            };

            if verboser.is_enabled(Verbosity::Trace) {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "\t\tForward LP result at idx_s = {idx_s}: {a_min_curr} <= a[k] <= {a_max_curr}."
                );
            }

            if a_max_curr.is_nan() || a_min_curr.is_nan() {
                let err = CoppError::Infeasible(
                    "reach_set2".into(),
                    format!(
                        "The reachable set is empty at index {idx_s} during the forward pass where a_max = {a_max_curr}, a_min = {a_min_curr}"
                    ),
                );
                if verboser.is_enabled(Verbosity::Debug) {
                    crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
                } else if verboser.is_enabled(Verbosity::Summary) {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "reach_set2: the forward pass failed at index {idx_s} due to infeasibility."
                    );
                }
                return Err(err);
            }

            a_max_prev = a_max_curr.min(*a_max_k);
            a_min_prev = a_min_curr.max(*a_min_k);
            if verboser.is_enabled(Verbosity::Trace) {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "\t\tAfter intersecting with backward interval at idx_s = {idx_s}: {a_min_prev} <= a[k] <= {a_max_prev}."
                );
            }
            match approx_order(
                a_max_prev,
                a_min_prev,
                options.a_cmp_abs_tol,
                options.a_cmp_rel_tol,
            ) {
                ApproxOrdering::Less => {
                    let err = CoppError::Infeasible(
                        "reach_set2".into(),
                        format!(
                            "The reachable set is empty at index {idx_s} during the forward pass where a_max = {a_max_prev}, a_min = {a_min_prev}"
                        ),
                    );
                    if verboser.is_enabled(Verbosity::Debug) {
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "{err:?}");
                    } else if verboser.is_enabled(Verbosity::Summary) {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "reach_set2: the forward pass failed at index {idx_s} due to infeasibility."
                        );
                    }
                    return Err(err);
                }
                ApproxOrdering::Equal => {
                    if verboser.is_enabled(Verbosity::Debug) && k < n - 1 {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "The forward reachable set at idx_s = {idx_s} is degenerate since a_max_prev and a_min_prev are approximately equal at {a_prev_mid}."
                        );
                    }
                    a_max_prev = 0.5 * (a_max_prev + a_min_prev);
                    a_min_prev = a_max_prev;
                }
                ApproxOrdering::Greater => {}
            }
            if verboser.is_enabled(Verbosity::Trace) {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "\t\tForward propagated interval to next step: {a_min_prev} <= a[k] <= {a_max_prev}."
                );
            }
            *a_max_k = a_max_prev;
            *a_min_k = a_min_prev;
        }
    }

    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "reach_set2: {}backward total elapsed time = {}.",
            if BIDIRECTION { "forward + " } else { "" },
            format_duration_human(verboser.elapsed())
        );
    }

    Ok(ReachSet2 { a_max, a_min })
}

/// Backward propagation to compute feasible `a[k]` bounds at the current step given the next step's `a[k+1]=a_next` bounds.  
/// a_b.0 * a[k+1] + a_b.1 * a[k] <= a_b.2
/// Returns (a_max_curr, a_min_curr)
fn backward_bound_a_next<const MAX: bool, const MIN: bool>(
    a_b: &mut [(f64, f64, f64)],
    a_next_mid: f64,
    lp_fea_tol: f64,
) -> (f64, f64) {
    let warm_start = Lp2dWarmStart {
        x0: (a_next_mid, LP_BOUND),
        skip: 2,
    };

    normalize_lp2d(a_b);
    let a_curr_max = if MAX {
        let (_, a_curr_max) = lp_2d_incre_max_y::<_, false>(
            a_b,
            &warm_start,
            &LpToleranceOptions::with_feas_tol(lp_fea_tol),
        );
        a_curr_max
    } else {
        f64::INFINITY
    };

    let a_curr_min = if MIN {
        // Better than transform the sign in the lp_2d_incre function.
        a_b.iter_mut().for_each(|(_, b, _)| {
            *b = -*b;
        });
        let (_, a_curr_min_neg) = lp_2d_incre_max_y::<_, false>(
            a_b,
            &warm_start,
            &LpToleranceOptions::with_feas_tol(lp_fea_tol),
        );
        -(a_curr_min_neg.min(0.0))
    } else {
        0.0
    };

    (a_curr_max, a_curr_min)
}

/// Builder for `ReachSet2Options`.
pub struct ReachSet2OptionsBuilder {
    /// Feasibility tolerance used by LP subproblems in reachable-set computation.
    pub lp_feas_tol: f64,
    /// Absolute tolerance for comparing interval bounds `a_max` and `a_min`.
    pub a_cmp_abs_tol: f64,
    /// Relative tolerance for comparing interval bounds `a_max` and `a_min`.
    pub a_cmp_rel_tol: f64,
    /// Verbosity level for diagnostics during reachability analysis.
    pub verbosity: Verbosity,
}

impl Default for ReachSet2OptionsBuilder {
    #[inline]
    fn default() -> Self {
        Self {
            lp_feas_tol: 1e-8,
            a_cmp_abs_tol: 1e-8,
            a_cmp_rel_tol: 1e-8,
            verbosity: Verbosity::default(),
        }
    }
}

impl ReachSet2OptionsBuilder {
    /// Create a new `ReachSet2OptionsBuilder` with default values.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the tolerance for checking the feasibility of the linear program.  
    /// The default value is 1E-8.
    pub fn lp_feas_tol(mut self, tol: f64) -> Self {
        self.lp_feas_tol = tol;
        self
    }

    /// Set the absolute tolerance for comparing `a_max` and `a_min` to determine whether the reachable set is empty (`a_max < a_min`) or degenerated (`a_max == a_min`).  
    /// Let `tol = max(a_cmp_abs_tol, a_cmp_rel_tol * max(|a_max|, |a_min|))`.  
    /// + If `a_max < a_min - tol`, then the reachable set is empty.  
    /// + If `a_max > a_min + tol`, then the reachable set is non-degenerated.  
    /// + Otherwise, the reachable set is degenerated into a single point.
    ///
    /// The default value is 1E-8.
    #[inline]
    pub fn a_cmp_abs_tol(mut self, tol: f64) -> Self {
        self.a_cmp_abs_tol = tol;
        self
    }

    /// Set the relative tolerance for comparing `a_max` and `a_min`. More details refer to `a_cmp_abs_tol`.
    /// The default value is 1E-8.
    #[inline]
    pub fn a_cmp_rel_tol(mut self, tol: f64) -> Self {
        self.a_cmp_rel_tol = tol;
        self
    }

    /// Set the verbosity level for logging. More details refer to `Verbosity`.  
    /// The default value is `Verbosity::Silent`.
    #[inline]
    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Build the `ReachSet2Options` from the builder where the validity of the options is checked.
    #[inline]
    pub fn build(self) -> Result<ReachSet2Options, CoppError> {
        self.validate()?;
        Ok(ReachSet2Options {
            lp_feas_tol: self.lp_feas_tol,
            a_cmp_abs_tol: self.a_cmp_abs_tol,
            a_cmp_rel_tol: self.a_cmp_rel_tol,
            verbosity: self.verbosity,
        })
    }

    /// This function checks the validity of the options and returns an error if any option is invalid.
    #[inline]
    pub fn validate(&self) -> Result<(), CoppError> {
        check_strictly_positive("ReachSet2OptionsBuilder", "lp_feas_tol", self.lp_feas_tol)?;
        check_abs_rel_tol(
            "ReachSet2OptionsBuilder",
            "a_cmp_abs_tol",
            self.a_cmp_abs_tol,
            "a_cmp_rel_tol",
            self.a_cmp_rel_tol,
        )?;
        Ok(())
    }
}

/// The options for `reach_set2`.
pub struct ReachSet2Options {
    pub(crate) lp_feas_tol: f64,
    pub(crate) a_cmp_abs_tol: f64,
    pub(crate) a_cmp_rel_tol: f64,
    pub(crate) verbosity: Verbosity,
}

impl ReachSet2Options {
    #[inline]
    pub fn lp_feas_tol(&self) -> f64 {
        self.lp_feas_tol
    }
    #[inline]
    pub fn a_cmp_abs_tol(&self) -> f64 {
        self.a_cmp_abs_tol
    }
    #[inline]
    pub fn a_cmp_rel_tol(&self) -> f64 {
        self.a_cmp_rel_tol
    }
    #[inline]
    pub fn verbosity(&self) -> Verbosity {
        self.verbosity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::copp2::stable::basic::Topp2ProblemBuilder;
    use crate::copp::copp2::stable::topp2_ra::topp2_ra;
    use crate::path::{add_symmetric_axial_limits_for_test, lissajous_path_for_test};
    use crate::robot::robot_core::Robot;

    #[test]
    fn test_reach_set2() -> Result<(), CoppError> {
        let dim = 7;
        let n: usize = 1000;

        let options = ReachSet2OptionsBuilder::new()
            .lp_feas_tol(1E-9)
            .a_cmp_abs_tol(1E-9)
            .a_cmp_rel_tol(1E-9)
            .verbosity(Verbosity::Summary)
            .build()?;

        let mut robot = Robot::with_capacity(dim, n);
        let mut rng = rand::rng();

        let (s, derivs, _, _) = lissajous_path_for_test(dim, n, &mut rng).map_err(|e| {
            CoppError::InvalidInput(
                "test_reach_set2_with_path_helper".into(),
                format!("failed to generate test path derivatives: {e}"),
            )
        })?;

        let dq = derivs.dq.as_ref().ok_or_else(|| {
            CoppError::InvalidInput(
                "test_reach_set2_with_path_helper".into(),
                "missing dq in path derivatives".into(),
            )
        })?;
        let ddq = derivs.ddq.as_ref().ok_or_else(|| {
            CoppError::InvalidInput(
                "test_reach_set2_with_path_helper".into(),
                "missing ddq in path derivatives".into(),
            )
        })?;
        let dddq = derivs.dddq.as_ref().ok_or_else(|| {
            CoppError::InvalidInput(
                "test_reach_set2_with_path_helper".into(),
                "missing dddq in path derivatives".into(),
            )
        })?;

        robot.with_s(&s.as_view())?;
        robot.with_q(
            &derivs.q.as_view(),
            &dq.as_view(),
            &ddq.as_view(),
            Some(&dddq.as_view()),
            0,
        )?;

        add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, Some(5.0)).map_err(|e| {
            CoppError::InvalidInput(
                "test_reach_set2_with_path_helper".into(),
                format!("failed to add symmetric axial limits: {e}"),
            )
        })?;

        let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;

        let reach_set_back = reach_set2_backward(&topp2_problem, &options)?;
        let reach_set_for = reach_set2_bidirectional(&topp2_problem, &options)?;
        let a_ra = topp2_ra(&topp2_problem, &options)?;

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "a_max_back.len() = {};",
            reach_set_back.a_max.len()
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "a_max_for.len() = {};",
            reach_set_for.a_max.len()
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "a_ra.len() = {};",
            a_ra.len()
        );

        Ok(())
    }
}
