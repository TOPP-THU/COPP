//! 2nd-order Convex-Objective Path Parameterization (COPP2) based on second-order cone programming (SOCP).
//!
//! # Method identity
//! This module implements the **optimization backend** for COPP2 by transforming path-parameterization
//! constraints/objectives into a Clarabel-compatible conic form and solving it with SOCP.
//!
//! # Discrete variables (local notation)
//! On a path grid `s[0..=n]`:
//! - `a[k]` denotes $\dot{s}_k^2$ (state variable, expected nonnegative in feasible solutions);
//! - decision vector is organized as `x = [a[0..=n], x_others]`, where `x_others` are auxiliary variables introduced by objective terms (e.g. reciprocal/soc slack variables);
//!
//! # High-level pipeline
//! 1. Validate interval and boundary consistency.
//! 2. Estimate capacities and assemble standard TOPP2 constraints.
//! 3. Add COPP2 objective-induced variables/cones (`Time`, `ThermalEnergy`, `TotalVariationTorque`, `Linear`).
//! 4. Build sparse matrices `A`, `P`, vector `q`, and solve by Clarabel.
//! 5. Apply status acceptance policy (`ClarabelOptions::is_allow`) and extract `a` when allowed.
//!
//! # API layering
//! - `copp2_socp`: strict/normal API, returns only accepted `a`.
//! - `copp2_socp_expert`: expert API, always returns full Clarabel solution for diagnosis.

use crate::copp::clarabel_backend::{ConstraintsClarabel, ObjConsClarabel};
use crate::copp::copp2::formulation::Copp2Problem;
use crate::copp::copp2::opt2::clarabel_constraints::{
    clarabel_standard_capacity_topp2, clarabel_standard_constraint_topp2,
};
use crate::copp::{ClarabelOptions, CoppObjective, clarabel_to_copp2_solution};
use crate::diag::{
    CoppError, DebugVerboser, SilentVerboser, SummaryVerboser, TraceVerboser, Verboser, Verbosity,
    format_duration_human,
};
use crate::robot::robot_core::{Robot, RobotBasic, RobotTorque};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{NonnegativeConeT, SecondOrderConeT};
use clarabel::solver::{DefaultSolution, DefaultSolver, IPSolver, SupportedConeT};
use core::f64;
use itertools::{Itertools, izip};
use nalgebra::{DMatrix, DVectorView};

#[cfg(test)]
use crate::copp::copp2::stable::basic::a_to_b_topp2;

/// Strict COPP2-SOCP API for production use.
///
/// # Purpose
/// Use this entry when caller only needs a valid trajectory profile `a` and treats
/// non-accepted solver statuses as hard failures.
///
/// # Contract
/// - Internally calls [`copp2_socp_expert`].
/// - Returns `Ok(a)` **iff** `options.is_allow(solution.status)` is `true`.
/// - Returns `Err(CoppError::ClarabelSolverStatus(...))` when status is not accepted.
///
/// # Returns
/// Returns accepted profile `a` for production usage.
///
/// # Errors
/// Returns `CoppError` for model/solver failures and when solver status is not accepted.
///
/// # Notes
/// For workflows requiring low-level diagnostics (`status`, iterate behavior, residual-related fields in
/// Clarabel solution), prefer [`copp2_socp_expert`].
pub fn copp2_socp<'a, M: RobotTorque>(
    problem: &Copp2Problem<'a, M>,
    options: &ClarabelOptions,
) -> Result<Vec<f64>, CoppError> {
    let (a_profile, solution) = copp2_socp_expert(problem, options)?;
    a_profile.ok_or_else(|| CoppError::ClarabelSolverStatus("copp2_socp".into(), solution.status))
}

/// Expert COPP2-SOCP API with full Clarabel solution exposure.
///
/// # Purpose
/// This API is intended for advanced users who need both:
/// - extracted high-level profile `Option<Vec<f64>>`, and
/// - raw solver result `DefaultSolution<f64>` for post-analysis.
///
/// # Return contract
/// - `Ok((Some(a), solution))`: status accepted by `options.is_allow(solution.status)`.
/// - `Ok((None, solution))`: solve finished but status not accepted by policy.
/// - `Err(...)`: true runtime failures only (input validation / model build / solver construction).
///
/// # Returns
/// Returns tuple `(Option<Vec<f64>>, DefaultSolution<f64>)` for diagnostic workflows.
///
/// # Errors
/// Returns `CoppError` only for real failures (input, model build, or solver runtime).
///
/// # Contract
/// - caller must handle `None` profile when status is not accepted;
/// - acceptance policy is fully controlled by `options.is_allow`.
///
/// # Verbosity behavior
/// Logging is layered by `options.verbosity()`:
/// - `Silent`: no algorithm logs;
/// - `Summary`: lifecycle milestones and elapsed time;
/// - `Debug`: assembly-level counters and stage summaries;
/// - `Trace`: fine-grained stage deltas and solver snapshot diagnostics.
pub fn copp2_socp_expert<'a, M: RobotTorque>(
    problem: &Copp2Problem<'a, M>,
    options: &ClarabelOptions,
) -> Result<(Option<Vec<f64>>, DefaultSolution<f64>), CoppError> {
    match options.verbosity() {
        Verbosity::Silent => copp2_socp_core(problem, (options, SilentVerboser)),
        Verbosity::Summary => copp2_socp_core(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => copp2_socp_core(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => copp2_socp_core(problem, (options, TraceVerboser::new())),
    }
}

/// Core implementation for COPP2-SOCP expert flow.
///
/// # Internal contract
/// `options_verboser` packs:
/// - `options`: acceptance policy and Clarabel numerical settings;
/// - `verboser`: concrete logger implementation chosen by external verbosity dispatch.
///
/// # Invariants
/// - decision-variable layout always starts with contiguous `a[0..=n]`;
/// - `q_object.len()` is treated as final `n_var` before solver build;
/// - extracted `a` is produced only through `clarabel_to_copp2_solution` when status is accepted.
fn copp2_socp_core<'a, M: RobotTorque>(
    problem: &Copp2Problem<'a, M>,
    options_verboser: (&ClarabelOptions, impl Verboser),
) -> Result<(Option<Vec<f64>>, DefaultSolution<f64>), CoppError> {
    let (options, mut verboser) = options_verboser;
    let (idx_s_start, idx_s_final) = problem.idx_s_interval;
    if verboser.is_enabled(Verbosity::Summary) {
        verboser.record_start_time();
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "\ncopp2_socp started: {} <= idx_s <= {}, objectives = {}, s_len = {}.",
            idx_s_start,
            idx_s_final,
            problem.objectives.len(),
            problem.s_len()
        );
    }
    if verboser.is_enabled(Verbosity::Trace) {
        let settings = options.clarabel_settings();
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: options snapshot -> allow(almost={}, max_iter={}, max_time={}, callback_term={}, insufficient_progress={}), tol_gap_rel={}, tol_feas={}, max_iter={}, verbose={}",
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
    let n = idx_s_final - idx_s_start;
    // Let x = [a[0], a[1], ..., a[n], x_others] \in R^{n+1+n_others}.
    // Step 1. Deal with constraints
    // Step 1.1 Compute the number of constraints
    let (cap_val_std, cap_b_std, cap_cone_std) =
        clarabel_standard_capacity_topp2(&problem.robot.constraints, problem.idx_s_interval);
    let (cap_val_obj, cap_b_obj, cap_cone_obj, n_vars) =
        clarabel_objective_capacity_copp2(n, problem.objectives, problem.robot);
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: capacity estimate std(val={cap_val_std}, b={cap_b_std}, cone={cap_cone_std}), obj(val={cap_val_obj}, b={cap_b_obj}, cone={cap_cone_obj}), n_vars={n_vars}."
        );
    }
    // s=b-A*x \in cone, where A[row[i],col[i]]=val[i], A \in R^{m*(n+1)}, b \in R^m, s \in R^m
    // -s=-b+A*x
    let mut row = Vec::<usize>::with_capacity(cap_val_std + cap_val_obj);
    let mut col = Vec::<usize>::with_capacity(cap_val_std + cap_val_obj);
    let mut val = Vec::<f64>::with_capacity(cap_val_std + cap_val_obj);
    let mut b = Vec::<f64>::with_capacity(cap_b_std + cap_b_obj);
    let mut cones = Vec::<SupportedConeT<f64>>::with_capacity(cap_cone_std + cap_cone_obj);
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: allocated capacities row/col/val/b/cones <= {}/{}/{}/{}/{}",
            cap_val_std + cap_val_obj,
            cap_val_std + cap_val_obj,
            cap_val_std + cap_val_obj,
            cap_b_std + cap_b_obj,
            cap_cone_std + cap_cone_obj
        );
    }
    // Step 1.2 set constraints
    let row_before_std = row.len();
    let col_before_std = col.len();
    let val_before_std = val.len();
    let b_before_std = b.len();
    let cones_before_std = cones.len();
    clarabel_standard_constraint_topp2(
        &problem.as_topp2_problem(),
        (&mut row, &mut col, &mut val, &mut b, &mut cones),
        &verboser,
    );
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: standard-constraints delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}",
            row.len() - row_before_std,
            col.len() - col_before_std,
            val.len() - val_before_std,
            b.len() - b_before_std,
            cones.len() - cones_before_std
        );
    }
    // Step 2. set objective
    // Step 2.1. determine whether eta=1/sqrt(a) is needed.
    let row_before_sqrt = row.len();
    let col_before_sqrt = col.len();
    let val_before_sqrt = val.len();
    let b_before_sqrt = b.len();
    let cones_before_sqrt = cones.len();
    let n_var_old = clarabel_sqrt_a_copp2(
        n,
        problem.objectives,
        (&mut row, &mut col, &mut val, &mut b, &mut cones),
    );
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: sqrt-a stage delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}, n_var_old={}",
            row.len() - row_before_sqrt,
            col.len() - col_before_sqrt,
            val.len() - val_before_sqrt,
            b.len() - b_before_sqrt,
            cones.len() - cones_before_sqrt,
            n_var_old
        );
    }
    let mut q_object = Vec::<f64>::with_capacity(n_vars);
    q_object.resize(n_var_old, 0.0);
    // Step 2.2. add constraints and objective for each term in the objective.
    let row_before_obj = row.len();
    let col_before_obj = col.len();
    let val_before_obj = val.len();
    let b_before_obj = b.len();
    let cones_before_obj = cones.len();
    let q_before_obj = q_object.len();
    clarable_objective_copp2(
        problem,
        (
            &mut row,
            &mut col,
            &mut val,
            &mut b,
            &mut cones,
            &mut q_object,
        ),
    )?;
    if verboser.is_enabled(Verbosity::Trace) {
        let (q_min, q_max) = q_object
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: objective stage delta row/col/val/b/cones/q = +{}/+{}/+{}/+{}/+{}/+{}, q_range=[{}, {}]",
            row.len() - row_before_obj,
            col.len() - col_before_obj,
            val.len() - val_before_obj,
            b.len() - b_before_obj,
            cones.len() - cones_before_obj,
            q_object.len() - q_before_obj,
            q_min,
            q_max
        );
    }
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: after objective assembly row={}, col={}, val={}, b={}, cones={}, q={}",
            row.len(),
            col.len(),
            val.len(),
            b.len(),
            cones.len(),
            q_object.len()
        );
    }
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: ready to solve with row/col/val/b/cones = {}/{}/{}/{}/{} and n_var = {}.",
            row.len(),
            col.len(),
            val.len(),
            b.len(),
            cones.len(),
            q_object.len()
        );
    }
    // Step 2.3 build the constraints
    let n_var = q_object.len();
    let a_csc = CscMatrix::new_from_triplets(b.len(), n_var, row, col, val);
    let p_object = CscMatrix::<f64>::zeros((n_var, n_var));
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: matrix built with m={}, n={}, A.nnz={}, P.nnz={}",
            b.len(),
            n_var,
            a_csc.nnz(),
            p_object.nnz()
        );
    }
    // Step 3. solve the SOCP problem
    let settings = options.clarabel_settings().clone();
    let mut solver = DefaultSolver::<f64>::new(&p_object, &q_object, &a_csc, &b, &cones, settings)
        .map_err(|e| CoppError::ClarabelSolverError("copp2_socp".into(), e))?;
    solver.solve();
    let solution = solver.solution;
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: solve done, status = {:?}, elapsed = {}.",
            solution.status,
            format_duration_human(verboser.elapsed())
        );
    }
    if verboser.is_enabled(Verbosity::Trace) {
        let show = solution.x.len().min(3);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: solution x_len={}, head={:?}",
            solution.x.len(),
            &solution.x[0..show]
        );
    }
    let a_profile = if options.is_allow(solution.status) {
        Some(clarabel_to_copp2_solution(problem.s_len(), &solution))
    } else {
        None
    };
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp2_socp: allow(status)={}, extracted_profile={}",
            options.is_allow(solution.status),
            if a_profile.is_some() {
                "Some(a)"
            } else {
                "None"
            }
        );
    }
    Ok((a_profile, solution))
}

/// Determine the number of clarabel's capacity for the objective in COPP2.
fn clarabel_objective_capacity_copp2<M: RobotBasic>(
    n: usize,
    objective: &[CoppObjective],
    robot: &Robot<M>,
) -> (usize, usize, usize, usize) {
    let flag_need_eta = objective.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::Time(_) | CoppObjective::ThermalEnergy(_, _)
        )
    });
    // Step 1. sqrt(a[k]) >= eta[k] >= 0
    // num_val <= 4*(n+1), num_b <= 4*(n+1), num_cones <= n+2
    let (mut capacity_val, mut capacity_b, mut capacity_cones, mut n_vars) = if flag_need_eta {
        (4 * (n + 1), 4 * (n + 1), n + 2, 2 * (n + 1))
    } else {
        (0, 0, 0, n + 1)
    };
    // Step 2. objective function
    let dim = robot.dim();
    for obj in objective {
        match obj {
            CoppObjective::Time(_) => {
                // num_val <= 6*n, num_b <= 3*n, num_cones <= n, n_var <= n
                capacity_val += 6 * n;
                capacity_b += 3 * n;
                capacity_cones += n;
                n_vars += n;
            }
            CoppObjective::ThermalEnergy(_, _) => {
                // num_val <= (6+2*dim)*n, num_b <= (dim+2)*n, num_cones <= n, n_var <= n
                capacity_val += (6 + 2 * dim) * n;
                capacity_b += (dim + 2) * n;
                capacity_cones += n;
                n_vars += n;
            }
            CoppObjective::TotalVariationTorque(_, _) => {
                // num_val <= 8*dim*n, num_b <= 2*dim*n, num_cones <= 1, n_var <= dim*n
                capacity_val += 8 * dim * n;
                capacity_b += 2 * dim * n;
                capacity_cones += 1;
                n_vars += dim * n;
            }
            _ => {}
        }
    }
    (capacity_val, capacity_b, capacity_cones, n_vars)
}

/// Add the constraints for sqrt(a) >= eta in COPP2 optimization.  
/// x = [a[0], a[1], ..., a[n], eta[0], eta[1], ..., eta[n], ...] \in R^{2*(n+1)+...}.
/// sqrt(a[k]) >= eta[k] >= 0  
/// num_val <= 4*(n+1), num_b <= 4*(n+1), num_cones <= n+2  
/// Return the len of the new x: n+1 or 2*(n+1)
fn clarabel_sqrt_a_copp2(
    n: usize,
    objective: &[CoppObjective],
    constraints: ConstraintsClarabel,
) -> usize {
    let (row, col, val, b, cones) = constraints;
    for obj in objective {
        match obj {
            CoppObjective::Time(_) | CoppObjective::ThermalEnergy(_, _) => {
                // eta >= 0
                // A*x-b = -s = -1*eta[k] <= 0
                row.extend(b.len()..b.len() + n + 1);
                col.extend((n + 1)..(2 * (n + 1)));
                val.resize(val.len() + n + 1, -1.0);
                b.resize(b.len() + n + 1, 0.0);
                cones.push(NonnegativeConeT(n + 1));
                // sqrt(a) >= eta
                // eta^2 <= a
                // eta^2 + (a - 0.25)^2 <= (a + 0.25)^2
                // -A*x+b = s = [a+0.25, a-0.25, eta] \in SOC
                row.extend(b.len()..b.len() + 3 * (n + 1));
                val.resize(val.len() + 3 * (n + 1), -1.0);
                cones.resize(cones.len() + n + 1, SecondOrderConeT(3));
                for k in 0..=n {
                    // row.extend(b.len()..b.len() + 3);
                    col.extend([k, k, k + n + 1]);
                    // val.resize(val.len() + 3, -1.0);
                    b.extend([0.25, -0.25, 0.0]);
                    // cones.push(SecondOrderConeT(3));
                }
                return 2 * (n + 1);
            }
            _ => {}
        }
    }
    n + 1
}

/// Add the constraints and objective for Time in COPP2 optimization.  
/// num_val <= 6*n, num_b <= 3*n, num_cones <= n, n_var <= n
fn clarabel_objective_time_copp2(
    s: &[f64],
    weight: f64,
    objective_constraints: ObjConsClarabel,
) -> bool {
    if weight < 0.0 {
        return false;
    }
    let (row, col, val, b, cones, q_object) = objective_constraints;
    // objective: minimize 2 * weight * \sum (s[k+1]-s[k]) / (eta[k] + eta[k+1])
    // Let: 1 / (eta[k] + eta[k+1]) <= 4 * t[k]
    // objective: minimize 8 * weight * \sum (s[k+1]-s[k]) * t[k]
    let weight = 8.0 * weight;
    let n_var_old = q_object.len();
    // objective: minimize weight * \sum (s[k+1]-s[k]) * t[k]
    q_object.extend(s.windows(2).map(|s_pair| weight * (s_pair[1] - s_pair[0])));
    // t[k] * (eta[k] + eta[k+1]) >= 1
    // (eta[k] + eta[k+1] + t[k])^2 >= (eta[k] + eta[k+1] - t[k])^2 + 1
    // -A*x+b = s = [eta[k] + eta[k+1] + t[k], eta[k] + eta[k+1] - t[k], 1] \in SOC
    let len = s.len();
    for k in 0..(len - 1) {
        // eta[k] + eta[k+1] + t[k]
        row.resize(row.len() + 3, b.len());
        col.extend([len + k, len + k + 1, n_var_old + k]);
        val.extend([-1.0, -1.0, -1.0]);
        b.push(0.0);
        // eta[k] + eta[k+1] - t[k]
        row.resize(row.len() + 3, b.len());
        col.extend([len + k, len + k + 1, n_var_old + k]);
        val.extend([-1.0, -1.0, 1.0]);
        b.push(0.0);
        // 1
        b.push(1.0);
        // cones.push(SecondOrderConeT(3));
    }
    cones.resize(cones.len() + len - 1, SecondOrderConeT(3));
    true
}

/// Add the constraints and objective for ThermalEnergy in COPP2 optimization.  
/// num_val <= (6+2*dim)*n, num_b <= (dim+2)*n, num_cones <= n, n_var <= n
fn clarabel_objective_thermal_energy_copp2(
    s: &[f64],
    weight: f64,
    normalize: &[f64],
    coeffs_torque: &(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>),
    objective_constraints: ObjConsClarabel,
) -> bool {
    if weight < 0.0 {
        return false;
    }
    let (row, col, val, b, cones, q_object) = objective_constraints;
    // minimize: 2 * weight * \sum (s[k+1]-s[k]) / (sqrt(a[k]) + sqrt(a[k+1])) * (tau[i][k] * normalize[i]) ^ 2
    // minimize: 2 * weight * \sum (s[k+1]-s[k]) / (eta[k] + eta[k+1]) * (tau[i][k] * normalize[i]) ^ 2
    // Let: \sum_i (tau[i][k] * normalize[i]) ^ 2 / (eta[k] + eta[k+1]) <= 4 * t[k]
    let len = s.len();
    let mut coeff_a_curr = coeffs_torque.0.clone();
    let mut coeff_a_next = coeffs_torque.1.clone();
    let mut coeff_g = coeffs_torque.2.clone();
    // objective: minimize 8 * weight * \sum (s[k+1]-s[k]) * t[k]
    let weight = 8.0 * weight;
    let n_var_old = q_object.len();
    // objective: minimize weight * \sum (s[k+1]-s[k]) * t[k]
    q_object.extend(s.windows(2).map(|s_pair| weight * (s_pair[1] - s_pair[0])));
    // \sum_i (tau[i][k] * normalize[i]) ^ 2 <= 4 * t[k] * (eta[k] + eta[k+1])
    let dim = coeff_a_curr.nrows();
    // tau[i][k] = coeff_a_curr[i][k] * a[k] + coeff_a_next[i][k] * a[k+1] + coeff_g[i][k]
    if normalize.len() != dim {
        return false;
    }
    let normalize = DVectorView::from_slice(normalize, dim);
    for mut col in coeff_a_curr.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_a_next.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_g.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    // Now: tau[i][k] * normalize[i] = coeff_a_curr[i][k] * a[k] + coeff_a_next[i][k] * a[k+1] + coeff_g[i][k]

    // (eta[k] + eta[k+1] - t[k])^2 + \sum_i (tau[i][k] * normalize[i]) ^ 2 <= (eta[k] + eta[k+1] + t[k])^2
    // -A*x+b = s = [eta[k] + eta[k+1] + t[k], eta[k] + eta[k+1] - t[k], tau[0][k] * normalize[0], tau[1][k] * normalize[1], ...] \in SOC
    for (k, (col_a_curr, col_a_next, col_g)) in izip!(
        coeff_a_curr.column_iter(),
        coeff_a_next.column_iter(),
        coeff_g.column_iter()
    )
    .enumerate()
    {
        // eta[k] + eta[k+1] + t[k]
        row.resize(row.len() + 3, b.len());
        col.extend([len + k, len + k + 1, n_var_old + k]);
        val.extend([-1.0, -1.0, -1.0]);
        b.push(0.0);
        // eta[k] + eta[k+1] - t[k]
        row.resize(row.len() + 3, b.len());
        col.extend([len + k, len + k + 1, n_var_old + k]);
        val.extend([-1.0, -1.0, 1.0]);
        b.push(0.0);
        // tau[i][k] * normalize[i] = col_a_curr[i] * x[k] + col_a_next[i] * x[k+1] + col_g[i]
        for (&v_a_curr, &v_a_next, &v_g) in
            izip!(col_a_curr.iter(), col_a_next.iter(), col_g.iter())
        {
            row.resize(row.len() + 2, b.len());
            col.extend([k, k + 1]);
            val.extend([v_a_curr, v_a_next]);
            b.push(v_g);
        }
    }
    cones.resize(cones.len() + len - 1, SecondOrderConeT(dim + 2));
    true
}

/// Add the constraints and objective for TotalVariationTorque in COPP2 optimization.  
/// num_val <= 8*dim*n, num_b <= 2*dim*n, num_cones <= 1, n_var <= dim*n
fn clarabel_objective_tv_torque_copp2(
    weight: f64,
    normalize: &[f64],
    coeffs_torque: &(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>),
    objective_constraints: ObjConsClarabel,
) -> bool {
    if weight < 0.0 {
        return false;
    }
    let (row, col, val, b, cones, q_object) = objective_constraints;
    // minimize: weight * \sum |tau[i][k+1]-tau[i][k]| * normalize[i]
    // Let: |tau[i][k+1]-tau[i][k]| * normalize[i] <= t[i][k]
    let mut coeff_a_curr = coeffs_torque.0.clone();
    let mut coeff_a_next = coeffs_torque.1.clone();
    let mut coeff_g = coeffs_torque.2.clone();
    let dim = coeff_a_curr.nrows();
    if normalize.len() != dim {
        return false;
    }
    // tau[i][k] = coeff_a_curr[i][k] * a[k] + coeff_a_next[i][k] * a[k+1] + coeff_g[i][k]
    let normalize = DVectorView::from_slice(normalize, dim);
    for mut col in coeff_a_curr.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_a_next.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_g.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    // Now: tau[i][k] * normalize[i] = coeff_a_curr[i][k] * a[k] + coeff_a_next[i][k] * a[k+1] + coeff_g[i][k]
    // (tau[i][k+1]-tau[i][k]) * normalize[i] = (coeff_a_curr[i][k+1] * a[k+1] + coeff_a_next[i][k+1] * a[k+2] + coeff_g[i][k+1]) - (coeff_a_curr[i][k] * a[k] + coeff_a_next[i][k] * a[k+1] + coeff_g[i][k])
    // = -coeff_a_curr[i][k] * a[k] + (coeff_a_curr[i][k+1] - coeff_a_next[i][k]) * a[k+1] + coeff_a_next[i][k+1] * a[k+2] + (coeff_g[i][k+1] - coeff_g[i][k])

    let n_b_old = b.len();
    // A*x-b = -s = -coeff_a_curr[i][k] * a[k] + (coeff_a_curr[i][k+1] - coeff_a_next[i][k]) * a[k+1] + coeff_a_next[i][k+1] * a[k+2] + (coeff_g[i][k+1] - coeff_g[i][k]) - t[i][k] <= 0
    // A*x-b = -s = coeff_a_curr[i][k] * a[k] - (coeff_a_curr[i][k+1] - coeff_a_next[i][k]) * a[k+1] - coeff_a_next[i][k+1] * a[k+2] - (coeff_g[i][k+1] - coeff_g[i][k]) - t[i][k] <= 0
    let mut buffer0 = vec![0.0; dim];
    let mut buffer1 = vec![0.0; dim];
    for (k, ((col_a_curr, col_b_curr, col_g_curr), (col_a_next, col_b_next, col_g_next))) in izip!(
        coeff_a_curr.column_iter(),
        coeff_a_next.column_iter(),
        coeff_g.column_iter()
    )
    .tuple_windows()
    .enumerate()
    {
        // dtau[i] * normalize[i] = -col_a_curr[i] * a[k] + (col_a_next[i] - col_b_curr[i]) * a[k+1] + col_b_next[i] * a[k+2] + (col_g_next[i] - col_g_curr[i])
        buffer0.clear();
        buffer1.clear();
        buffer0.extend(
            col_a_next
                .iter()
                .zip(col_b_curr.iter())
                .map(|(&v_a_next, &v_b_curr)| v_b_curr - v_a_next),
        );
        buffer1.extend(
            col_g_curr
                .iter()
                .zip(col_g_next.iter())
                .map(|(&v_g_curr, &v_g_next)| v_g_curr - v_g_next),
        );
        // dtau[i] * normalize[i] = -col_a_curr[i] * a[k] + buffer0[i] * a[k+1] + col_b_next[i] * a[k+2] + buffer1[i]

        let n_var_old = q_object.len();
        for (i, (&v0, &v1, &v_a_curr, &v_b_next)) in izip!(
            buffer0.iter(),
            buffer1.iter(),
            col_a_curr.iter(),
            col_b_next.iter()
        )
        .enumerate()
        {
            // -col_a_curr[i] * a[k] + buffer0[i] * a[k+1] + col_b_next[i] * a[k+2] + buffer1[i] <= t[i][k]
            // A*x-b = -s = -col_a_curr[i] * a[k] + buffer0[i] * a[k+1] + col_b_next[i] * a[k+2] + buffer1[i] - t[i][k] <= 0
            row.resize(row.len() + 4, b.len());
            col.extend([k, k + 1, k + 2, n_var_old + i]);
            val.extend([-v_a_curr, v0, v_b_next, -1.0]);
            b.push(v1);

            // -(-col_a_curr[i] * a[k] + buffer0[i] * a[k+1] + col_b_next[i] * a[k+2]) <= t[i][k]
            // A*x-b = -s = -(-col_a_curr[i] * a[k] + buffer0[i] * a[k+1] + col_b_next[i] * a[k+2]) - t[i][k] <= 0
            row.resize(row.len() + 4, b.len());
            col.extend([k, k + 1, k + 2, n_var_old + i]);
            val.extend([v_a_curr, -v0, -v_b_next, -1.0]);
            b.push(-v1);
        }

        // objective: minimize weight * \sum t[i][k]
        q_object.resize(q_object.len() + dim, weight);
    }
    cones.push(NonnegativeConeT(b.len() - n_b_old));
    true
}

/// Add the constraints and objective for Linear in COPP2 optimization.
fn clarabel_objective_linear_copp2(
    s: &[f64],
    weight: f64,
    alpha: &[f64],
    beta: &[f64],
    q_object: &mut [f64],
) -> bool {
    if alpha.len() != s.len() || beta.len() != s.len() - 1 {
        return false;
    }
    // objective: minimize weight * \sum (alpha[k]*a[k] + beta[k]*b[k])
    // weight * \sum alpha[k]*a[k] + 0.5 * beta[k]*(a[k+1]-a[k])/(s[k+1]-s[k])
    for (va, q) in alpha.iter().zip(q_object.iter_mut()) {
        // weight * \sum alpha[k]*a[k]
        *q += weight * va;
    }
    for (s_pair, vb, q_curr) in izip!(s.windows(2), beta.iter(), q_object.iter_mut()) {
        // weight * \sum 0.5 * beta[k]*(a[k+1]-a[k])/(s[k+1]-s[k])
        *q_curr -= 0.5 * weight * vb / (s_pair[1] - s_pair[0]);
    }
    for (s_pair, vb, q_next) in izip!(s.windows(2), beta.iter(), q_object.iter_mut().skip(1)) {
        // weight * \sum 0.5 * beta[k]*(a[k+1]-a[k])/(s[k+1]-s[k])
        *q_next += 0.5 * weight * vb / (s_pair[1] - s_pair[0]);
    }
    true
}

fn clarable_objective_copp2<M: RobotTorque>(
    problem: &Copp2Problem<M>,
    objective_constraints: ObjConsClarabel,
) -> Result<(), CoppError> {
    let (row, col, val, b, cones, q_object) = objective_constraints;
    let s = problem
        .robot
        .constraints
        .s_vec(problem.idx_s_interval.0, problem.idx_s_interval.1 + 1)?;
    let coeffs_torque = if problem.objectives.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::ThermalEnergy(_, _) | CoppObjective::TotalVariationTorque(_, _)
        )
    }) {
        // shape: (dim, n) since there are n+1 a and n b.
        problem.robot.torque2_coeff_a(
            problem.idx_s_interval.0,
            problem.idx_s_interval.1 - problem.idx_s_interval.0,
        )
    } else {
        (
            DMatrix::<f64>::zeros(0, 0),
            DMatrix::<f64>::zeros(0, 0),
            DMatrix::<f64>::zeros(0, 0),
        )
    };
    for obj in problem.objectives {
        match obj {
            CoppObjective::Time(weight) => {
                if !clarabel_objective_time_copp2(&s, *weight, (row, col, val, b, cones, q_object))
                {
                    return Err(CoppError::InvalidInput(
                        "copp2_socp".into(),
                        "Invalid Time objective.".into(),
                    ));
                }
            }
            CoppObjective::ThermalEnergy(weight, normalize) => {
                if !clarabel_objective_thermal_energy_copp2(
                    &s,
                    *weight,
                    normalize,
                    &coeffs_torque,
                    (row, col, val, b, cones, q_object),
                ) {
                    return Err(CoppError::InvalidInput(
                        "copp2_socp".into(),
                        "Invalid ThermalEnergy objective.".into(),
                    ));
                }
            }
            CoppObjective::TotalVariationTorque(weight, normalize) => {
                if !clarabel_objective_tv_torque_copp2(
                    *weight,
                    normalize,
                    &coeffs_torque,
                    (row, col, val, b, cones, q_object),
                ) {
                    return Err(CoppError::InvalidInput(
                        "copp2_socp".into(),
                        "Invalid TotalVariationTorque objective.".into(),
                    ));
                }
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                if !clarabel_objective_linear_copp2(&s, *weight, alpha, beta, q_object) {
                    return Err(CoppError::InvalidInput(
                        "copp2_socp".into(),
                        "Invalid Linear objective.".into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Compute the objective value for COPP2 optimization.
#[cfg(test)]
pub(crate) fn objective_value_copp2_opt<M: RobotTorque>(
    robot: &Robot<M>,
    start_idx_s: usize,
    objective: &[CoppObjective],
    a_profile: &[f64],
) -> (f64, Vec<f64>) {
    let Ok(s) = robot
        .constraints
        .s_vec(start_idx_s, start_idx_s + a_profile.len())
    else {
        return (f64::INFINITY, vec![0.0; objective.len()]);
    };
    if a_profile.len() != s.len() {
        return (f64::INFINITY, vec![0.0; objective.len()]);
    }
    let b_profile = a_to_b_topp2(&s, a_profile);
    let torque = if objective.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::ThermalEnergy(_, _) | CoppObjective::TotalVariationTorque(_, _)
        )
    }) {
        let torque_result =
            robot.get_torque_with_ab(&a_profile[0..a_profile.len() - 1], &b_profile, start_idx_s);
        match torque_result {
            Ok(torque) => torque,
            _ => return (f64::INFINITY, vec![0.0; objective.len()]),
        }
    } else {
        DMatrix::<f64>::zeros(0, 0)
    };
    let a_sqrt = if objective.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::Time(_) | CoppObjective::ThermalEnergy(_, _)
        )
    }) {
        a_profile.iter().map(|a| a.sqrt()).collect()
    } else {
        Vec::new()
    };
    let mut obj_val = Vec::with_capacity(objective.len());
    let mut obj_val_total = 0.0;
    for obj in objective {
        match obj {
            CoppObjective::Time(weight) => {
                let obj_here = objective_value_time_copp2(&s, &a_sqrt);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
            CoppObjective::ThermalEnergy(weight, normalize) => {
                let obj_here =
                    objective_value_thermal_energy_copp2(&s, &a_sqrt, &torque, normalize);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
            CoppObjective::TotalVariationTorque(weight, normalize) => {
                let obj_here = objective_value_tv_torque_copp2(&torque, normalize);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                let obj_here = objective_value_linear_copp2(&s, a_profile, alpha, beta);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
        }
    }
    (obj_val_total, obj_val)
}

/// Compute the time value in COPP2 optimization.  
/// Input: s, a_sqrt = sqrt(a)  
#[cfg(test)]
#[inline(always)]
fn objective_value_time_copp2(s: &[f64], a_sqrt: &[f64]) -> f64 {
    // objective: minimize 2 * weight * \sum (s[k+1]-s[k]) / (sqrt(a[k]) + sqrt(a[k+1]))
    let mut objective = 0.0;
    for (s_pair, a_sqrt_pair) in s.windows(2).zip(a_sqrt.windows(2)) {
        objective += (s_pair[1] - s_pair[0]) / (a_sqrt_pair[0] + a_sqrt_pair[1]);
    }
    2.0 * objective
}

/// Compute the thermal energy value in COPP2 optimization.
#[cfg(test)]
#[inline(always)]
fn objective_value_thermal_energy_copp2(
    s: &[f64],
    a_sqrt: &[f64],
    torque: &DMatrix<f64>,
    normalize: &[f64],
) -> f64 {
    // minimize: 2 * \sum (s[k+1]-s[k]) / (sqrt(a[k]) + sqrt(a[k+1])) * (tau[i][k] * normalize[i]) ^ 2
    let mut objective = 0.0;
    for (s_pair, a_sqrt_pair, torque_col) in
        izip!(s.windows(2), a_sqrt.windows(2), torque.column_iter())
    {
        let mut sum = 0.0;
        for (torque, normal) in torque_col.iter().zip(normalize.iter()) {
            sum += (torque * normal).powi(2);
        }
        objective += (s_pair[1] - s_pair[0]) / (a_sqrt_pair[0] + a_sqrt_pair[1]) * sum;
    }
    2.0 * objective
}

/// Compute the total variation of torque value in COPP2 optimization.
#[cfg(test)]
#[inline(always)]
fn objective_value_tv_torque_copp2(torque: &DMatrix<f64>, normalize: &[f64]) -> f64 {
    // minimize: weight * \sum |tau[i][k+1]-tau[i][k]| * normalize[i]
    let mut objective = 0.0;
    for (torque_col_curr, torque_col_next) in torque.column_iter().tuple_windows() {
        for (torque_prev, torque_next, normal) in izip!(
            torque_col_curr.iter(),
            torque_col_next.iter(),
            normalize.iter()
        ) {
            objective += (torque_next - torque_prev).abs() * normal;
        }
    }
    objective
}

/// Compute the objective value for Linear in COPP2 optimization.
#[cfg(test)]
#[inline(always)]
fn objective_value_linear_copp2(s: &[f64], a_profile: &[f64], alpha: &[f64], beta: &[f64]) -> f64 {
    // objective: minimize \sum (alpha[k]*a[k] + beta[k]*b[k])
    // \sum alpha[k]*a[k] + 0.5*beta[k]*(a[k+1]-a[k])/(s[k+1]-s[k])
    let mut objective = 0.0;
    for (a_curr, alpha_curr) in a_profile.iter().zip(alpha.iter()) {
        // alpha[k]*a[k]
        objective += a_curr * alpha_curr;
    }
    for (a_pair, s_pair, beta_curr) in izip!(a_profile.windows(2), s.windows(2), beta.iter()) {
        // 0.5*beta[k]*(a[k+1]-a[k])/(s[k+1]-s[k])
        objective += 0.5 * beta_curr * (a_pair[1] - a_pair[0]) / (s_pair[1] - s_pair[0]);
    }
    objective
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::copp2::stable::basic::{
        Copp2ProblemBuilder, Topp2ProblemBuilder, s_to_t_topp2,
    };
    use crate::copp::copp2::stable::reach_set2::{ReachSet2Options, ReachSet2OptionsBuilder};
    use crate::copp::copp2::stable::topp2_ra::topp2_ra;
    use crate::copp::{ClarabelOptions, ClarabelOptionsBuilder};
    use crate::path::{add_symmetric_axial_limits_for_test, lissajous_path_for_test};
    use crate::robot::demo::Plannar2LinkEnd;
    use crate::robot::robot_core::Robot;
    use core::panic;
    use std::time::Instant;
    use std::vec;

    #[test]
    fn test_copp2_socp_only_time() -> Result<(), CoppError> {
        run_test_copp2_socp_only_time_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average 100 experiments: tc_ra = 0.2005 ms, tc_lp = 29.7361 ms, tc_qp = 166.5122 ms, tf_ra = 4.766745, tf_lp = 4.766745, tf_qp = 4.766735
    #[test]
    #[ignore = "slow"]
    fn test_copp2_socp_only_time_robust() -> Result<(), CoppError> {
        run_test_copp2_socp_only_time_repeated(100, true)?;
        Ok(())
    }

    #[test]
    fn test_copp2_socp() -> Result<(), CoppError> {
        run_test_copp2_socp_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average 100 experiments:
    //  Case 0: tc=161.691ms, obj=[4.824561313613876, 2576.8873693126284, 71.69242263342399, -1.048938713665848e-14]
    //  Case 1: tc=200.959ms, obj=[4.830121022357044, 2576.928260526994, 66.14588916597324, -7.651101974204267e-15]
    //  Case 2: tc=218.223ms, obj=[4.830230167437296, 2576.9529624501106, 66.12305109242398, -1.09470765785602e-14]
    //  Case 3: tc=93.588ms, obj=[361.45110144118144, 194870.1172166501, 48.05756488144189, 9.743247875171334e-16]
    //  Case 4: tc=164.505ms, obj=[4.824561266838617, 2576.887346690997, 71.69241993761281, 2.7478852526741092e-14]
    #[test]
    #[ignore = "slow"]
    fn test_copp2_socp_robust() -> Result<(), CoppError> {
        run_test_copp2_socp_repeated(100, true)?;
        Ok(())
    }

    fn run_test_copp2_socp_repeated(n_exp: usize, flag_print_step: bool) -> Result<(), CoppError> {
        let options_socp = ClarabelOptionsBuilder::new()
            .allow_almost_solved(true)
            .build()?;

        let mut tc_sum_case0 = 0.0;
        let mut tc_sum_case1 = 0.0;
        let mut tc_sum_case2 = 0.0;
        let mut tc_sum_case3 = 0.0;
        let mut tc_sum_case4 = 0.0;
        let mut obj_sum_case0 = vec![0.0; 4];
        let mut obj_sum_case1 = vec![0.0; 4];
        let mut obj_sum_case2 = vec![0.0; 4];
        let mut obj_sum_case3 = vec![0.0; 4];
        let mut obj_sum_case4 = vec![0.0; 4];

        for i_exp in 0..n_exp {
            let n: usize = 1000;
            let mut robot = Robot::with_capacity(Plannar2LinkEnd::new(1.0, 1.0, 1.0, 1.0), n);
            let dim = robot.dim();

            let mut rng = rand::rng();
            let (s, derivs, omega, phi) =
                lissajous_path_for_test(dim, n, &mut rng).expect("random range is valid");
            robot.with_s(&s.as_view())?;
            robot.with_q(
                &derivs.q.as_view(),
                &derivs.dq.as_ref().unwrap().as_view(),
                &derivs.ddq.as_ref().unwrap().as_view(),
                derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
                0,
            )?;
            add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, None)?;

            // Test different objectives in COPP2 optimization
            let objectives_test = [
                CoppObjective::Time(1.0),
                CoppObjective::ThermalEnergy(1.0, &vec![1.0; dim]),
                CoppObjective::TotalVariationTorque(1.0, &vec![1.0; dim]),
                CoppObjective::Linear(1.0, &vec![0.0; n], &vec![1.0; n - 1]),
            ];
            let a_feasible = vec![0.0; n];

            // Case 0: Time only
            let mut copp2_problem = Copp2ProblemBuilder::new(
                &robot,
                (0, n - 1),
                (0.0, 0.0),
                &[CoppObjective::Time(1.0)],
            )
            .build()?;
            let start = Instant::now();
            let mut a_case0 = copp2_socp(&copp2_problem, &options_socp)?;
            let tc_copp2_case0 = start.elapsed().as_secs_f64() * 1E3;
            robot
                .constraints
                .project_to_feasible_topp2(&mut a_case0, &a_feasible, 0)?;
            let (_, obj_case0) = objective_value_copp2_opt(&robot, 0, &objectives_test, &a_case0);

            // Case 1: Time and ThermalEnergy
            let obj_case1 = [
                CoppObjective::Time(1.0),
                CoppObjective::ThermalEnergy(1.0, &vec![1.0; dim]),
            ];
            copp2_problem.objectives = &obj_case1;
            let start = Instant::now();
            let mut a_case1 = copp2_socp(&copp2_problem, &options_socp)?;
            let tc_copp2_case1 = start.elapsed().as_secs_f64() * 1E3;
            robot
                .constraints
                .project_to_feasible_topp2(&mut a_case1, &a_feasible, 0)?;
            let (_, obj_case1) = objective_value_copp2_opt(&robot, 0, &objectives_test, &a_case1);
            if obj_case1[0] < obj_case0[0] - 1E-3 || obj_case1[1] - 1E-3 > obj_case0[1] {
                let (tf_case0, _) = s_to_t_topp2(s.as_slice(), &a_case0, 0.0);
                let (tf_case1, _) = s_to_t_topp2(s.as_slice(), &a_case1, 0.0);
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 0: obj_time = {:.6}, obj_thermal_energy = {:.6}, tf = {:.6}",
                    obj_case0[0],
                    obj_case0[1],
                    tf_case0
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {:.6}, obj_thermal_energy = {:.6}, tf = {:.6}",
                    obj_case1[0],
                    obj_case1[1],
                    tf_case1
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Interesting... Cases 0 and 1"
                );
            }

            // Case 2: Time and More ThermalEnergy
            let obj_case2 = [
                CoppObjective::Time(1.0),
                CoppObjective::ThermalEnergy(10.0, &vec![1.0; dim]),
            ];
            copp2_problem.objectives = &obj_case2;
            let start = Instant::now();
            let mut a_case2 = copp2_socp(&copp2_problem, &options_socp)?;
            let tc_copp2_case2 = start.elapsed().as_secs_f64() * 1E3;
            robot
                .constraints
                .project_to_feasible_topp2(&mut a_case2, &a_feasible, 0)?;
            let (_, obj_case2) = objective_value_copp2_opt(&robot, 0, &objectives_test, &a_case2);
            if obj_case2[0] < obj_case1[0] - 1E-3 || obj_case2[1] - 1E-3 > obj_case1[1] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {:.6}, obj_thermal_energy = {:.6}",
                    obj_case1[0],
                    obj_case1[1]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 2: obj_time = {:.6}, obj_thermal_energy = {:.6}",
                    obj_case2[0],
                    obj_case2[1]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Interesting... Cases 1 and 2"
                );
            }

            // Case 3: Time and TotalVariationTorque
            let obj_case3 = [
                CoppObjective::Time(1.0),
                CoppObjective::TotalVariationTorque(1.0, &vec![1.0; dim]),
            ];
            copp2_problem.objectives = &obj_case3;
            let start = Instant::now();
            let mut a_case3 = copp2_socp(&copp2_problem, &options_socp)?;
            let tc_copp2_case3 = start.elapsed().as_secs_f64() * 1E3;
            robot
                .constraints
                .project_to_feasible_topp2(&mut a_case3, &a_feasible, 0)?;
            let (_, obj_case3) = objective_value_copp2_opt(&robot, 0, &objectives_test, &a_case3);
            if obj_case3[1] < obj_case1[1] - 1E-3 || obj_case3[2] - 1E-3 > obj_case1[2] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {:.6}, obj_thermal_energy = {:.6}, obj_total_variation_torque = {:.6}",
                    obj_case1[0],
                    obj_case1[1],
                    obj_case1[2]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 3: obj_time = {:.6}, obj_thermal_energy = {:.6}, obj_total_variation_torque = {:.6}",
                    obj_case3[0],
                    obj_case3[1],
                    obj_case3[2]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Interesting... Cases 1 and 3"
                );
            }

            // Case 4: Time and Linear
            let obj_case4 = [
                CoppObjective::Time(1.0),
                CoppObjective::Linear(1.0, &vec![0.0; n], &vec![1.0; n - 1]),
            ];
            copp2_problem.objectives = &obj_case4;
            let start = Instant::now();
            let mut a_case4 = copp2_socp(&copp2_problem, &options_socp)?;
            let tc_copp2_case4 = start.elapsed().as_secs_f64() * 1E3;
            robot
                .constraints
                .project_to_feasible_topp2(&mut a_case4, &a_feasible, 0)?;
            let (_, obj_case4) = objective_value_copp2_opt(&robot, 0, &objectives_test, &a_case4);
            if obj_case4[1] < obj_case1[1] - 1E-3 || obj_case4[3] - 1E-3 > obj_case1[3] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {:.6}, obj_thermal_energy = {:.6}, obj_linear = {:.6}",
                    obj_case1[0],
                    obj_case1[1],
                    obj_case1[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 4: obj_time = {:.6}, obj_thermal_energy = {:.6}, obj_linear = {:.6}",
                    obj_case4[0],
                    obj_case4[1],
                    obj_case4[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Interesting... Cases 1 and 4"
                );
            }
            if obj_case4[2] < obj_case2[2] - 1E-3 || obj_case4[3] - 1E-3 > obj_case2[3] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 2: obj_time = {:.6}, obj_total_variation_torque = {:.6}, obj_linear = {:.6}",
                    obj_case2[0],
                    obj_case2[2],
                    obj_case2[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 4: obj_time = {:.6}, obj_total_variation_torque = {:.6}, obj_linear = {:.6}",
                    obj_case4[0],
                    obj_case4[2],
                    obj_case4[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Interesting... Cases 2 and 4"
                );
            }

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}:\n Case 0: tc={:.3}ms, obj={:?}\n Case 1: tc={:.3}ms, obj={:?}\n Case 2: tc={:.3}ms, obj={:?}\n Case 3: tc={:.3}ms, obj={:?}\n Case 4: tc={:.3}ms, obj={:?}",
                    i_exp + 1,
                    tc_copp2_case0,
                    obj_case0,
                    tc_copp2_case1,
                    obj_case1,
                    tc_copp2_case2,
                    obj_case2,
                    tc_copp2_case3,
                    obj_case3,
                    tc_copp2_case4,
                    obj_case4
                );
            }

            tc_sum_case0 += tc_copp2_case0;
            tc_sum_case1 += tc_copp2_case1;
            tc_sum_case2 += tc_copp2_case2;
            tc_sum_case3 += tc_copp2_case3;
            tc_sum_case4 += tc_copp2_case4;
            for i in 0..obj_case0.len() {
                obj_sum_case0[i] += obj_case0[i];
                obj_sum_case1[i] += obj_case1[i];
                obj_sum_case2[i] += obj_case2[i];
                obj_sum_case3[i] += obj_case3[i];
                obj_sum_case4[i] += obj_case4[i];
            }
        }

        for i in 0..4 {
            obj_sum_case0[i] /= n_exp as f64;
            obj_sum_case1[i] /= n_exp as f64;
            obj_sum_case2[i] /= n_exp as f64;
            obj_sum_case3[i] /= n_exp as f64;
            obj_sum_case4[i] /= n_exp as f64;
        }

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average {} experiments:\n Case 0: tc={:.3}ms, obj={:?}\n Case 1: tc={:.3}ms, obj={:?}\n Case 2: tc={:.3}ms, obj={:?}\n Case 3: tc={:.3}ms, obj={:?}\n Case 4: tc={:.3}ms, obj={:?}",
            n_exp,
            tc_sum_case0 / n_exp as f64,
            obj_sum_case0,
            tc_sum_case1 / n_exp as f64,
            obj_sum_case1,
            tc_sum_case2 / n_exp as f64,
            obj_sum_case2,
            tc_sum_case3 / n_exp as f64,
            obj_sum_case3,
            tc_sum_case4 / n_exp as f64,
            obj_sum_case4
        );

        Ok(())
    }

    fn run_one_copp2_socp_only_time_case(
        options_ra: &ReachSet2Options,
        options_socp: &ClarabelOptions,
    ) -> Result<(f64, f64, f64, f64, f64, f64), CoppError> {
        let n: usize = 1000;
        let mut robot = Robot::with_capacity(Plannar2LinkEnd::new(1.0, 1.0, 1.0, 1.0), n);
        let dim = robot.dim();

        let mut rng = rand::rng();
        let (s, derivs, omega, phi) =
            lissajous_path_for_test(dim, n, &mut rng).expect("random range is valid");
        robot.with_s(&s.as_view())?;
        robot.with_q(
            &derivs.q.as_view(),
            &derivs.dq.as_ref().unwrap().as_view(),
            &derivs.ddq.as_ref().unwrap().as_view(),
            derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
            0,
        )?;
        add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, None)?;

        let topp2_problem =
            Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
        let start = Instant::now();
        let a_ra = topp2_ra(&topp2_problem, options_ra)?;
        let tc_ra = start.elapsed().as_secs_f64() * 1E3;
        let (tf_ra, _) = s_to_t_topp2(s.as_slice(), &a_ra, 0.0);

        let obj1 = [CoppObjective::Linear(
            1.0,
            &vec![-1.0; n],
            &vec![0.0; n - 1],
        )];
        let mut copp2_problem =
            Copp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0), &obj1).build()?;
        let start = Instant::now();
        let a_lp = copp2_socp(&copp2_problem, options_socp)?;
        let tc_lp = start.elapsed().as_secs_f64() * 1E3;
        let (tf_lp, _) = s_to_t_topp2(s.as_slice(), &a_lp, 0.0);

        copp2_problem.objectives = &[CoppObjective::Time(1.0)];
        let start = Instant::now();
        let a_qp = copp2_socp(&copp2_problem, options_socp)?;
        let tc_qp = start.elapsed().as_secs_f64() * 1E3;
        let (tf_qp, _) = s_to_t_topp2(s.as_slice(), &a_qp, 0.0);

        if (tf_lp - tf_ra).abs() > 1e-3 || (tf_qp - tf_ra).abs() > 1e-3 {
            crate::verbosity_log!(
                crate::diag::Verbosity::Summary,
                "omega = {omega:?}\nphi = {phi:?}"
            );
            panic!("COPP2 time optimality failed!");
        }

        Ok((tc_ra, tc_lp, tc_qp, tf_ra, tf_lp, tf_qp))
    }

    fn run_test_copp2_socp_only_time_repeated(
        n_exp: usize,
        flag_print_step: bool,
    ) -> Result<(), CoppError> {
        let mut tc_sum_ra = 0.0;
        let mut tc_sum_lp = 0.0;
        let mut tc_sum_qp = 0.0;
        let mut tf_sum_ra = 0.0;
        let mut tf_sum_lp = 0.0;
        let mut tf_sum_qp = 0.0;

        let options_ra = ReachSet2OptionsBuilder::new()
            .lp_feas_tol(1E-9)
            .a_cmp_abs_tol(1E-9)
            .a_cmp_rel_tol(1E-9)
            .build()?;
        let options_socp = ClarabelOptionsBuilder::new()
            .allow_almost_solved(true)
            .build()?;

        for i_exp in 0..n_exp {
            let (tc_ra, tc_lp, tc_qp, tf_ra, tf_lp, tf_qp) =
                run_one_copp2_socp_only_time_case(&options_ra, &options_socp)?;

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_qp = {:.4} ms, tf_ra = {:.6}, tf_lp = {:.6}, tf_qp = {:.6}",
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
        }

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average {} experiments: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_qp = {:.4} ms, tf_ra = {:.6}, tf_lp = {:.6}, tf_qp = {:.6}",
            n_exp,
            tc_sum_ra / n_exp as f64,
            tc_sum_lp / n_exp as f64,
            tc_sum_qp / n_exp as f64,
            tf_sum_ra / n_exp as f64,
            tf_sum_lp / n_exp as f64,
            tf_sum_qp / n_exp as f64
        );

        Ok(())
    }
}
