//! 3rd-order Convex-Objective Path Parameterization (COPP3) based on second-order cone programming (SOCP).
//!
//! # Method identity
//! This module implements the **optimization backend** for COPP3 by transforming
//! third-order path-parameterization constraints/objectives into Clarabel-compatible
//! conic form and solving with SOCP.
//!
//! # Discrete variables (local notation)
//! On a path grid `s[0..=n]`:
//! - `a[k]` denotes $\dot{s}_k^2$;
//! - `b[k]` denotes $\ddot{s}_k$;
//! - decision vector starts with `x = [a[0..=n], b[0..=n], x_others]`, where
//!   `x_others` are auxiliary variables introduced by objectives (`Time`,
//!   `ThermalEnergy`, `TotalVariationTorque`, `Linear`).
//!
//! # High-level pipeline
//! 1. Validate interval/boundary/objective contract.
//! 2. Assemble standard TOPP3 conic constraints.
//! 3. Add COPP3 objective-induced variables/cones.
//! 4. Build sparse matrices `A`, `P`, vector `q`, and solve by Clarabel.
//! 5. Apply status acceptance policy (`ClarabelOptions::is_allow`) and extract
//!    `(a,b)` only when accepted.
//!
//! # API layering
//! - `copp3_socp`: strict/normal API, returns only accepted `(a,b,num_stationary)`.
//! - `copp3_socp_expert`: expert API returning `(Option<Copp3Result>, DefaultSolution<f64>)`.

use crate::copp::clarabel_backend::{ConstraintsClarabel, ObjConsClarabel};
use crate::copp::copp3::Copp3Result;
use crate::copp::copp3::formulation::{Copp3Problem, get_weight_a_copp3, get_weight_a_topp3};
use crate::copp::copp3::opt3::clarabel_constraints::{
    clarabel_standard_capacity_topp3, clarabel_standard_constraint_topp3,
};
use crate::copp::{
    ClarabelOptions, CoppObjective, clarabel_to_copp3_solution, validate_copp3_objectives,
};
use crate::diag::{
    CoppError, DebugVerboser, SilentVerboser, SummaryVerboser, TraceVerboser, Verboser, Verbosity,
    check_boundary_state_copp3_valid, check_s_interval_valid, format_duration_human,
};
use crate::robot::robot_core::{Robot, RobotBasic, RobotTorque};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{NonnegativeConeT, SecondOrderConeT};
use clarabel::solver::{DefaultSolution, DefaultSolver, IPSolver, SupportedConeT};
use core::f64;
use itertools::{Itertools, izip};
use nalgebra::{DMatrix, DVectorView};

/// Strict COPP3-SOCP API for production use.
///
/// # Purpose
/// Use this entry when caller only needs a valid profile `(a,b,num_stationary)` and treats
/// non-accepted solver statuses as hard failures.
///
/// # Contract
/// - Internally calls [`copp3_socp_expert`].
/// - Returns `Ok((a,b,num_stationary))` **iff** `options.is_allow(solution.status)` is `true`.
/// - Returns [`Err(CoppError::ClarabelSolverStatus(...))`](CoppError::ClarabelSolverStatus) when status is not accepted.
///
/// # Returns
/// Returns accepted COPP3 profile `(a, b, num_stationary)`.
///
/// # Errors
/// Returns `CoppError` on model/solver failures and non-accepted solver status.
///
/// # Notes
/// For workflows requiring low-level diagnostics (`status` and raw Clarabel solution fields),
/// prefer [`copp3_socp_expert`].
pub fn copp3_socp<'a, M: RobotTorque>(
    problem: &Copp3Problem<'a, M>,
    options: &ClarabelOptions,
) -> Result<Copp3Result, CoppError> {
    let (result, solution) = copp3_socp_expert(problem, options)?;
    result.ok_or_else(|| CoppError::ClarabelSolverStatus("copp3_socp".into(), solution.status))
}

/// Expert COPP3-SOCP API with full Clarabel solution exposure.
///
/// # Return contract
/// - `Ok((Some(result), solution))`: status accepted by `options.is_allow(solution.status)`.
/// - `Ok((None, solution))`: solve finished but status not accepted.
/// - `Err(...)`: input/model/solver-construction runtime failures.
///
/// # Returns
/// Returns tuple `(Option<Copp3Result>, DefaultSolution<f64>)` for diagnostic use.
///
/// # Errors
/// Returns [`CoppError`] only for true runtime failures.
///
/// # Contract
/// - caller must handle `None` profile for non-accepted statuses;
/// - status acceptance policy is defined by `options.is_allow`.
///
/// # Verbosity behavior
/// Logging is layered by `options.verbosity()`:
/// - [`Silent`](Verbosity::Silent): no algorithm logs;
/// - [`Summary`](Verbosity::Summary): lifecycle milestones and elapsed time;
/// - [`Debug`](Verbosity::Debug): assembly-level counters and stage summaries;
/// - [`Trace`](Verbosity::Trace): fine-grained stage deltas and solver snapshot diagnostics.
pub fn copp3_socp_expert<'a, M: RobotTorque>(
    problem: &Copp3Problem<'a, M>,
    options: &ClarabelOptions,
) -> Result<(Option<Copp3Result>, DefaultSolution<f64>), CoppError> {
    match options.verbosity() {
        Verbosity::Silent => copp3_socp_core(problem, (options, SilentVerboser)),
        Verbosity::Summary => copp3_socp_core(problem, (options, SummaryVerboser::new())),
        Verbosity::Debug => copp3_socp_core(problem, (options, DebugVerboser::new())),
        Verbosity::Trace => copp3_socp_core(problem, (options, TraceVerboser::new())),
    }
}

/// Core implementation for COPP3-SOCP expert flow.
///
/// # Internal contract
/// `options_verboser` packs:
/// - `options`: acceptance policy and Clarabel numerical settings;
/// - `verboser`: concrete logger implementation chosen by external verbosity dispatch.
///
/// # Invariants
/// - decision-variable layout always starts with contiguous `a[0..=n]` and `b[0..=n]`;
/// - `q_object.len()` is treated as final `n_var` before solver build;
/// - extracted `(a,b)` is produced only through `clarabel_to_copp3_solution` when status is accepted.
fn copp3_socp_core<'a, M: RobotTorque>(
    problem: &Copp3Problem<'a, M>,
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
            "copp3_socp: options snapshot -> allow(almost={}, max_iter={}, max_time={}, callback_term={}, insufficient_progress={}), tol_gap_rel={}, tol_feas={}, max_iter={}, verbose={}",
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
            "\ncopp3_socp started: {} <= idx_s <= {}, objectives = {}, s_len = {}.",
            idx_s_start,
            idx_s_final,
            problem.objectives.len(),
            problem.a_linearization.len()
        );
    }
    check_s_interval_valid("copp3_socp", idx_s_start, idx_s_final)?;
    validate_copp3_objectives(
        "copp3_socp",
        problem.objectives,
        problem.robot.dim(),
        problem.a_linearization.len(),
    )?;
    // Let x = [a[0,1,...,n],
    //          b[0,1,...,n],
    //          xi[0,1,...,len_xi-1], (if xi exists.)
    //          x_others] \in R^{2*(n+1), x_others}.
    // Step 1. Deal with constraints
    // Step 1.1 Compute the number of constraints
    let (cap_val_std, cap_b_std, cap_cone_std) =
        clarabel_standard_capacity_topp3(&problem.robot.constraints, (idx_s_start, idx_s_final));
    let (cap_val_obj, cap_b_obj, cap_cone_obj, n_vars) =
        clarabel_objective_capacity_copp3(n, problem.objectives, problem.robot);
    if verboser.is_enabled(Verbosity::Debug) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: capacity estimate std(val={cap_val_std}, b={cap_b_std}, cone={cap_cone_std}), obj(val={cap_val_obj}, b={cap_b_obj}, cone={cap_cone_obj}), n_vars={n_vars}."
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
            "copp3_socp: allocated capacities row/col/val/b/cones <= {}/{}/{}/{}/{}",
            cap_val_std + cap_val_obj,
            cap_val_std + cap_val_obj,
            cap_val_std + cap_val_obj,
            cap_b_std + cap_b_obj,
            cap_cone_std + cap_cone_obj
        );
    }
    // Step 1.2 set constraints of the standard topp3-lp problem
    let s = problem
        .robot
        .constraints
        .s_vec(idx_s_start, idx_s_final + 1)?;
    let row_before_std = row.len();
    let col_before_std = col.len();
    let val_before_std = val.len();
    let b_before_std = b.len();
    let cones_before_std = cones.len();
    clarabel_standard_constraint_topp3(
        &problem.as_topp3_problem(),
        &s,
        (&mut row, &mut col, &mut val, &mut b, &mut cones),
        num_stationary,
        &verboser,
    )?;
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: standard-constraints delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}",
            row.len() - row_before_std,
            col.len() - col_before_std,
            val.len() - val_before_std,
            b.len() - b_before_std,
            cones.len() - cones_before_std
        );
    }
    // Step 2. set objective
    // Step 2.1. determine whether xi=sqrt(a) is needed.
    let row_before_sqrt = row.len();
    let col_before_sqrt = col.len();
    let val_before_sqrt = val.len();
    let b_before_sqrt = b.len();
    let cones_before_sqrt = cones.len();
    let n_var_old = clarabel_sqrt_a_copp3(
        n,
        problem.objectives,
        (&mut row, &mut col, &mut val, &mut b, &mut cones),
        num_stationary,
    );
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: sqrt-a stage delta row/col/val/b/cones = +{}/+{}/+{}/+{}/+{}, n_var_old={}",
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
    clarabel_objective_copp3(
        problem,
        num_stationary,
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
            "copp3_socp: objective stage delta row/col/val/b/cones/q = +{}/+{}/+{}/+{}/+{}/+{}, q_range=[{}, {}]",
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
            "copp3_socp: after objective assembly row={}, col={}, val={}, b={}, cones={}, q={}",
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
    let row_len = row.len();
    let col_len = col.len();
    let val_len = val.len();
    let b_len = b.len();
    let cones_len = cones.len();
    let a_csc = CscMatrix::new_from_triplets(b.len(), n_var, row, col, val);
    let p_object = CscMatrix::<f64>::zeros((n_var, n_var));
    if verboser.is_enabled(Verbosity::Trace) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: matrix built with m={}, n={}, A.nnz={}, P.nnz={}",
            b.len(),
            n_var,
            a_csc.nnz(),
            p_object.nnz()
        );
    }
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: ready to solve with row/col/val/b/cones = {row_len}/{col_len}/{val_len}/{b_len}/{cones_len} and n_var = {n_var}.",
        );
    }
    // Step 3. solve the SOCP problem
    let settings = options.clarabel_settings().clone();
    let mut solver = DefaultSolver::<f64>::new(&p_object, &q_object, &a_csc, &b, &cones, settings)
        .map_err(|e| CoppError::ClarabelSolverError("copp3_socp".into(), e))?;
    solver.solve();
    let solution = solver.solution;
    if verboser.is_enabled(Verbosity::Summary) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: solve done, status = {:?}, elapsed = {}.",
            solution.status,
            format_duration_human(verboser.elapsed())
        );
    }
    if verboser.is_enabled(Verbosity::Trace) {
        let show = solution.x.len().min(3);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "copp3_socp: solution x_len={}, head={:?}",
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
            "copp3_socp: allow(status)={}, extracted_profile={}",
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

/// Determine the length of xi[k] = sqrt(a[k + k_skip]) in the decision variable x.
#[inline(always)]
fn length_xi(n: usize, num_stationary: (usize, usize)) -> usize {
    n + 1 - num_stationary.0.max(1) - num_stationary.1.max(1)
}

/// Return k_skip, where xi[k] = sqrt(a[k + k_skip])
#[inline(always)]
fn skip_a_for_xi(num_stationary_start: usize) -> usize {
    num_stationary_start.max(1)
}

/// Add the constraints for sqrt(a) >= xi in COPP3 optimization.  
/// x = [a[0,...,n], b[0,...,n], xi[0,...,len_xi-1], ...] \in R^{2*(n+1)+len_xi+...}.
/// sqrt(a[k]) >= xi[k] >= 0  
/// num_val <= 4*n, num_b <= 4*n, num_cones <= n  
/// Return the len of the new x: n+1 or 2*(n+1)
fn clarabel_sqrt_a_copp3(
    n: usize,
    objective: &[CoppObjective],
    constraints: ConstraintsClarabel,
    num_stationary: (usize, usize),
) -> usize {
    let (row, col, val, b, cones) = constraints;
    for obj in objective {
        match obj {
            CoppObjective::Time(_) | CoppObjective::ThermalEnergy(_, _) => {
                let n_skip = skip_a_for_xi(num_stationary.0);
                let len_xi = length_xi(n, num_stationary); // < n
                // xi >= 0
                // A*x-b = -s = -1*xi[k] <= 0
                row.extend(b.len()..b.len() + len_xi);
                col.extend((2 * (n + 1))..(2 * (n + 1) + len_xi));
                val.resize(val.len() + len_xi, -1.0);
                b.resize(b.len() + len_xi, 0.0);
                cones.push(NonnegativeConeT(len_xi));
                // sqrt(a) >= xi
                // xi^2 <= a
                // xi^2 + (a - 0.25)^2 <= (a + 0.25)^2
                // [a[k]+0.25, a[k]-0.25, xi[k]] \in SOC
                // -A*x+b = s = [x[k+n_skip]+0.25, x[k+n_skip]-0.25, x[2*(n+1)+k]] \in SOC
                row.extend(b.len()..b.len() + 3 * len_xi);
                val.resize(val.len() + 3 * len_xi, -1.0);
                cones.resize(cones.len() + len_xi, SecondOrderConeT(3));
                for k in 0..len_xi {
                    // row.extend(b.len()..b.len() + 3);
                    col.extend([k + n_skip, k + n_skip, 2 * (n + 1) + k]);
                    // val.resize(val.len() + 3, -1.0);
                    b.extend([0.25, -0.25, 0.0]);
                    // cones.push(SecondOrderConeT(3));
                }
                return 2 * (n + 1) + len_xi;
            }
            _ => {}
        }
    }
    2 * (n + 1)
}

/// Determine the number of clarabel's capacity for the objective in COPP3.
fn clarabel_objective_capacity_copp3<M: RobotBasic>(
    n: usize,
    objective: &[CoppObjective],
    robot: &Robot<M>,
) -> (usize, usize, usize, usize) {
    // Step 1. sqrt(a[k]) >= xi[k] >= 0
    // num_val <= 4*n, num_b <= 4*n, num_cones <= n, n_var <= n
    let (mut capacity_val, mut capacity_b, mut capacity_cones, mut n_vars) =
        if objective.iter().any(|obj| {
            matches!(
                obj,
                CoppObjective::Time(_) | CoppObjective::ThermalEnergy(_, _)
            )
        }) {
            (4 * n, 4 * n, n, 2 * (n + 1))
        } else {
            (0, 0, 0, 2 * (n + 1))
        };
    // Step 2. objective function
    let dim = robot.dim();
    for obj in objective {
        match obj {
            CoppObjective::Time(_) => {
                // num_val <= 5*n, num_b <= 4*n, num_cones <= n, n_var <= n
                capacity_val += 5 * n;
                capacity_b += 4 * n;
                capacity_cones += n;
                n_vars += n;
            }
            CoppObjective::ThermalEnergy(_, _) => {
                // num_val <= (4+2*dim)*(n+1), num_b <= (2+dim)*(n+1), num_cones <= n+1, n_var <= n+1
                capacity_val += (4 + 2 * dim) * (n + 1);
                capacity_b += (dim + 2) * (n + 1);
                capacity_cones += n + 1;
                n_vars += n + 1;
            }
            CoppObjective::TotalVariationTorque(_, _) => {
                // num_val <= 10*n*dim, num_b <= 2*n*dim, num_cones <= 1, n_var <= n*dim
                capacity_val += 10 * dim * n;
                capacity_b += 2 * dim * n;
                capacity_cones += 1;
                n_vars += dim * n;
            }
            _ => {}
        }
    }
    (capacity_val, capacity_b, capacity_cones, n_vars)
}

fn clarabel_objective_copp3<M: RobotTorque>(
    problem: &Copp3Problem<M>,
    num_stationary: (usize, usize),
    objective_constraints: ObjConsClarabel,
) -> Result<(), CoppError> {
    let (row, col, val, b, cones, q_object) = objective_constraints;
    let n = problem.a_linearization.len() - 1;
    let s = problem
        .robot
        .constraints
        .s_vec(problem.idx_s_start, problem.idx_s_start + n + 1)?;
    let weight_a_time = if problem
        .objectives
        .iter()
        .any(|obj| matches!(obj, CoppObjective::Time(_)))
    {
        get_weight_a_topp3(&s, num_stationary)
    } else {
        vec![]
    };
    let weight_a_torque = if problem
        .objectives
        .iter()
        .any(|obj| matches!(obj, CoppObjective::ThermalEnergy(_, _)))
    {
        get_weight_a_copp3(&s, num_stationary)
    } else {
        vec![]
    };
    let coeffs_torque = if problem.objectives.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::ThermalEnergy(_, _) | CoppObjective::TotalVariationTorque(_, _)
        )
    }) {
        // shape: (dim, n) since there are n+1 a and n b.
        problem.robot.torque_coeff(problem.idx_s_start, n + 1)
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
                if !clarabel_objective_time_copp3(
                    &s,
                    *weight,
                    &weight_a_time,
                    num_stationary,
                    (row, col, val, b, cones, q_object),
                ) {
                    return Err(CoppError::InvalidInput(
                        "clarabel_objective_copp3".into(),
                        "Invalid Time objective".into(),
                    ));
                }
            }
            CoppObjective::ThermalEnergy(weight, normalize) => {
                if !clarabel_objective_thermal_energy_copp3(
                    &weight_a_torque,
                    *weight,
                    normalize,
                    &coeffs_torque,
                    num_stationary,
                    (row, col, val, b, cones, q_object),
                ) {
                    return Err(CoppError::InvalidInput(
                        "clarabel_objective_copp3".into(),
                        "Invalid ThermalEnergy objective".into(),
                    ));
                }
            }
            CoppObjective::TotalVariationTorque(weight, normalize) => {
                if !clarabel_objective_tv_torque_copp3(
                    *weight,
                    normalize,
                    &coeffs_torque,
                    num_stationary,
                    (row, col, val, b, cones, q_object),
                ) {
                    return Err(CoppError::InvalidInput(
                        "clarabel_objective_copp3".into(),
                        "Invalid TotalVariationTorque objective".into(),
                    ));
                }
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                if !clarabel_objective_linear_copp3(
                    &s,
                    *weight,
                    alpha,
                    beta,
                    q_object,
                    num_stationary,
                ) {
                    return Err(CoppError::InvalidInput(
                        "clarabel_objective_copp3".into(),
                        "Invalid Linear objective".into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Add the constraints and objective for Time in COPP3 optimization.  
/// num_val <= 5*n, num_b <= 4*n, num_cones <= n, n_var <= n
fn clarabel_objective_time_copp3(
    s: &[f64],
    weight: f64,
    weight_a: &[f64],
    num_stationary: (usize, usize),
    objective_constraints: ObjConsClarabel,
) -> bool {
    if weight < 0.0 {
        return false;
    }

    let (row, col, val, b, cones, q_object) = objective_constraints;
    let n = s.len() - 1;
    let len_xi = length_xi(n, num_stationary);
    let k_skip = skip_a_for_xi(num_stationary.0);
    // Add constraints for xi and eta, where xi[k]=sqrt(a[k+k_skip]), eta[k]=1/xi[k]
    // eta[k] >= 0,
    // norm2([2, xi[k] - eta[k]]) <= xi[k] + eta[k]
    let id_xi_start = 2 * (n + 1);
    let id_eta_start = q_object.len();
    // Step 1. eta[k] >= 0
    // A*x-b = -s = -1*eta[k] = -1*x[id_eta_start + k] <= 0
    row.extend(b.len()..(b.len() + len_xi));
    col.extend(id_eta_start..(id_eta_start + len_xi));
    val.resize(val.len() + len_xi, -1.0);
    b.resize(b.len() + len_xi, 0.0);
    cones.push(NonnegativeConeT(len_xi));
    // Step 2. norm2([2, xi[k] - eta[k]]) <= xi[k] + eta[k]
    // -A*x+b = s = [xi[k] + eta[k], xi[k] - eta[k], 2] \in SOC
    for k in 0..len_xi {
        // -A*x+b = s = [x[id_xi_start + k] + x[id_eta_start + k], x[id_xi_start + k] - x[id_eta_start + k], 2] \in SOC
        col.extend([
            id_xi_start + k,
            id_eta_start + k,
            id_xi_start + k,
            id_eta_start + k,
        ]);
        row.extend([b.len(), b.len(), b.len() + 1, b.len() + 1]);
        val.extend([-1.0, -1.0, -1.0, 1.0]);
        b.extend([0.0, 0.0, 2.0]);
    }
    cones.resize(cones.len() + len_xi, SecondOrderConeT(3));
    // Minimize sum[k in 0..len_xi] {weight_a[k+k_skip] / sqrt(a[k+k_skip])}
    // Minimize sum[k in 0..len_xi] {weight_a[k+k_skip] * eta[k]}
    q_object.extend(
        weight_a
            .iter()
            .skip(k_skip)
            .take(len_xi)
            .map(|w| weight * w),
    );

    true
}

/// Add the constraints and objective for ThermalEnergy in COPP3 optimization.  
/// num_val <= (4+2*dim)*(n+1), num_b <= (2+dim)*(n+1), num_cones <= n+1, n_var <= n+1
fn clarabel_objective_thermal_energy_copp3(
    weight_a: &[f64],
    weight: f64,
    normalize: &[f64],
    coeffs_torque: &(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>),
    num_stationary: (usize, usize),
    objective_constraints: ObjConsClarabel,
) -> bool {
    if weight < 0.0 {
        return false;
    }
    let (row, col, val, b, cones, q_object) = objective_constraints;
    // minimize: weight * \int_{s[0]}^{s[n]} {\sum[i] {(tau[i][s] * normalize[i])^2 / sqrt(a[s])} ds}
    let mut coeff_a = coeffs_torque.0.clone();
    let mut coeff_b = coeffs_torque.1.clone();
    let mut coeff_g = coeffs_torque.2.clone();
    let dim = coeff_a.nrows();
    if normalize.len() != dim {
        return false;
    }
    let n = coeff_a.ncols() - 1;
    // tau[i][k] = coeff_a[i][k] * a[k] + coeff_b[i][k] * b[k] + coeff_g[i][k]
    let normalize = DVectorView::from_slice(normalize, dim);
    for mut col in coeff_a.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_b.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_g.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    // tau[i][k] * normalize[i] = coeff_a[i][k] * a[k] + coeff_b[i][k] * b[k] + coeff_g[i][k]
    let k_skip = skip_a_for_xi(num_stationary.0);
    let len_xi = length_xi(n, num_stationary);

    if num_stationary.0 > 0 {
        // minimize: weight * \int_{s[0]}^{s[num_stationary.0]} {\sum[i] {tau_normal[i][s]^2 / sqrt(a[s])} ds}
        // \approx weight * \sum[i] { tau_average[i]^2 * \int_{s[0]}^{s[num_stationary.0]} {1 / sqrt(a[s])} ds} }
        // \int_{s[0]}^{s[num_stationary.0]} {1 / sqrt(a[s])} ds} = weight[0] / xi[0]
        // minmize: weight * weight_a[0] * \sum[i] { tau_average[i]^2 / xi[0] }
        // let tau_average[i] = (tau_normal[i][0] + tau_normal[i][num_stationary.0]) / 2
        let col_a = coeff_a.column(num_stationary.0);
        let col_b = coeff_b.column(num_stationary.0);
        let col_g = coeff_g.column(num_stationary.0);
        let col_g_0 = coeff_g.column(0);
        // tau[0] = col_g_0
        // tau[num_stationary.0] = col_a * a[num_stationary.0] + col_b * b[num_stationary.0] + col_g
        // let \sum[i] { tau_average[i]^2 } <= t * xi[0]
        // \sum[i](col_a[i] * a[num_stationary.0] + col_b[i] * b[num_stationary.0] + col_g[i] + col_g_0[i])^2 <= 4 * t
        // -A*x+b = s = [t + xi[0], t - xi[0], -(col_a[i] * a[num_stationary.0] + col_b[i] * b[num_stationary.0] + col_g[i] + col_g_0[i])] \in SOC
        let id_t = q_object.len();
        // -A*x+b = [t + xi[0], t - xi[0]]
        row.extend(b.len()..(b.len() + 2));
        col.resize(col.len() + 2, id_t);
        row.extend(b.len()..(b.len() + 2));
        col.resize(col.len() + 2, 2 * (n + 1));
        val.resize(val.len() + 3, -1.0);
        val.push(1.0);
        b.resize(b.len() + 2, 0.0);
        // -A*x+b = [-(col_a[i] * a[num_stationary.0] + col_b[i] * b[num_stationary.0] + col_g[i] + col_g_0[i])] for i in 0..dim
        row.extend((b.len())..(b.len() + dim));
        col.resize(col.len() + dim, num_stationary.0);
        val.extend(col_a.iter().take(dim));
        row.extend((b.len())..(b.len() + dim));
        col.resize(col.len() + dim, n + 1 + num_stationary.0);
        val.extend(col_b.iter().take(dim));
        b.extend(
            col_g
                .iter()
                .zip(col_g_0.iter())
                .take(dim)
                .map(|(&g, &g_0)| -(g + g_0)),
        );
        cones.push(SecondOrderConeT(dim + 2));
        q_object.push(weight * weight_a[0]);
    }
    if num_stationary.1 > 0 {
        // minimize: weight * \int_{s[n-num_stationary.1]}^{s[n]} {\sum[i] {tau_normal[i][s]^2 / sqrt(a[s])} ds}
        // \approx weight * \sum[i] { tau_average[i]^2 * \int_{s[n-num_stationary.1]}^{s[n]} {1 / sqrt(a[s])} ds} }
        // \int_{s[n-num_stationary.1]}^{s[n]} {1 / sqrt(a[s])} ds} = weight[n-num_stationary.1] / xi[len_xi-1]
        // minmize: weight * weight[n] * \sum[i] { tau_average[i]^2 }
        // let tau_average[i] = (tau_normal[i][n] + tau_normal[i][n-num_stationary.1]) / 2
        let col_a = coeff_a.column(n - num_stationary.1);
        let col_b = coeff_b.column(n - num_stationary.1);
        let col_g = coeff_g.column(n - num_stationary.1);
        let col_g_f = coeff_g.column(n);
        // tau[f] = col_g_n
        // tau[n-num_stationary.1] = col_a * a[n-num_stationary.1] + col_b * b[n-num_stationary.1] + col_g
        // let \sum[i] { tau_average[i]^2 } <= t * xi[len_xi-1]
        // \sum[i](col_a[i] * a[n-num_stationary.1] + col_b[i] * b[n-num_stationary.1] + col_g[i] + col_g_f[i])^2 <= 4 * t * xi[len_xi-1]
        // -A*x+b = s = [t + xi[len_xi-1], t - xi[len_xi-1], -(col_a[i] * a[n-num_stationary.1] + col_b[i] * b[n-num_stationary.1] + col_g[i] + col_g_f[i])] \in SOC
        let id_t = q_object.len();
        // -A*x+b = [t+xi[len_xi-1], t-xi[len_xi-1]]
        row.extend(b.len()..(b.len() + 2));
        col.resize(col.len() + 2, id_t);
        row.extend(b.len()..(b.len() + 2));
        col.resize(col.len() + 2, 2 * n + len_xi + 1);
        val.resize(val.len() + 3, -1.0);
        val.push(1.0);
        b.resize(b.len() + 2, 0.0);
        // -A*x+b = [-(col_a[i] * a[n-num_stationary.1] + col_b[i] * b[n-num_stationary.1] + col_g[i] + col_g_f[i])] for i in 0..dim
        row.extend((b.len())..(b.len() + dim));
        col.resize(col.len() + dim, n - num_stationary.1);
        val.extend(col_a.iter().take(dim));
        row.extend((b.len())..(b.len() + dim));
        col.resize(col.len() + dim, 2 * n + 1 - num_stationary.1);
        val.extend(col_b.iter().take(dim));
        b.extend(
            col_g
                .iter()
                .zip(col_g_f.iter())
                .take(dim)
                .map(|(&g, &g_f)| -(g + g_f)),
        );
        cones.push(SecondOrderConeT(dim + 2));
        q_object.push(weight * weight_a[n]);
    }

    // minimize: weight * \sum[k] { \int_{s[k]}^{s[k+1]} {\sum[i] {tau_normal[i][s]^2 / sqrt(a[s])} ds} }
    // Decouple the integral
    // minimize: weight * \sum[k] { \sum[i] {tau_normal[i][k]^2} / sqrt(a[k]) * 0.5 * (s[k+1]-s[k-1]) }
    let id_t_start = q_object.len();
    for (k, (col_a, col_b, col_g)) in izip!(
        coeff_a.column_iter(),
        coeff_b.column_iter(),
        coeff_g.column_iter()
    )
    .skip(k_skip)
    .take(len_xi)
    .enumerate()
    {
        // \sum[i] {(col_a[i] * a[k+k_skip] + col_b[i] * b[k+k_skip] + col_g[i])^2} / xi[k] <= 4 * t[k]
        // \sum[i] {(col_a[i] * a[k+k_skip] + col_b[i] * b[k+k_skip] + col_g[i])^2} <= 4 * t[k] * xi[k]
        // -A*x+b = s = [t[k] + xi[k], t[k] - xi[k], -(col_a[i] * a[k+k_skip] + col_b[i] * b[k+k_skip] + col_g[i])] \in SOC
        // -A*x+b = s = [t[k] + xi[k], t[k] - xi[k]]
        row.extend(b.len()..(b.len() + 2));
        col.resize(col.len() + 2, id_t_start + k);
        row.extend(b.len()..(b.len() + 2));
        col.resize(col.len() + 2, 2 * (n + 1) + k);
        val.resize(val.len() + 3, -1.0);
        val.push(1.0);
        b.resize(b.len() + 2, 0.0);
        // -A*x+b = s = [-(col_a[i] * a[k+k_skip] + col_b[i] * b[k+k_skip] + col_g[i])]
        row.extend((b.len())..(b.len() + dim));
        col.resize(col.len() + dim, k + k_skip);
        val.extend(col_a.iter().take(dim));
        row.extend((b.len())..(b.len() + dim));
        col.resize(col.len() + dim, n + 1 + k + k_skip);
        val.extend(col_b.iter().take(dim));
        b.extend(col_g.iter().take(dim).map(|&g| -g));
    }
    cones.resize(cones.len() + len_xi, SecondOrderConeT(dim + 2));
    // minimize: 2 * weight * \sum[k] { t[k] * (s[k+1]-s[k-1]) }
    // if num_stationary == (0,0), then \sum[k in 1..n] { t[k] * (s[k+1]-s[k-1]) }
    // if num_stationary == (n1>0,n2>0), then \sum[k in (n1+1)..(n-n2-1)] { t[k] * (s[k+1]-s[k-1]) } + t[n1] * (s[n1+1]-s[n1]) + t[n-n2] * (s[n-n2]-s[n-n2-1])
    let weight_four = 4.0 * weight;
    q_object.extend(
        weight_a
            .iter()
            .skip(k_skip)
            .take(len_xi)
            .map(|w| weight_four * w),
    );
    true
}

/// Add the constraints and objective for TotalVariationTorque in COPP3 optimization.  
/// num_val <= 10*n*dim, num_b <= 2*n*dim, num_cones <= 1, n_var <= n*dim
fn clarabel_objective_tv_torque_copp3(
    weight: f64,
    normalize: &[f64],
    coeffs_torque: &(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>),
    num_stationary: (usize, usize),
    objective_constraints: ObjConsClarabel,
) -> bool {
    if weight < 0.0 {
        return false;
    }
    let (row, col, val, b, cones, q_object) = objective_constraints;
    // minimize: weight * \sum |tau[i][k+1]-tau[i][k]| * normalize[i]
    // Let: |tau[i][k+1]-tau[i][k]| * normalize[i] <= t[i][k]
    let mut coeff_a = coeffs_torque.0.clone();
    let mut coeff_b = coeffs_torque.1.clone();
    let mut coeff_g = coeffs_torque.2.clone();
    let dim = coeff_a.nrows();
    if normalize.len() != dim {
        return false;
    }
    let n = coeff_a.ncols() - 1;
    // tau[i][k] = coeff_a[i][k] * a[k] + coeff_b[i][k] * b[k] + coeff_g[i][k]
    let normalize = DVectorView::from_slice(normalize, dim);
    for mut col in coeff_a.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_b.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    for mut col in coeff_g.column_iter_mut() {
        col.component_mul_assign(&normalize);
    }
    // tau[i][k] * normalize[i] = coeff_a[i][k] * a[k] + coeff_b[i][k] * b[k] + coeff_g[i][k]
    // (tau[i][k+1] - tau[i][k]) * normalize[i] = coeff_a[i][k+1] * a[k+1] + coeff_b[i][k+1] * b[k+1] + coeff_g[i][k+1] - coeff_a[i][k] * a[k] - coeff_b[i][k] * b[k] - coeff_g[i][k]

    let n_b_old = b.len();
    // A*x-b = -s = -coeff_a[i][k] * x[k] + coeff_a[i][k+1] * x[k+1] - coeff_b[i][k] * x[n+k+1] + coeff_b[i][k+1] * x[n+k+2] + coeff_g[i][k+1] - coeff_g[i][k] - t[i][k] <= 0
    // A*x-b = -s = -(-coeff_a[i][k] * x[k] + coeff_a[i][k+1] * x[k+1] - coeff_b[i][k] * x[n+k+1] + coeff_b[i][k+1] * x[n+k+2] + coeff_g[i][k+1] - coeff_g[i][k] - t[i][k]) <= 0
    if num_stationary.0 > 0 {
        // Consider (tau[i][num_stationary.0] - tau[i][0]) * normalize[i] = coeff_a[i][num_stationary.0] * a[num_stationary.0] + coeff_b[i][num_stationary.0] * b[num_stationary.0] + coeff_g[i][num_stationary.0] - coeff_g[i][0]
        let col_a = coeff_a.column(num_stationary.0);
        let col_b = coeff_b.column(num_stationary.0);
        let col_g = coeff_g.column(num_stationary.0);
        let col_g_0 = coeff_g.column(0);
        let n_var_old = q_object.len();
        for (i, (&v_a, &v_b, &v_g, &v_g_0)) in
            izip!(col_a.iter(), col_b.iter(), col_g.iter(), col_g_0.iter())
                .take(dim)
                .enumerate()
        {
            // dtau_normal = v_a * a[num_stationary.0] + v_b * b[num_stationary.0] + v_g - v_g_0
            // A*x-b = -s = v_a * a[num_stationary.0] + v_b * b[num_stationary.0] + v_g - v_g_0 - t[i] <= 0
            row.resize(row.len() + 3, b.len());
            col.extend([num_stationary.0, n + num_stationary.0 + 1, n_var_old + i]);
            val.extend([v_a, v_b, -1.0]);
            b.push(v_g - v_g_0);
            // A*x-b = -s = -(v_a * a[num_stationary.0] + v_b * b[num_stationary.0] + v_g - v_g_0) - t[i] <= 0
            row.resize(row.len() + 3, b.len());
            col.extend([num_stationary.0, n + num_stationary.0 + 1, n_var_old + i]);
            val.extend([-v_a, -v_b, -1.0]);
            b.push(v_g_0 - v_g);
        }
        q_object.resize(q_object.len() + dim, weight);
    }

    if num_stationary.1 > 0 {
        // Consider (tau[i][n-num_stationary.1] - tau[i][n]) * normalize[i] = coeff_a[i][n-num_stationary.1] * a[n-num_stationary.1] + coeff_b[i][n-num_stationary.1] * b[n-num_stationary.1] + coeff_g[i][n-num_stationary.1] - coeff_g[i][n]
        let col_a = coeff_a.column(n - num_stationary.1);
        let col_b = coeff_b.column(n - num_stationary.1);
        let col_g = coeff_g.column(n - num_stationary.1);
        let col_g_f = coeff_g.column(n);
        let n_var_old = q_object.len();
        for (i, (&v_a, &v_b, &v_g, &v_g_f)) in
            izip!(col_a.iter(), col_b.iter(), col_g.iter(), col_g_f.iter())
                .take(dim)
                .enumerate()
        {
            // dtau_normal = v_a * a[n-num_stationary.1] + v_b * b[n-num_stationary.1] + v_g - v_g_0
            // A*x-b = -s = v_a * a[n-num_stationary.1] + v_b * b[n-num_stationary.1] + v_g - v_g_0 - t[i] <= 0
            row.resize(row.len() + 3, b.len());
            col.extend([
                n - num_stationary.1,
                n + n - num_stationary.1 + 1,
                n_var_old + i,
            ]);
            val.extend([v_a, v_b, -1.0]);
            b.push(v_g - v_g_f);
            // A*x-b = -s = -(v_a * a[n-num_stationary.1] + v_b * b[n-num_stationary.1] + v_g - v_g_f) - t[i] <= 0
            row.resize(row.len() + 3, b.len());
            col.extend([
                n - num_stationary.1,
                n + n - num_stationary.1 + 1,
                n_var_old + i,
            ]);
            val.extend([-v_a, -v_b, -1.0]);
            b.push(v_g_f - v_g);
        }
        q_object.resize(q_object.len() + dim, weight);
    }

    // Consider (tau[i][k+1] - tau[i][k]) * normalize[i] for k in num_stationary.0..(n-num_stationary.1)
    for (k, ((col_a_curr, col_b_curr, col_g_curr), (col_a_next, col_b_next, col_g_next))) in izip!(
        coeff_a.column_iter(),
        coeff_b.column_iter(),
        coeff_g.column_iter()
    )
    .tuple_windows()
    .enumerate()
    .skip(num_stationary.0)
    .take(n - num_stationary.0 - num_stationary.1)
    {
        let n_var_old = q_object.len();
        for (i, (&v_a_curr, &v_b_curr, &v_g_curr, &v_a_next, &v_b_next, &v_g_next)) in izip!(
            col_a_curr.iter(),
            col_b_curr.iter(),
            col_g_curr.iter(),
            col_a_next.iter(),
            col_b_next.iter(),
            col_g_next.iter()
        )
        .enumerate()
        {
            // dtau_normal[i] = -v_a_curr * a[k] + v_a_next * a[k+1] - v_b_curr * b[k] + v_b_next * b[k+1] + v_g_next- v_g_curr
            // A*x-b = -s = -v_a_curr * a[k] + v_a_next * a[k+1] - v_b_curr * b[k] + v_b_next * b[k+1] + v_g_next - v_g_curr - t[i][k] <= 0
            row.resize(row.len() + 5, b.len());
            col.extend([k, k + 1, n + k + 1, n + k + 2, n_var_old + i]);
            val.extend([-v_a_curr, v_a_next, -v_b_curr, v_b_next, -1.0]);
            b.push(v_g_next - v_g_curr);
            // A*x-b = -s = -(-v_a_curr * a[k] + v_a_next * a[k+1] - v_b_curr * b[k] + v_b_next * b[k+1] + v_g_next - v_g_curr) - t[i][k] <= 0
            row.resize(row.len() + 5, b.len());
            col.extend([k, k + 1, n + k + 1, n + k + 2, n_var_old + i]);
            val.extend([v_a_curr, -v_a_next, v_b_curr, -v_b_next, -1.0]);
            b.push(v_g_curr - v_g_next);
        }
        q_object.resize(q_object.len() + dim, weight);
    }
    cones.push(NonnegativeConeT(b.len() - n_b_old));

    true
}

/// Add the constraints and objective for Linear in COPP3 optimization.
fn clarabel_objective_linear_copp3(
    s: &[f64],
    weight: f64,
    alpha: &[f64],
    beta: &[f64],
    q_object: &mut [f64],
    num_stationary: (usize, usize),
) -> bool {
    if alpha.len() != s.len() || beta.len() != s.len() {
        return false;
    }
    let n = s.len() - 1;
    // objective: minimize weight * \sum (alpha[k]*a[k] + beta[k]*b[k])
    if num_stationary.0 > 1 {
        let &s_start = s.first().unwrap();
        let ds_start = s[num_stationary.0] - s_start;
        let q_n1 = &mut q_object[num_stationary.0];
        for (&s_k, &alpha_k, &beta_k) in izip!(s.iter(), alpha.iter(), beta.iter())
            .skip(1)
            .take(num_stationary.0 - 1)
        {
            let dsk_start = s_k - s_start;
            let gamma = dsk_start / ds_start;
            // a_k = a[num_stationary.0] * gamma;
            // b_k = a[num_stationary.0] * gamma / (1.5 * dsk_start);
            *q_n1 += weight * gamma * gamma.cbrt() * (alpha_k + beta_k / (1.5 * dsk_start));
        }
    }
    if num_stationary.1 > 1 {
        let &s_final = s.last().unwrap();
        let ds_final = s[n - num_stationary.1] - s_final;
        let q_n2 = &mut q_object[n - num_stationary.1];
        for (&s_k, &alpha_k, &beta_k) in izip!(s.iter(), alpha.iter(), beta.iter())
            .skip(1)
            .take(num_stationary.1 - 1)
        {
            let dsk_final = s_k - s_final;
            let gamma = dsk_final / ds_final;
            // a_k = a[n - num_stationary.1] * gamma;
            // b_k = a[n - num_stationary.1] * gamma / (1.5 * dsk_start);
            *q_n2 += weight * gamma * gamma.cbrt() * (alpha_k + beta_k / (1.5 * dsk_final));
        }
    }
    for (q_k, &alpha_k) in q_object
        .iter_mut()
        .zip(alpha.iter())
        .take(n + 1 - num_stationary.1)
        .skip(num_stationary.0)
    {
        *q_k += weight * alpha_k;
    }
    for (q_k, &beta_k) in q_object
        .iter_mut()
        .skip(n + 1)
        .zip(beta.iter())
        .take(n + 1 - num_stationary.1)
        .skip(num_stationary.0)
    {
        *q_k += weight * beta_k;
    }

    true
}

/// Compute the time value in COPP3 optimization.  
/// Input: a_sqrt_down = 1 / sqrt(a)  
#[cfg(test)]
#[inline(always)]
fn objective_value_time_copp3(
    a_sqrt_down: &[f64],
    weight_a: &[f64],
    num_stationary: (usize, usize),
) -> f64 {
    let n = a_sqrt_down.len() - 1;
    let k_skip = skip_a_for_xi(num_stationary.0);
    let len_xi = length_xi(n, num_stationary);
    // objective: minimize \sum  weight_a[k] / sqrt(a[k])
    let mut objective = 0.0;
    for (a_sqrt_down, weight_a) in a_sqrt_down
        .iter()
        .zip(weight_a.iter())
        .skip(k_skip)
        .take(len_xi)
    {
        objective += weight_a * a_sqrt_down;
    }
    objective
}

/// Compute the thermal energy value in COPP3 optimization.
#[cfg(test)]
#[inline(always)]
fn objective_value_thermal_energy_copp3(
    a_sqrt_down: &[f64],
    weight_a: &[f64],
    num_stationary: (usize, usize),
    torque: &DMatrix<f64>,
    normalize: &[f64],
) -> f64 {
    let mut objective = 0.0;
    let n = a_sqrt_down.len() - 1;
    if num_stationary.0 > 0 {
        // minmize: weight_a[0] / sqrt(a[num_stationary.0]) * \sum[i] { (tau_average[i] * normalize[i])^2 }
        // tau_average[i] = (tau_normal[i][0] + tau_normal[i][num_stationary.0]) / 2
        let torque_n1 = torque.column(num_stationary.0);
        let torque_0 = torque.column(0);
        let weight = weight_a[0] * a_sqrt_down[num_stationary.0];
        for (tau_n1, tau_0, &normalize_i) in
            izip!(torque_n1.iter(), torque_0.iter(), normalize.iter())
        {
            let tau_average = 0.5 * normalize_i * (tau_n1 + tau_0);
            objective += weight * tau_average * tau_average;
        }
    }
    if num_stationary.1 > 0 {
        // minmize: weight_a[n] / sqrt(a[n-num_stationary.1]) * \sum[i] { (tau_average[i] * normalize[i])^2 }
        // tau_average[i] = (tau_normal[i][n] + tau_normal[i][n-num_stationary.1]) / 2
        let torque_n2 = torque.column(n - num_stationary.1);
        let torque_n = torque.column(n);
        let weight = weight_a[n] * a_sqrt_down[n - num_stationary.1];
        for (tau_n2, tau_n, &normalize_i) in
            izip!(torque_n2.iter(), torque_n.iter(), normalize.iter())
        {
            let tau_average = 0.5 * normalize_i * (tau_n2 + tau_n);
            objective += weight * tau_average * tau_average;
        }
    }
    let k_skip = skip_a_for_xi(num_stationary.0);
    let len_xi = length_xi(n, num_stationary);
    for (torque_k, &a_sqrt_down_k, &weight_a_k) in
        izip!(torque.column_iter(), a_sqrt_down.iter(), weight_a.iter())
            .skip(k_skip)
            .take(len_xi)
    {
        for (tau_k, &normalize_i) in torque_k.iter().zip(normalize.iter()) {
            let tau_normal = tau_k * normalize_i;
            objective += 4.0 * weight_a_k * tau_normal * tau_normal * a_sqrt_down_k;
        }
    }

    objective
}

/// Compute the thermal energy value in COPP3 optimization.
#[cfg(test)]
#[inline(always)]
fn objective_value_tv_torque_copp3(
    torque: &DMatrix<f64>,
    normalize: &[f64],
    num_stationary: (usize, usize),
) -> f64 {
    let mut objective = 0.0;
    let n = torque.ncols() - 1;
    if num_stationary.0 > 0 {
        // |tau[i][num_stationary.0] - tau[i][0]| * normalize[i]
        let torque_n1 = torque.column(num_stationary.0);
        let torque_0 = torque.column(0);
        for (tau_n1, tau_0, &normalize_i) in
            izip!(torque_n1.iter(), torque_0.iter(), normalize.iter())
        {
            objective += normalize_i * (tau_n1 - tau_0).abs();
        }
    }
    if num_stationary.1 > 0 {
        // |tau[i][n-num_stationary.1] - tau[i][n]| * normalize[i]
        let torque_n2 = torque.column(n - num_stationary.1);
        let torque_n = torque.column(n);
        for (tau_n2, tau_n, &normalize_i) in
            izip!(torque_n2.iter(), torque_n.iter(), normalize.iter())
        {
            objective += normalize_i * (tau_n2 - tau_n).abs();
        }
    }
    // Consider (tau[i][k+1] - tau[i][k]) * normalize[i] for k in num_stationary.0..(n-num_stationary.1)
    for (torque_col_curr, torque_col_next) in torque
        .column_iter()
        .tuple_windows()
        .skip(num_stationary.0)
        .take(n - num_stationary.0 - num_stationary.1)
    {
        for (tau_k, tau_k_next, &normalize_i) in izip!(
            torque_col_curr.iter(),
            torque_col_next.iter(),
            normalize.iter()
        ) {
            objective += normalize_i * (tau_k_next - tau_k).abs();
        }
    }

    objective
}

/// Compute the objective value for Linear in COPP3 optimization.
#[cfg(test)]
#[inline(always)]
fn objective_value_linear_copp3(
    a_profile: &[f64],
    b_profile: &[f64],
    alpha: &[f64],
    beta: &[f64],
) -> f64 {
    // objective: minimize \sum (alpha[k]*a[k] + beta[k]*b[k])
    let mut objective = 0.0;
    for (a_curr, alpha_curr) in a_profile.iter().zip(alpha.iter()) {
        // alpha[k]*a[k]
        objective += a_curr * alpha_curr;
    }
    for (b_curr, beta_curr) in b_profile.iter().zip(beta.iter()) {
        // beta[k]*b[k]
        objective += b_curr * beta_curr;
    }
    objective
}

/// Compute the objective value for COPP3 optimization.
#[cfg(test)]
pub(crate) fn objective_value_copp3_opt<M: RobotTorque>(
    problem: &Copp3Problem<M>,
    a_profile: &[f64],
    b_profile: &[f64],
    num_stationary: (usize, usize),
) -> (f64, Vec<f64>) {
    let s = problem
        .robot
        .constraints
        .s_vec(problem.idx_s_start, problem.idx_s_start + a_profile.len());
    let Ok(s) = s else {
        return (f64::INFINITY, vec![0.0; problem.objectives.len()]);
    };
    if a_profile.len() != s.len() || b_profile.len() != s.len() {
        return (f64::INFINITY, vec![0.0; problem.objectives.len()]);
    }
    let (a_sqrt_down, weight_a_time) = if problem.objectives.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::Time(_) | CoppObjective::ThermalEnergy(_, _)
        )
    }) {
        (
            a_profile
                .iter()
                .map(|a| 1.0 / a.sqrt().max(1E-16))
                .collect(),
            get_weight_a_topp3(&s, num_stationary),
        )
    } else {
        (vec![], vec![])
    };
    let weight_a_torque = if problem
        .objectives
        .iter()
        .any(|obj| matches!(obj, CoppObjective::ThermalEnergy(_, _)))
    {
        get_weight_a_copp3(&s, num_stationary)
    } else {
        vec![]
    };
    let torque = if problem.objectives.iter().any(|obj| {
        matches!(
            obj,
            CoppObjective::ThermalEnergy(_, _) | CoppObjective::TotalVariationTorque(_, _)
        )
    }) {
        let torque_result =
            problem
                .robot
                .get_torque_with_ab(a_profile, b_profile, problem.idx_s_start);
        match torque_result {
            Ok(torque) => torque,
            _ => return (f64::INFINITY, vec![0.0; problem.objectives.len()]),
        }
    } else {
        DMatrix::<f64>::zeros(0, 0)
    };
    let mut obj_val = Vec::with_capacity(problem.objectives.len());
    let mut obj_val_total = 0.0;
    for obj in problem.objectives {
        match obj {
            CoppObjective::Time(weight) => {
                let obj_here =
                    objective_value_time_copp3(&a_sqrt_down, &weight_a_time, num_stationary);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
            CoppObjective::ThermalEnergy(weight, normalize) => {
                let obj_here = objective_value_thermal_energy_copp3(
                    &a_sqrt_down,
                    &weight_a_torque,
                    num_stationary,
                    &torque,
                    normalize,
                );
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
            CoppObjective::TotalVariationTorque(weight, normalize) => {
                let obj_here = objective_value_tv_torque_copp3(&torque, normalize, num_stationary);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                let obj_here = objective_value_linear_copp3(a_profile, b_profile, alpha, beta);
                obj_val.push(obj_here);
                obj_val_total += weight * obj_here;
            }
        }
    }
    (obj_val_total, obj_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::copp2::stable::basic::{Topp2ProblemBuilder, s_to_t_topp2};
    use crate::copp::copp2::stable::reach_set2::ReachSet2OptionsBuilder;
    use crate::copp::copp2::stable::topp2_ra::topp2_ra;
    use crate::copp::copp3::stable::basic::{Copp3ProblemBuilder, s_to_t_topp3};
    use crate::copp::copp3::stable::topp3_lp::topp3_lp;
    use crate::copp::copp3::stable::topp3_socp::topp3_socp;
    use crate::copp::{ClarabelOptionsBuilder, default_clarabel_settings};
    use crate::path::{add_symmetric_axial_limits_for_test, lissajous_path_for_test};
    use crate::robot::robot_core::Robot;
    use std::time::Instant;
    use std::vec;

    #[test]
    fn test_copp3_lp() -> Result<(), CoppError> {
        run_test_copp3_lp_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average 100 experiments: tc_ra = 0.3270 ms, tc_lp = 237.4646 ms, tc_copp = 232.0414 ms, tf_ra = 6.304760, tf_lp = 6.524883, tf_copp = 6.524883, obj_lp = -0.031621, obj_copp = -0.031621
    #[test]
    #[ignore = "slow"]
    fn test_copp3_lp_robust() -> Result<(), CoppError> {
        run_test_copp3_lp_repeated(100, true)
    }

    #[test]
    fn test_copp3_qp() -> Result<(), CoppError> {
        run_test_copp3_qp_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average 100 experiments (fail 0): tc_ra = 0.3452 ms, tc_qp = 329.8604 ms, tc_copp = 302.8725 ms, tf_ra = 6.122942, tf_qp = 6.342830, tf_copp = 6.342830, obj_qp = 6.348896, obj_copp = 6.348896
    #[test]
    #[ignore = "slow"]
    fn test_copp3_qp_robust() -> Result<(), CoppError> {
        run_test_copp3_qp_repeated(100, true)
    }

    #[test]
    fn test_all_objectives() -> Result<(), CoppError> {
        run_test_all_objectives_repeated(1, false)
    }

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// Average 100 experiments (fail 0):
    /// Case 0: tc=301.449ms, obj=[6.355272621082504, 37.88484670012183, 30.538889633761094, 0.0014588588234334063]
    /// Case 1: tc=345.585ms, obj=[8.929951413267652, 11.232320784989641, 15.611002914308818, 0.001319403151583034]
    /// Case 2: tc=303.617ms, obj=[15.7679747735929, 2.0044043744079847, 5.695644764673545, 0.0010702913648672737]
    /// Case 3: tc=440.100ms, obj=[11.999657677994689, 5.6638102612967565, 5.999807538045939, 0.00022405699762245119]
    /// Case 4: tc=306.763ms, obj=[6.355272615415654, 37.884856482672014, 30.621301350094082, 0.0014588597553749254]
    #[test]
    #[ignore = "slow"]
    fn test_all_objectives_robust() -> Result<(), CoppError> {
        run_test_all_objectives_repeated(100, true)
    }

    fn run_test_copp3_lp_repeated(n_exp: usize, flag_print_step: bool) -> Result<(), CoppError> {
        let mut tc_sum_ra = 0.0;
        let mut tc_sum_lp = 0.0;
        let mut tc_sum_copp = 0.0;
        let mut tf_sum_ra = 0.0;
        let mut tf_sum_lp = 0.0;
        let mut tf_sum_copp = 0.0;
        let mut obj_sum_lp = 0.0;
        let mut obj_sum_copp = 0.0;

        for i_exp in 0..n_exp {
            let n: usize = 1000;
            let dim = 7;
            let mut robot = Robot::with_capacity(dim, n);

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
            add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, Some(5.0))?;

            let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
            // Step 1. Topp2-RA
            let start = Instant::now();
            let options_ra0 = ReachSet2OptionsBuilder::new()
                .lp_feas_tol(1E-9)
                .a_cmp_abs_tol(1E-9)
                .a_cmp_rel_tol(1E-9)
                .build()?;
            let a_ra0 = topp2_ra(&topp2_problem, &options_ra0)?;
            let tc_ra0 = start.elapsed().as_secs_f64() * 1E3;
            let (tf_ra0, _) = s_to_t_topp2(s.as_slice(), &a_ra0, 0.0);

            let objectives = [CoppObjective::Linear(
                1.0,
                &get_weight_a_topp3(&s.as_slice()[0..n], (1, 1))
                    .iter()
                    .map(|&w_a| -w_a)
                    .collect_vec(),
                &vec![0.0; n],
            )];
            let copp3_problem = Copp3ProblemBuilder::new(
                &mut robot,
                &objectives,
                0,
                &a_ra0,
                (0.0, 0.0),
                (0.0, 0.0),
            )
            .build_with_linearization()?;

            // Step 2. Test Copp3-SOCP
            let start = Instant::now();
            let (a_copp, b_copp, num_stationary_copp) = {
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = copp3_socp_expert(&copp3_problem, &options)?;
                if let Some(result) = result {
                    result
                } else {
                    return Err(CoppError::ClarabelSolverStatus(
                        "copp3_socp".into(),
                        solution.status,
                    ));
                }
            };
            let tc_copp = start.elapsed().as_secs_f64() * 1E3;
            // Test time profile generation
            let (tf_copp, _) =
                s_to_t_topp3(s.as_slice(), &a_copp, &b_copp, num_stationary_copp, 0.0);

            // Step 3. Test Topp3-LP
            let options_lp = ClarabelOptionsBuilder::new()
                .allow_almost_solved(true)
                .build()?;
            let start = Instant::now();
            let (a_lp, b_lp, _) = topp3_lp(&copp3_problem.as_topp3_problem(), &options_lp)?;
            let tc_lp = start.elapsed().as_secs_f64() * 1E3;
            // Test time profile generation
            let (tf_lp, _) = s_to_t_topp3(s.as_slice(), &a_lp, &b_lp, num_stationary_copp, 0.0);

            let (obj_lp, _) =
                objective_value_copp3_opt(&copp3_problem, &a_lp, &b_lp, num_stationary_copp);
            let (obj_copp, _) =
                objective_value_copp3_opt(&copp3_problem, &a_copp, &b_copp, num_stationary_copp);

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_copp = {:.4} ms, tf_ra = {:.6}, tf_lp = {:.6}, tf_copp = {:.6}, obj_lp = {:.6}, obj_copp = {:.6}",
                    i_exp + 1,
                    tc_ra0,
                    tc_lp,
                    tc_copp,
                    tf_ra0,
                    tf_lp,
                    tf_copp,
                    obj_lp,
                    obj_copp
                );
            }

            if (tf_copp - tf_lp).abs() > 1e-8 {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Debug,
                    "COPP3 time optimality failed! tf_copp - tf_lp = {}",
                    tf_copp - tf_lp
                );
            }

            tc_sum_ra += tc_ra0;
            tc_sum_lp += tc_lp;
            tc_sum_copp += tc_copp;
            tf_sum_ra += tf_ra0;
            tf_sum_lp += tf_lp;
            tf_sum_copp += tf_copp;
            obj_sum_lp += obj_lp;
            obj_sum_copp += obj_copp;
        }

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average {} experiments: tc_ra = {:.4} ms, tc_lp = {:.4} ms, tc_copp = {:.4} ms, tf_ra = {:.6}, tf_lp = {:.6}, tf_copp = {:.6}, obj_lp = {:.6}, obj_copp = {:.6}",
            n_exp,
            tc_sum_ra / n_exp as f64,
            tc_sum_lp / n_exp as f64,
            tc_sum_copp / n_exp as f64,
            tf_sum_ra / n_exp as f64,
            tf_sum_lp / n_exp as f64,
            tf_sum_copp / n_exp as f64,
            obj_sum_lp / n_exp as f64,
            obj_sum_copp / n_exp as f64
        );

        Ok(())
    }

    fn run_test_copp3_qp_repeated(n_exp: usize, flag_print_step: bool) -> Result<(), CoppError> {
        let mut tc_sum_ra = 0.0;
        let mut tc_sum_qp = 0.0;
        let mut tc_sum_copp = 0.0;
        let mut tf_sum_ra = 0.0;
        let mut tf_sum_qp = 0.0;
        let mut tf_sum_copp = 0.0;
        let mut obj_sum_qp = 0.0;
        let mut obj_sum_copp = 0.0;
        let mut succeed = 0;

        for i_exp in 0..n_exp {
            let n: usize = 1000;
            let dim = 7;
            let mut robot = Robot::with_capacity(dim, n);

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
            add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, Some(5.0))?;

            let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
            // Step 1. Topp2-RA
            let start = Instant::now();
            let options_ra0 = ReachSet2OptionsBuilder::new()
                .lp_feas_tol(1E-9)
                .a_cmp_abs_tol(1E-9)
                .a_cmp_rel_tol(1E-9)
                .build()?;
            let a_ra0 = topp2_ra(&topp2_problem, &options_ra0)?;
            let tc_ra0 = start.elapsed().as_secs_f64() * 1E3;
            let (tf_ra0, _) = s_to_t_topp2(s.as_slice(), &a_ra0, 0.0);

            let objective = [CoppObjective::Time(1.0)];
            let copp3_problem =
                Copp3ProblemBuilder::new(&mut robot, &objective, 0, &a_ra0, (0.0, 0.0), (0.0, 0.0))
                    .build_with_linearization()?;

            // Step 2. Test Copp3-SOCP
            let start = Instant::now();
            let (a_copp, b_copp, num_stationary_copp) = {
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = match copp3_socp_expert(&copp3_problem, &options) {
                    Ok(res) => res,
                    Err(_) => {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "Exp #{}: Clarabel solver failed in copp3_socp_expert!",
                            i_exp + 1
                        );
                        continue;
                    }
                };
                if let Some(result) = result {
                    result
                } else {
                    return Err(CoppError::ClarabelSolverStatus(
                        "copp3_socp".into(),
                        solution.status,
                    ));
                }
            };
            let tc_copp = start.elapsed().as_secs_f64() * 1E3;
            // Test time profile generation
            let (tf_copp, _) =
                s_to_t_topp3(s.as_slice(), &a_copp, &b_copp, num_stationary_copp, 0.0);

            // Step 3. Test Topp3-LP
            let start = Instant::now();
            let options_qp = ClarabelOptionsBuilder::new()
                .allow_almost_solved(true)
                .build()?;
            let (a_qp, b_qp, _) = topp3_socp(&copp3_problem.as_topp3_problem(), &options_qp)?;
            let tc_qp = start.elapsed().as_secs_f64() * 1E3;
            // Test time profile generation
            let (tf_qp, _) = s_to_t_topp3(s.as_slice(), &a_qp, &b_qp, num_stationary_copp, 0.0);

            let (obj_qp, _) =
                objective_value_copp3_opt(&copp3_problem, &a_qp, &b_qp, num_stationary_copp);
            let (obj_copp, _) =
                objective_value_copp3_opt(&copp3_problem, &a_copp, &b_copp, num_stationary_copp);

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}: tc_ra = {:.4} ms, tc_qp = {:.4} ms, tc_copp = {:.4} ms, tf_ra = {:.6}, tf_qp = {:.6}, tf_copp = {:.6}, obj_qp = {:.6}, obj_copp = {:.6}",
                    i_exp + 1,
                    tc_ra0,
                    tc_qp,
                    tc_copp,
                    tf_ra0,
                    tf_qp,
                    tf_copp,
                    obj_qp,
                    obj_copp
                );
            }

            if (tf_copp - tf_qp).abs() > 1e-4 || (obj_copp - obj_qp).abs() > 1e-4 {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Debug,
                    "COPP3 time optimality failed at Exp #{}! tf_copp - tf_qp = {}, obj_copp - obj_qp = {}",
                    i_exp + 1,
                    tf_copp - tf_qp,
                    obj_copp - obj_qp
                );
            }

            tc_sum_ra += tc_ra0;
            tc_sum_qp += tc_qp;
            tc_sum_copp += tc_copp;
            tf_sum_ra += tf_ra0;
            tf_sum_qp += tf_qp;
            tf_sum_copp += tf_copp;
            obj_sum_qp += obj_qp;
            obj_sum_copp += obj_copp;
            succeed += 1;
        }

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average {n_exp} experiments (fail {}): tc_ra = {:.4} ms, tc_qp = {:.4} ms, tc_copp = {:.4} ms, tf_ra = {:.6}, tf_qp = {:.6}, tf_copp = {:.6}, obj_qp = {:.6}, obj_copp = {:.6}",
            n_exp - succeed,
            tc_sum_ra / succeed as f64,
            tc_sum_qp / succeed as f64,
            tc_sum_copp / succeed as f64,
            tf_sum_ra / succeed as f64,
            tf_sum_qp / succeed as f64,
            tf_sum_copp / succeed as f64,
            obj_sum_qp / succeed as f64,
            obj_sum_copp / succeed as f64
        );

        Ok(())
    }

    fn run_test_all_objectives_repeated(
        n_exp: usize,
        flag_print_step: bool,
    ) -> Result<(), CoppError> {
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

        let mut succeed = 0;

        for i_exp in 0..n_exp {
            let n: usize = 1000;
            let dim = 7;
            let mut robot = Robot::with_capacity(dim, n);

            let mut rng = rand::rng();
            let (s, derivs, omega, phi) =
                lissajous_path_for_test(dim, n, &mut rng).expect("random range is valid");
            // let omega: Vec<f64> = [
            //     4.723299430689122,
            //     1.2937933921386273,
            //     3.163544832429868,
            //     2.554983998551911,
            //     3.1752440065420076,
            //     6.036677287860053,
            //     5.4585988904049145,
            // ]
            // .into();
            // let phi: Vec<f64> = [
            //     0.3140306040123813,
            //     5.341058950177041,
            //     4.076168494444343,
            //     5.801150775658696,
            //     2.306341572703676,
            //     0.2550634807632447,
            //     3.986342358667442,
            // ]
            // .into();

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
            }
            robot.with_s(&s.as_view())?;
            robot.with_q(
                &derivs.q.as_view(),
                &derivs.dq.as_ref().unwrap().as_view(),
                &derivs.ddq.as_ref().unwrap().as_view(),
                derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
                0,
            )?;
            add_symmetric_axial_limits_for_test(&mut robot, 1.0, 1.0, Some(5.0))?;

            let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
            // Step 1. Topp2-RA
            let options_ra0 = ReachSet2OptionsBuilder::new()
                .lp_feas_tol(1E-9)
                .a_cmp_abs_tol(1E-9)
                .a_cmp_rel_tol(1E-9)
                .build()?;
            let a_ra0 = topp2_ra(&topp2_problem, &options_ra0)?;

            // Test different objectives in COPP2 optimization
            let objectives_test = [
                CoppObjective::Time(1.0),
                CoppObjective::ThermalEnergy(1.0, &vec![1.0; dim]),
                CoppObjective::TotalVariationTorque(1.0, &vec![1.0; dim]),
                CoppObjective::Linear(1.0, &vec![0.0; n], &vec![1.0; n - 1]),
            ];

            // Case 0: Time only
            let obj_case0_src = [CoppObjective::Time(1.0)];
            let start = Instant::now();
            let (a_case0, b_case0, num_stationary) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &obj_case0_src,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = copp3_socp_expert(&copp3_problem, &options)?;
                if let Some(result) = result {
                    result
                } else {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "{:?}",
                        CoppError::ClarabelSolverStatus(
                            "copp3_socp (case 0)".into(),
                            solution.status,
                        )
                    );
                    continue;
                }
            };
            let tc_copp3_case0 = start.elapsed().as_secs_f64() * 1E3;
            let (_, obj_case0) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives_test,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                objective_value_copp3_opt(&copp3_problem, &a_case0, &b_case0, num_stationary)
            };

            // Case 1: Time and ThermalEnergy
            let obj_case1 = [
                CoppObjective::Time(1.0),
                CoppObjective::ThermalEnergy(1.0, &vec![1.0; dim]),
            ];
            let start = Instant::now();
            let (a_case1, b_case1, num_stationary) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &obj_case1,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = copp3_socp_expert(&copp3_problem, &options)?;
                if let Some(result) = result {
                    result
                } else {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "{:?}",
                        CoppError::ClarabelSolverStatus(
                            "copp3_socp (case 1)".into(),
                            solution.status,
                        )
                    );
                    continue;
                }
            };
            let tc_copp3_case1 = start.elapsed().as_secs_f64() * 1E3;
            let (_, obj_case1) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives_test,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                objective_value_copp3_opt(&copp3_problem, &a_case1, &b_case1, num_stationary)
            };
            if obj_case1[0] < obj_case0[0] - 1E-3 || obj_case1[1] - 1E-3 > obj_case0[1] {
                let (tf_case0, _) = s_to_t_topp2(s.as_slice(), &a_case0, 0.0);
                let (tf_case1, _) = s_to_t_topp2(s.as_slice(), &a_case1, 0.0);
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 0: obj_time = {}, obj_thermal_energy = {}, tf = {}",
                    obj_case0[0],
                    obj_case0[1],
                    tf_case0
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {}, obj_thermal_energy = {}, tf = {}",
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
            let start = Instant::now();
            let (a_case2, b_case2, num_stationary) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &obj_case2,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = copp3_socp_expert(&copp3_problem, &options)?;
                if let Some(result) = result {
                    result
                } else {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "{:?}",
                        CoppError::ClarabelSolverStatus(
                            "copp3_socp (case 2)".into(),
                            solution.status,
                        )
                    );
                    continue;
                }
            };
            let tc_copp3_case2 = start.elapsed().as_secs_f64() * 1E3;
            let (_, obj_case2) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives_test,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                objective_value_copp3_opt(&copp3_problem, &a_case2, &b_case2, num_stationary)
            };
            if obj_case2[0] < obj_case1[0] - 1E-3 || obj_case2[1] - 1E-3 > obj_case1[1] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {}, obj_thermal_energy = {}",
                    obj_case1[0],
                    obj_case1[1]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 2: obj_time = {}, obj_thermal_energy = {}",
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
            let start = Instant::now();
            let (a_case3, b_case3, num_stationary) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &obj_case3,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = copp3_socp_expert(&copp3_problem, &options)?;
                if let Some(result) = result {
                    result
                } else {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "{:?}",
                        CoppError::ClarabelSolverStatus(
                            "copp3_socp (case 3)".into(),
                            solution.status,
                        )
                    );
                    continue;
                }
            };
            let tc_copp3_case3 = start.elapsed().as_secs_f64() * 1E3;
            let (_, obj_case3) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives_test,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                objective_value_copp3_opt(&copp3_problem, &a_case3, &b_case3, num_stationary)
            };
            if obj_case3[1] < obj_case1[1] - 1E-3 || obj_case3[2] - 1E-3 > obj_case1[2] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {}, obj_thermal_energy = {}, obj_total_variation_torque = {}",
                    obj_case1[0],
                    obj_case1[1],
                    obj_case1[2]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 3: obj_time = {}, obj_thermal_energy = {}, obj_total_variation_torque = {}",
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
                CoppObjective::Linear(1.0, &vec![0.0; n], &vec![1.0; n]),
            ];
            let start = Instant::now();
            let (a_case4, b_case4, num_stationary) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &obj_case4,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                let mut settings = default_clarabel_settings();
                settings.tol_gap_rel = 1E-6;
                let options = ClarabelOptionsBuilder::with_clarabel_setting(settings)
                    .allow_almost_solved(true)
                    .build()?;
                let (result, solution) = copp3_socp_expert(&copp3_problem, &options)?;
                if let Some(result) = result {
                    result
                } else {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "{:?}",
                        CoppError::ClarabelSolverStatus(
                            "copp3_socp (case 4)".into(),
                            solution.status,
                        )
                    );
                    continue;
                }
            };
            let tc_copp3_case4 = start.elapsed().as_secs_f64() * 1E3;
            let (_, obj_case4) = {
                let copp3_problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives_test,
                    0,
                    &a_ra0,
                    (0.0, 0.0),
                    (0.0, 0.0),
                )
                .build_with_linearization()?;
                objective_value_copp3_opt(&copp3_problem, &a_case4, &b_case4, num_stationary)
            };
            if obj_case4[1] < obj_case1[1] - 1E-3 || obj_case4[3] - 1E-3 > obj_case1[3] {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "omega = {omega:?}\nphi = {phi:?}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 1: obj_time = {}, obj_thermal_energy = {}, obj_linear = {}",
                    obj_case1[0],
                    obj_case1[1],
                    obj_case1[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 4: obj_time = {}, obj_thermal_energy = {}, obj_linear = {}",
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
                    "Case 2: obj_time = {}, obj_total_variation_torque = {}, obj_linear = {}",
                    obj_case2[0],
                    obj_case2[2],
                    obj_case2[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Case 4: obj_time = {}, obj_total_variation_torque = {}, obj_linear = {}",
                    obj_case4[0],
                    obj_case4[2],
                    obj_case4[3]
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Interesting... Cases 2 and 4"
                );
            }

            succeed += 1;

            if flag_print_step {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}:\n Case 0: tc={:.3}ms, obj={:?}\n Case 1: tc={:.3}ms, obj={:?}\n Case 2: tc={:.3}ms, obj={:?}\n Case 3: tc={:.3}ms, obj={:?}\n Case 4: tc={:.3}ms, obj={:?}",
                    i_exp + 1,
                    tc_copp3_case0,
                    obj_case0,
                    tc_copp3_case1,
                    obj_case1,
                    tc_copp3_case2,
                    obj_case2,
                    tc_copp3_case3,
                    obj_case3,
                    tc_copp3_case4,
                    obj_case4
                );
            }

            tc_sum_case0 += tc_copp3_case0;
            tc_sum_case1 += tc_copp3_case1;
            tc_sum_case2 += tc_copp3_case2;
            tc_sum_case3 += tc_copp3_case3;
            tc_sum_case4 += tc_copp3_case4;
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
            "Average {n_exp} experiments (fail {}):\n Case 0: tc={:.3}ms, obj={obj_sum_case0:?}\n Case 1: tc={:.3}ms, obj={obj_sum_case1:?}\n Case 2: tc={:.3}ms, obj={obj_sum_case2:?}\n Case 3: tc={:.3}ms, obj={obj_sum_case3:?}\n Case 4: tc={:.3}ms, obj={obj_sum_case4:?}",
            n_exp - succeed,
            tc_sum_case0 / succeed as f64,
            tc_sum_case1 / succeed as f64,
            tc_sum_case2 / succeed as f64,
            tc_sum_case3 / succeed as f64,
            tc_sum_case4 / succeed as f64
        );

        Ok(())
    }
}
