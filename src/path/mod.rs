//! Path abstraction for trajectory planning.
//!
//! The main entry point is [`Path`].  It supports two construction modes and a
//! uniform evaluation API — pick the one that fits your use case:
//!
//! | You have … | Use |
//! |---|---|
//! | An analytic formula `q(s)` | [`Path::from_parametric`] |
//! | A set of waypoint positions | [`Path::from_waypoints`] |
//!
//! Once built, call one of the evaluation methods with a one-dimensional parameter slice:
//!
//! | Method | Output |
//! |---|---|
//! | [`Path::evaluate_q`]         | position `q` only |
//! | [`Path::evaluate_up_to_2nd`] | `q`, `dq`, `ddq` |
//! | [`Path::evaluate_up_to_3rd`] | `q`, `dq`, `ddq`, `dddq` |
//!
//! # Examples
//!
//! ## Analytic path — automatic differentiation
//!
//! ```rust,no_run
//! use copp::path::{Path, sin, cos};
//! use copp::path::autodiff::Jet3;
//!
//! // Build a 2-DOF path: q0 = sin(s), q1 = cos(s), s ∈ [0, 1]
//! let path = Path::from_parametric(
//!     |s: Jet3| vec![sin(s), cos(s)],
//!     0.0, 1.0,
//! ).unwrap();
//!
//! // Evaluate at 5 uniformly-spaced parameter values
//! let s = [0.0, 0.25, 0.5, 0.75, 1.0];
//! let out = path.evaluate_up_to_3rd(&s).unwrap();
//! // out.q    – shape (2, 5)
//! // out.dq   – Some, shape (2, 5)   first derivative w.r.t. s
//! // out.dddq – Some, shape (2, 5)   third derivative w.r.t. s
//! ```
//!
//! ## Spline path — waypoint interpolation
//!
//! ```rust,no_run
//! use copp::path::{Path, SplineConfig};
//! use nalgebra::DMatrix;
//!
//! // 2-DOF path with 5 waypoints; the spline passes exactly through each one.
//! // waypoints shape: (dim=2, n_points=5)
//! let waypoints = DMatrix::from_row_slice(2, 5, &[
//!     0.0, 0.25, 0.5, 0.75, 1.0,   // dim 0
//!     0.0, 0.1, -0.1, 0.2,  0.0,   // dim 1
//! ]);
//! let path = Path::from_waypoints(&waypoints, SplineConfig::default()).unwrap();
//!
//! let s = [0.0, 0.5, 1.0];
//! let out = path.evaluate_q(&s).unwrap();
//! // out.q shape (2, 3);  out.dq / ddq / dddq are all None
//! ```

pub mod autodiff;
mod path_core;
pub mod spline;

pub use autodiff::{Jet3, cos, exp, ln, powi, sin, sqrt};
pub use path_core::{ParametricFn, Path, PathDerivatives};
pub use spline::{Parametrization, SplineConfig};

#[cfg(test)]
use crate::diag::PathError;

#[derive(Clone, Copy, Debug)]
pub enum OutOfRangeMode {
    /// Return an error when s is outside `[s_min, s_max]`.
    Error,
    /// Silently clamp s to `[s_min, s_max]`.
    Clamp,
}

/// Build a Lissajous analytic path and evaluate derivatives up to third order.
///
/// This helper is intended for cross-module unit tests to avoid repeating the
/// same hand-written `q/dq/ddq/dddq` generation logic.
#[cfg(test)]
pub(crate) fn lissajous_path_for_test(
    dim: usize,
    s_len: usize,
    rng: &mut impl rand::RngExt,
) -> Result<(nalgebra::DMatrix<f64>, PathDerivatives, Vec<f64>, Vec<f64>), PathError> {
    let omega = (0..dim)
        .map(|_| rng.random_range(0.1..(2.0 * std::f64::consts::PI)))
        .collect::<Vec<f64>>();
    let phi = (0..dim)
        .map(|_| rng.random_range(0.0..(2.0 * std::f64::consts::PI)))
        .collect::<Vec<f64>>();

    let (s, derivs) = lissajous_path_fixed_for_test(dim, s_len, omega.clone(), phi.clone())?;

    Ok((s, derivs, omega, phi))
}

/// Build a Lissajous analytic path and evaluate derivatives up to third order.
///
/// This helper is intended for cross-module unit tests to avoid repeating the
/// same hand-written `q/dq/ddq/dddq` generation logic.
#[cfg(test)]
pub(crate) fn lissajous_path_fixed_for_test(
    dim: usize,
    s_len: usize,
    omega: Vec<f64>,
    phi: Vec<f64>,
) -> Result<(nalgebra::DMatrix<f64>, PathDerivatives), PathError> {
    let s = nalgebra::DMatrix::<f64>::from_fn(1, s_len, |_, j| j as f64 / (s_len - 1) as f64);

    let path = Path::from_parametric(
        move |s: Jet3| {
            (0..dim)
                .map(|i| sin(omega[i] * s + phi[i]))
                .collect::<Vec<Jet3>>()
        },
        0.0,
        1.0,
    )?;

    let derivs = path.evaluate_up_to_3rd(s.as_slice())?;
    Ok((s, derivs))
}

/// Add symmetric axial limits (`+limit` / `-limit`) to all currently stored stations.
///
/// The helper applies velocity, acceleration and jerk constraints in one call.
#[cfg(test)]
pub(crate) fn add_symmetric_axial_limits_for_test<M: crate::robot::robot_core::RobotBasic>(
    robot: &mut crate::robot::robot_core::Robot<M>,
    vel_limit: f64,
    acc_limit: f64,
    jerk_limit: Option<f64>,
) -> Result<(), crate::diag::ConstraintError> {
    let dim = robot.dim();
    let n = robot.constraints.len();
    if n == 0 {
        return Ok(());
    }

    let vel_max = vec![vel_limit; dim];
    let vel_min = vec![-vel_limit; dim];
    robot.with_axial_velocity((vel_max.as_slice(), n), (vel_min.as_slice(), n), 0)?;

    let acc_max = vec![acc_limit; dim];
    let acc_min = vec![-acc_limit; dim];
    robot.with_axial_acceleration((acc_max.as_slice(), n), (acc_min.as_slice(), n), 0)?;

    if let Some(jerk_limit) = jerk_limit {
        let jerk_max = vec![jerk_limit; dim];
        let jerk_min = vec![-jerk_limit; dim];
        robot.with_axial_jerk((jerk_max.as_slice(), n), (jerk_min.as_slice(), n), 0)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copp::copp2::stable::basic::{Topp2ProblemBuilder, s_to_t_topp2};
    use crate::copp::copp2::stable::reach_set2::ReachSet2OptionsBuilder;
    use crate::copp::copp2::stable::topp2_ra::topp2_ra;
    use crate::copp::copp3::stable::basic::{Topp3ProblemBuilder, s_to_t_topp3, t_to_s_topp3};
    use crate::copp::copp3::stable::topp3_lp::topp3_lp;
    use crate::copp::{ClarabelOptionsBuilder, InterpolationMode};
    use crate::diag::ConstraintError;
    use crate::diag::Verbosity;
    use crate::robot::robot_core::Robot;
    use nalgebra::DMatrix;
    use plotters::prelude::*;
    use rand::RngExt;
    use std::error::Error;
    use std::fs::create_dir_all;
    use std::path::Path;
    use std::time::Instant;

    const DIM: usize = 6;
    const N: usize = 1000;

    #[test]
    #[ignore = "plotting"]
    fn test_topp3_with_spline_path() -> Result<(), Box<dyn Error>> {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "\n=== TOPP3 with Spline Path (6-DOF) ==="
        );

        let n_waypoints = 10;
        let start = Instant::now();
        let waypoints = make_random_waypoints(DIM, n_waypoints);
        let sample_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let path = super::Path::from_waypoints(&waypoints, SplineConfig::default())?;
        let build_ms = start.elapsed().as_secs_f64() * 1e3;

        let s = make_s_vector(N);
        let start = Instant::now();
        let derivs = path.evaluate_up_to_3rd(s.as_slice())?;
        let eval_ms = start.elapsed().as_secs_f64() * 1e3;

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[Waypoint Sample] {sample_ms:.3} ms, [Spline Build] {build_ms:.3} ms, [Spline Eval] {eval_ms:.3} ms"
        );

        let mut robot = setup_robot_with_constraints(&s, &derivs)?;
        let s_slice: Vec<f64> = (0..N).map(|j| s[(0, j)]).collect();
        let (a, b, num_stat, timings) = solve_topp_pipeline(&mut robot, &s_slice)?;

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[TOPP2-RA] {:.3} ms, t_motion = {:.3} s",
            timings.topp2_ms,
            timings.t_motion_2
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[TOPP3-LP] {:.3} ms, t_motion = {:.3} s",
            timings.topp3_ms,
            timings.t_motion_3
        );

        let (t, q_t, dq_t, ddq_t, dddq_t, interp_ms) =
            interpolate_to_time_domain(&path, &s_slice, &a, &b, num_stat, 1e-3)?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[Interpolation] {:.3} ms, {} samples",
            interp_ms,
            t.len()
        );

        plot_topp3_grid(
            "data/path_topp_plots/spline_path_topp3.png",
            "TOPP3 Result: Spline Path (6-DOF, 50 waypoints)",
            &t,
            &q_t,
            &dq_t,
            &ddq_t,
            &dddq_t,
        )?;

        Ok(())
    }

    fn make_s_vector(n: usize) -> DMatrix<f64> {
        DMatrix::<f64>::from_fn(1, n, |_, j| j as f64 / (n - 1) as f64)
    }

    fn make_parametric_path_6dof() -> Result<super::Path, PathError> {
        super::Path::from_parametric(
            |s: Jet3| {
                vec![
                    sin(s),
                    cos(s),
                    exp(0.3 * s) - 1.0,
                    s + 0.1 * s * s - 0.01 * s * s * s * s,
                    sin(2.0 * s) + 0.15 * cos(3.0 * s),
                    sin(s) * cos(s),
                ]
            },
            0.0,
            1.0,
        )
    }

    fn make_random_waypoints(dim: usize, n_pts: usize) -> DMatrix<f64> {
        let mut rng = rand::rng();
        // Build a random-walk waypoint matrix: each row is one dimension,
        // each column is a waypoint; steps are small bounded increments.
        let mut waypoints = DMatrix::<f64>::zeros(dim, n_pts);
        for mut row in waypoints.row_iter_mut() {
            row[0] = rng.random_range(-1.0..1.0);
            for j in 1..n_pts {
                let step = rng.random_range(-0.5..0.5);
                row[j] = row[j - 1] + step;
            }
        }
        waypoints
    }

    fn setup_robot_with_constraints(
        s: &DMatrix<f64>,
        derivs: &PathDerivatives,
    ) -> Result<Robot<usize>, ConstraintError> {
        let n = s.ncols();
        let mut robot = Robot::with_capacity(DIM, n);

        robot.with_s(&s.as_view())?;
        robot.with_q(
            &derivs.q.as_view(),
            &derivs.dq.as_ref().unwrap().as_view(),
            &derivs.ddq.as_ref().unwrap().as_view(),
            derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
            0,
        )?;

        let vel_max = DMatrix::<f64>::from_element(DIM, n, 1.0);
        let vel_min = DMatrix::<f64>::from_element(DIM, n, -1.0);
        robot.with_axial_velocity(&vel_max.as_view(), &vel_min.as_view(), 0)?;

        let acc_max = DMatrix::<f64>::from_element(DIM, n, 1.0);
        let acc_min = DMatrix::<f64>::from_element(DIM, n, -1.0);
        robot.with_axial_acceleration(&acc_max.as_view(), &acc_min.as_view(), 0)?;

        let jerk_max = DMatrix::<f64>::from_element(DIM, n, 5.0);
        let jerk_min = DMatrix::<f64>::from_element(DIM, n, -5.0);
        robot.with_axial_jerk(&jerk_max.as_view(), &jerk_min.as_view(), 0)?;

        Ok(robot)
    }

    struct ToppTimings {
        topp2_ms: f64,
        topp3_ms: f64,
        t_motion_2: f64,
        t_motion_3: f64,
    }

    fn solve_topp_pipeline(
        robot: &mut Robot<usize>,
        s_slice: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, (usize, usize), ToppTimings), Box<dyn Error>> {
        let n = s_slice.len();

        let topp2_problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;

        let options = ReachSet2OptionsBuilder::new()
            .a_cmp_abs_tol(1e-9)
            .a_cmp_rel_tol(1e-9)
            .lp_feas_tol(1e-9)
            .verbosity(Verbosity::Silent)
            .build()?;

        let start = Instant::now();
        let a_ra = topp2_ra(&topp2_problem, &options)?;
        let topp2_ms = start.elapsed().as_secs_f64() * 1e3;
        let (t_motion_2, _) = s_to_t_topp2(s_slice, &a_ra, 0.0);

        robot.constraints.amax_substitute(&a_ra, 0)?;

        let topp3_problem = Topp3ProblemBuilder::new(robot, 0, &a_ra, (0.0, 0.0), (0.0, 0.0))
            .with_num_stationary_max(2)
            .build_with_linearization()?;
        let options_lp = ClarabelOptionsBuilder::new()
            .allow_almost_solved(true)
            .build()?;

        let start = Instant::now();
        let (a_lp, b_lp, num_stationary) = topp3_lp(&topp3_problem, &options_lp)?;
        let topp3_ms = start.elapsed().as_secs_f64() * 1e3;
        let (t_motion_3, _) = s_to_t_topp3(s_slice, &a_lp, &b_lp, num_stationary, 0.0);

        Ok((
            a_lp,
            b_lp,
            num_stationary,
            ToppTimings {
                topp2_ms,
                topp3_ms,
                t_motion_2,
                t_motion_3,
            },
        ))
    }

    fn interpolate_to_time_domain(
        path: &super::Path,
        s_slice: &[f64],
        a: &[f64],
        b: &[f64],
        num_stationary: (usize, usize),
        dt: f64,
    ) -> Result<
        (
            Vec<f64>,
            DMatrix<f64>,
            DMatrix<f64>,
            DMatrix<f64>,
            DMatrix<f64>,
            f64,
        ),
        Box<dyn Error>,
    > {
        let start = Instant::now();
        let (t_final, t_s) = s_to_t_topp3(s_slice, a, b, num_stationary, 0.0);

        let t_sample: Vec<f64> = (0..=((t_final / dt).floor() as usize))
            .map(|i| i as f64 * dt)
            .collect();

        let s_t = t_to_s_topp3(
            s_slice,
            a,
            b,
            num_stationary,
            &t_s,
            InterpolationMode::NonUniformTimeGrid(&t_sample),
        );

        // Evaluate q(s(t)) – only position needed, derivatives computed via finite diff
        let n_t = t_sample.len();
        let s_t_matrix = DMatrix::<f64>::from_row_slice(1, n_t, &s_t);
        let q_t = path.evaluate_q(s_t_matrix.as_slice())?.q;

        // Compute time-domain derivatives using finite differences
        let mut dq_t = DMatrix::<f64>::zeros(DIM, n_t);
        let mut ddq_t = DMatrix::<f64>::zeros(DIM, n_t);
        let mut dddq_t = DMatrix::<f64>::zeros(DIM, n_t);

        // Compute dq_t, ddq_t, dddq_t via central finite differences (forward/backward at ends).
        finite_diff_inplace(&q_t, &mut dq_t, dt);
        finite_diff_inplace(&dq_t, &mut ddq_t, dt);
        finite_diff_inplace(&ddq_t, &mut dddq_t, dt);

        let interp_ms = start.elapsed().as_secs_f64() * 1e3;

        Ok((t_sample, q_t, dq_t, ddq_t, dddq_t, interp_ms))
    }

    fn plot_topp3_grid(
        filename: &str,
        title: &str,
        t: &[f64],
        q_t: &DMatrix<f64>,
        dq_t: &DMatrix<f64>,
        ddq_t: &DMatrix<f64>,
        dddq_t: &DMatrix<f64>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = Path::new(filename).parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }

        let root = BitMapBackend::new(filename, (2400, 1600)).into_drawing_area();
        root.fill(&WHITE)?;

        let areas = root.split_evenly((4, DIM));
        let mats = [q_t, dq_t, ddq_t, dddq_t];
        let labels = ["q(t)", "dq/dt", "d²q/dt²", "d³q/dt³"];
        let constraints_bounds = [
            None,
            Some((-1.0, 1.0)),
            Some((-1.0, 1.0)),
            Some((-5.0, 5.0)),
        ];

        for row in 0..4 {
            for col in 0..DIM {
                let area = &areas[row * DIM + col];
                let series: Vec<f64> = (0..mats[row].ncols())
                    .map(|j| mats[row][(col, j)])
                    .collect();

                let (mut y_min, mut y_max) = min_max(&series);
                if let Some((lb, ub)) = constraints_bounds[row] {
                    y_min = y_min.min(lb);
                    y_max = y_max.max(ub);
                }
                let pad = 0.1 * (y_max - y_min).max(0.1);
                y_min -= pad;
                y_max += pad;

                let mut chart = ChartBuilder::on(area)
                    .margin(10)
                    .caption(
                        format!("{} axis {}", labels[row], col + 1),
                        ("sans-serif", 16),
                    )
                    .x_label_area_size(30)
                    .y_label_area_size(45)
                    .build_cartesian_2d(t[0]..t[t.len() - 1], y_min..y_max)?;

                chart
                    .configure_mesh()
                    .x_desc(if row == 3 { "t (s)" } else { "" })
                    .draw()?;

                chart.draw_series(LineSeries::new(
                    t.iter().zip(series.iter()).map(|(&x, &y)| (x, y)),
                    &BLACK,
                ))?;

                if let Some((lb, ub)) = constraints_bounds[row] {
                    chart.draw_series(LineSeries::new(
                        vec![(t[0], lb), (t[t.len() - 1], lb)],
                        ShapeStyle::from(&RED.mix(0.5)).stroke_width(2).filled(),
                    ))?;
                    chart.draw_series(LineSeries::new(
                        vec![(t[0], ub), (t[t.len() - 1], ub)],
                        ShapeStyle::from(&RED.mix(0.5)).stroke_width(2).filled(),
                    ))?;
                }
            }
        }

        root.titled(title, ("sans-serif", 32))?;
        root.present()?;
        crate::verbosity_log!(crate::diag::Verbosity::Summary, "Saved plot: {filename}");
        Ok(())
    }

    /// Compute first-order finite differences of `src` into `dst` (in-place).
    /// Uses forward difference at index 0, backward at index n-1, central elsewhere.
    fn finite_diff_inplace(src: &DMatrix<f64>, dst: &mut DMatrix<f64>, h: f64) {
        let n = src.ncols();
        if n < 2 {
            return;
        }
        let inv2h = 1.0 / (2.0 * h);
        // Iterate over each row (dimension) paired between src and dst.
        for (src_row, mut dst_row) in src.row_iter().zip(dst.row_iter_mut()) {
            // Forward difference at left boundary
            dst_row[0] = (src_row[1] - src_row[0]) / h;
            // Backward difference at right boundary
            dst_row[n - 1] = (src_row[n - 1] - src_row[n - 2]) / h;
            // Central differences for all interior points
            for j in 1..n - 1 {
                dst_row[j] = (src_row[j + 1] - src_row[j - 1]) * inv2h;
            }
        }
    }

    fn min_max(data: &[f64]) -> (f64, f64) {
        data.iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            })
    }

    #[test]
    #[ignore = "plotting"]
    fn test_topp3_with_parametric_path() -> Result<(), Box<dyn Error>> {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "\n=== TOPP3 with Parametric Path (6-DOF) ==="
        );

        let start = Instant::now();
        let path = make_parametric_path_6dof()?;
        let build_ms = start.elapsed().as_secs_f64() * 1e3;

        let s = make_s_vector(N);
        let start = Instant::now();
        let derivs = path.evaluate_up_to_3rd(s.as_slice())?;
        let eval_ms = start.elapsed().as_secs_f64() * 1e3;

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[Path Build] {build_ms:.3} ms, [Path Eval] {eval_ms:.3} ms"
        );

        let mut robot = setup_robot_with_constraints(&s, &derivs)?;
        let s_slice: Vec<f64> = (0..N).map(|j| s[(0, j)]).collect();
        let (a, b, num_stat, timings) = solve_topp_pipeline(&mut robot, &s_slice)?;

        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[TOPP2-RA] {:.3} ms, t_motion = {:.3} s",
            timings.topp2_ms,
            timings.t_motion_2
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[TOPP3-LP] {:.3} ms, t_motion = {:.3} s",
            timings.topp3_ms,
            timings.t_motion_3
        );

        let (t, q_t, dq_t, ddq_t, dddq_t, interp_ms) =
            interpolate_to_time_domain(&path, &s_slice, &a, &b, num_stat, 1e-3)?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "[Interpolation] {:.3} ms, {} samples",
            interp_ms,
            t.len()
        );

        plot_topp3_grid(
            "data/path_topp_plots/parametric_path_topp3.png",
            "TOPP3 Result: Parametric Path (6-DOF)",
            &t,
            &q_t,
            &dq_t,
            &ddq_t,
            &dddq_t,
        )?;

        Ok(())
    }
}
