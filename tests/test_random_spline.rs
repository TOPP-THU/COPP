//! Random-spline benchmark tests for TOPP/COPP solvers.
//!
//! This file contains two long-running integration tests:
//! - `test_topp`: time-optimal objective comparison.
//! - `test_copp`: convex-objective comparison (time + thermal energy).
//!
//! Both tests generate deterministic random spline paths/constraints with a fixed seed,
//! then report summary statistics in the form of mean ± std.

use copp::diag::CoppError;
use copp::path::{Path, SplineConfig};
use copp::robot::{Robot, RobotTorque};
use copp::solver::{copp2_socp::*, copp3_socp::*, topp2_ra::*, topp3_lp::*, topp3_socp::*};
use itertools::izip;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

struct TestConfig {
    dim: usize,
    num_waypoints: usize,
    s_len: usize,
    velocity_max: f64,
    acceleration_max: f64,
    jerk_max: f64,
    seed: u64,
    n_exp: usize,
}

/// Generate random spline waypoints in `[-0.5, 0.5]` for each axis.
fn make_random_waypoints(dim: usize, num_waypoints: usize, rng: &mut StdRng) -> DMatrix<f64> {
    let mut waypoints = DMatrix::<f64>::zeros(dim, num_waypoints);

    for i in 0..dim {
        for j in 0..num_waypoints {
            waypoints[(i, j)] = rng.random_range(-0.5..0.5);
        }
    }

    waypoints
}

/// Build a random path and corresponding robot constraints for one experiment.
///
/// The path is sampled on a uniform `s` grid, then velocity/acceleration/jerk bounds
/// are randomized around configured nominal limits.
fn make_random_robot(
    config: &TestConfig,
    rng: &mut StdRng,
) -> Result<(Robot<usize>, Path), CoppError> {
    let waypoints = make_random_waypoints(config.dim, config.num_waypoints, rng);
    let path = Path::from_waypoints(&waypoints, SplineConfig::default())?;

    let s: Vec<f64> = (0..config.s_len)
        .map(|j| j as f64 / (config.s_len - 1) as f64)
        .collect();
    let derivs = path.evaluate_up_to_3rd(&s)?;

    let mut robot = Robot::with_capacity(config.dim, config.s_len);
    robot.with_s(s.as_slice())?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().expect("dq must exist").as_view(),
        &derivs.ddq.as_ref().expect("ddq must exist").as_view(),
        derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
        0,
    )?;

    let vel_max = (0..config.dim)
        .map(|_| config.velocity_max * rng.random_range(0.9..1.1))
        .collect::<Vec<_>>();
    let vel_min = (0..config.dim)
        .map(|_| -config.velocity_max * rng.random_range(0.9..1.1))
        .collect::<Vec<_>>();
    robot.with_axial_velocity(
        (vel_max.as_slice(), config.s_len),
        (vel_min.as_slice(), config.s_len),
        0,
    )?;

    let acc_max = (0..config.dim)
        .map(|_| config.acceleration_max * rng.random_range(0.9..1.1))
        .collect::<Vec<_>>();
    let acc_min = (0..config.dim)
        .map(|_| -config.acceleration_max * rng.random_range(0.9..1.1))
        .collect::<Vec<_>>();
    robot.with_axial_acceleration(
        (acc_max.as_slice(), config.s_len),
        (acc_min.as_slice(), config.s_len),
        0,
    )?;

    let jerk_max = (0..config.dim)
        .map(|_| config.jerk_max * rng.random_range(0.9..1.1))
        .collect::<Vec<_>>();
    let jerk_min = (0..config.dim)
        .map(|_| -config.jerk_max * rng.random_range(0.9..1.1))
        .collect::<Vec<_>>();
    robot.with_axial_jerk(
        (jerk_max.as_slice(), config.s_len),
        (jerk_min.as_slice(), config.s_len),
        0,
    )?;

    Ok((robot, path))
}

/// Compute sample mean and sample standard deviation.
///
/// Uses Bessel's correction (`n - 1`) for the variance denominator.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    (mean, var.sqrt())
}

/// Evaluate 2nd-order thermal energy integral from a solved `a(s)` profile.
///
/// This function reconstructs joint velocity/acceleration over each interval,
/// maps them to torque via inverse dynamics, and integrates squared torque
/// with a trapezoid-like quadratic interpolation term.
fn thermal_energy_2(
    robot: &Robot<usize>,
    path: &Path,
    normalize: &[f64],
    a: &[f64],
) -> Result<f64, CoppError> {
    let s = robot.constraints.s_vec(0, robot.constraints.len())?;
    let t_s = s_to_t_topp2(&s, a, 0.0).1;
    let path_derivs = path.evaluate_up_to_2nd(&s)?;
    let dim = robot.dim();
    let robot_torque = robot.dim();
    let mut tau_left = vec![0.0; dim];
    let mut tau_right = vec![0.0; dim];
    let mut dqt = vec![0.0; dim];
    let mut ddqt = vec![0.0; dim];
    let b = a_to_b_topp2(&s, a);
    let mut energy = 0.0;
    for (k, (t_pair, a_pair, &b)) in izip!(t_s.windows(2), a.windows(2), b.iter()).enumerate() {
        let a_left = a_pair[0];
        let a_right = a_pair[1];
        for (dqt, &dqs) in dqt.iter_mut().zip(
            path_derivs
                .dq
                .as_ref()
                .expect("dq must exist")
                .column(k)
                .iter(),
        ) {
            *dqt = dqs * a_left.sqrt();
        }
        for ((ddqt, &dqs), &ddqs) in ddqt
            .iter_mut()
            .zip(
                path_derivs
                    .dq
                    .as_ref()
                    .expect("dq must exist")
                    .column(k)
                    .iter(),
            )
            .zip(
                path_derivs
                    .ddq
                    .as_ref()
                    .expect("ddq must exist")
                    .column(k)
                    .iter(),
            )
        {
            *ddqt = dqs * b + ddqs * a_left;
        }
        robot_torque.inverse_dynamics(
            &path_derivs.q.column(k).as_slice()[0..dim],
            &dqt,
            &ddqt,
            &mut tau_left,
        );

        for (dqt, &dqs) in dqt.iter_mut().zip(
            path_derivs
                .dq
                .as_ref()
                .expect("dq must exist")
                .column(k + 1)
                .iter(),
        ) {
            *dqt = dqs * a_right.sqrt();
        }
        for ((ddqt, &dqs), &ddqs) in ddqt
            .iter_mut()
            .zip(
                path_derivs
                    .dq
                    .as_ref()
                    .expect("dq must exist")
                    .column(k + 1)
                    .iter(),
            )
            .zip(
                path_derivs
                    .ddq
                    .as_ref()
                    .expect("ddq must exist")
                    .column(k + 1)
                    .iter(),
            )
        {
            *ddqt = dqs * b + ddqs * a_right;
        }
        robot_torque.inverse_dynamics(
            &path_derivs.q.column(k + 1).as_slice()[0..dim],
            &dqt,
            &ddqt,
            &mut tau_right,
        );

        let dt = t_pair[1] - t_pair[0];
        for (tau_left, tau_right, normalize) in izip!(&tau_left, &tau_right, normalize.iter()) {
            energy += (tau_left * tau_left + tau_right * tau_right + tau_left * tau_right) * dt
                / 3.0
                * normalize
                * normalize;
        }
    }
    Ok(energy)
}

/// Evaluate 3rd-order thermal energy integral from solved `a(s), b(s)` profiles.
///
/// Similar to `thermal_energy_2`, but time reconstruction uses 3rd-order mapping
/// and stationary-segment information.
fn thermal_energy_3(
    robot: &Robot<usize>,
    path: &Path,
    normalize: &[f64],
    a: &[f64],
    b: &[f64],
    num_stationary: (usize, usize),
) -> Result<f64, CoppError> {
    let s = robot.constraints.s_vec(0, robot.constraints.len())?;
    let t_s = s_to_t_topp3(&s, a, b, num_stationary, 0.0).1;
    let path_derivs = path.evaluate_up_to_2nd(&s)?;
    let dim = robot.dim();
    let robot_torque = robot.dim();
    let mut tau_left = vec![0.0; dim];
    let mut tau_right = vec![0.0; dim];
    let mut dqt = vec![0.0; dim];
    let mut ddqt = vec![0.0; dim];
    let mut energy = 0.0;
    for (k, (t_pair, a_pair, b_pair)) in
        izip!(t_s.windows(2), a.windows(2), b.windows(2)).enumerate()
    {
        let a_left = a_pair[0];
        let a_right = a_pair[1];
        let b_left = b_pair[0];
        let b_right = b_pair[1];
        for (dqt, &dqs) in dqt.iter_mut().zip(
            path_derivs
                .dq
                .as_ref()
                .expect("dq must exist")
                .column(k)
                .iter(),
        ) {
            *dqt = dqs * a_left.sqrt();
        }
        for ((ddqt, &dqs), &ddqs) in ddqt
            .iter_mut()
            .zip(
                path_derivs
                    .dq
                    .as_ref()
                    .expect("dq must exist")
                    .column(k)
                    .iter(),
            )
            .zip(
                path_derivs
                    .ddq
                    .as_ref()
                    .expect("ddq must exist")
                    .column(k)
                    .iter(),
            )
        {
            *ddqt = dqs * b_left + ddqs * a_left;
        }
        robot_torque.inverse_dynamics(
            &path_derivs.q.column(k).as_slice()[0..dim],
            &dqt,
            &ddqt,
            &mut tau_left,
        );

        for (dqt, &dqs) in dqt.iter_mut().zip(
            path_derivs
                .dq
                .as_ref()
                .expect("dq must exist")
                .column(k + 1)
                .iter(),
        ) {
            *dqt = dqs * a_right.sqrt();
        }
        for ((ddqt, &dqs), &ddqs) in ddqt
            .iter_mut()
            .zip(
                path_derivs
                    .dq
                    .as_ref()
                    .expect("dq must exist")
                    .column(k + 1)
                    .iter(),
            )
            .zip(
                path_derivs
                    .ddq
                    .as_ref()
                    .expect("ddq must exist")
                    .column(k + 1)
                    .iter(),
            )
        {
            *ddqt = dqs * b_right + ddqs * a_right;
        }
        robot_torque.inverse_dynamics(
            &path_derivs.q.column(k + 1).as_slice()[0..dim],
            &dqt,
            &ddqt,
            &mut tau_right,
        );

        let dt = t_pair[1] - t_pair[0];
        for (tau_left, tau_right, normalize) in izip!(&tau_left, &tau_right, normalize.iter()) {
            energy += (tau_left * tau_left + tau_right * tau_right + tau_left * tau_right) * dt
                / 3.0
                * normalize
                * normalize;
        }
    }
    Ok(energy)
}

mod tests {
    use super::*;
    use copp::solver::copp2_socp::copp2_socp;

    /// Conditions: release, --include-ignored, CPU = Intel(R) Core(TM) Ultra 9 285K.
    /// ==== Summary over 100 experiments ====
    /// Tc(ms) mean/std:
    ///   TOPP2-RA   : mean = 0.665447, std = 0.278833
    ///   COPP2-SOCP : mean = 161.544955, std = 13.342861
    ///   TOPP3-LP   : mean = 346.354708, std = 38.865584
    ///   TOPP3-SOCP : mean = 312.118867, std = 20.544208
    ///   COPP3-SOCP : mean = 305.931003, std = 22.910681
    /// Tf(s) mean/std:
    ///   TOPP2-RA   : mean = 40.903420, std = 1.378671
    ///   COPP2-SOCP : mean = 40.900036, std = 1.378611
    ///   TOPP3-LP   : mean = 41.422937, std = 1.381852
    ///   TOPP3-SOCP : mean = 41.418608, std = 1.381202
    ///   COPP3-SOCP : mean = 41.418608, std = 1.381202
    ///
    /// Compares runtime (`Tc`) and final traversal time (`Tf`) under a pure
    /// time-optimal objective.
    #[test]
    #[ignore = "slow"]
    fn test_topp() -> Result<(), CoppError> {
        let config = TestConfig {
            dim: 7,
            num_waypoints: 50,
            s_len: 1001,
            velocity_max: 1.0,
            acceleration_max: 5.0,
            jerk_max: 50.0,
            seed: 20260414,
            n_exp: 100,
        };
        let objectives = vec![CoppObjective::Time(1.0)];

        let mut rng = StdRng::seed_from_u64(config.seed);

        let idx_s_interval = (0, config.s_len - 1);
        let a_boundary = (0.0, 0.0);
        let b_boundary = (0.0, 0.0);
        let idx_s_start = 0;

        let mut tc_topp2_ra_vec = Vec::with_capacity(config.n_exp);
        let mut tc_copp2_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tc_topp3_lp_vec = Vec::with_capacity(config.n_exp);
        let mut tc_topp3_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tc_copp3_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tf_topp2_ra_vec = Vec::with_capacity(config.n_exp);
        let mut tf_copp2_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tf_topp3_lp_vec = Vec::with_capacity(config.n_exp);
        let mut tf_topp3_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tf_copp3_socp_vec = Vec::with_capacity(config.n_exp);

        for i_exp in 0..config.n_exp {
            let (mut robot, _) = make_random_robot(&config, &mut rng)?;
            let s = robot.constraints.s_vec(0, robot.constraints.len())?;

            // Test TOPP2-RA
            let start = Instant::now();
            let a_topp2_ra = {
                let problem =
                    Topp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary).build()?;
                let options = ReachSet2OptionsBuilder::new().build()?;
                topp2_ra(&problem, &options)?
            };
            let tc_topp2_ra = start.elapsed();
            let tf_topp2_ra = s_to_t_topp2(&s, &a_topp2_ra, 0.0).0;

            // Test COPP2-SOCP
            let start = Instant::now();
            let a_copp2_socp = {
                let problem =
                    Copp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary, &objectives)
                        .build()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .build()?;
                copp2_socp(&problem, &options)?
            };
            let tc_copp2_socp = start.elapsed();
            let tf_copp2_socp = s_to_t_topp2(&s, &a_copp2_socp, 0.0).0;

            // Test TOPP3-LP
            let start = Instant::now();
            let (a_topp3_lp, b_topp3_lp, num_stat_topp3_lp) = {
                let problem = Topp3ProblemBuilder::new(
                    &mut robot,
                    idx_s_start,
                    &a_topp2_ra,
                    a_boundary,
                    b_boundary,
                )
                .build_with_linearization()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .allow_insufficient_progress(true)
                    .build()?;
                topp3_lp(&problem, &options)?
            };
            let tc_topp3_lp = tc_topp2_ra + start.elapsed();
            let tf_topp3_lp = s_to_t_topp3(&s, &a_topp3_lp, &b_topp3_lp, num_stat_topp3_lp, 0.0).0;

            // Test TOPP3-SOCP
            let start = Instant::now();
            let (a_topp3_socp, b_topp3_socp, num_stat_topp3_socp) = {
                let problem = Topp3ProblemBuilder::new(
                    &mut robot,
                    idx_s_start,
                    &a_topp2_ra,
                    a_boundary,
                    b_boundary,
                )
                .build_with_linearization()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .allow_insufficient_progress(true)
                    .build()?;
                topp3_socp(&problem, &options)?
            };
            let tc_topp3_socp = tc_topp2_ra + start.elapsed();
            let tf_topp3_socp =
                s_to_t_topp3(&s, &a_topp3_socp, &b_topp3_socp, num_stat_topp3_socp, 0.0).0;

            // Test COPP3-SOCP
            let start = Instant::now();
            let (a_copp3_socp, b_copp3_socp, num_stat_copp3_socp) = {
                let problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives,
                    idx_s_start,
                    &a_topp2_ra,
                    a_boundary,
                    b_boundary,
                )
                .build_with_linearization()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .allow_insufficient_progress(true)
                    .build()?;
                copp3_socp(&problem, &options)?
            };
            let tc_copp3_socp = tc_topp2_ra + start.elapsed();
            let tf_copp3_socp =
                s_to_t_topp3(&s, &a_copp3_socp, &b_copp3_socp, num_stat_copp3_socp, 0.0).0;

            println!("Experiment {}/{}:", i_exp + 1, config.n_exp);
            println!(
                "Tc: TOPP2-RA = {tc_topp2_ra:?}, COPP2-SOCP = {tc_copp2_socp:?}, TOPP3-LP = {tc_topp3_lp:?}, TOPP3-SOCP = {tc_topp3_socp:?}, COPP3-SOCP = {tc_copp3_socp:?}",
            );
            println!(
                "Tf: TOPP2-RA = {tf_topp2_ra:.4}, COPP2-SOCP = {tf_copp2_socp:.4}, TOPP3-LP = {tf_topp3_lp:.4}, TOPP3-SOCP = {tf_topp3_socp:.4}, COPP3-SOCP = {tf_copp3_socp:.4}",
            );

            tc_topp2_ra_vec.push(tc_topp2_ra.as_secs_f64() * 1E3);
            tc_copp2_socp_vec.push(tc_copp2_socp.as_secs_f64() * 1E3);
            tc_topp3_lp_vec.push(tc_topp3_lp.as_secs_f64() * 1E3);
            tc_topp3_socp_vec.push(tc_topp3_socp.as_secs_f64() * 1E3);
            tc_copp3_socp_vec.push(tc_copp3_socp.as_secs_f64() * 1E3);
            tf_topp2_ra_vec.push(tf_topp2_ra);
            tf_copp2_socp_vec.push(tf_copp2_socp);
            tf_topp3_lp_vec.push(tf_topp3_lp);
            tf_topp3_socp_vec.push(tf_topp3_socp);
            tf_copp3_socp_vec.push(tf_copp3_socp);
        }

        let (tc_topp2_ra_mean, tc_topp2_ra_std) = mean_std(&tc_topp2_ra_vec);
        let (tc_copp2_socp_mean, tc_copp2_socp_std) = mean_std(&tc_copp2_socp_vec);
        let (tc_topp3_lp_mean, tc_topp3_lp_std) = mean_std(&tc_topp3_lp_vec);
        let (tc_topp3_socp_mean, tc_topp3_socp_std) = mean_std(&tc_topp3_socp_vec);
        let (tc_copp3_socp_mean, tc_copp3_socp_std) = mean_std(&tc_copp3_socp_vec);

        let (tf_topp2_ra_mean, tf_topp2_ra_std) = mean_std(&tf_topp2_ra_vec);
        let (tf_copp2_socp_mean, tf_copp2_socp_std) = mean_std(&tf_copp2_socp_vec);
        let (tf_topp3_lp_mean, tf_topp3_lp_std) = mean_std(&tf_topp3_lp_vec);
        let (tf_topp3_socp_mean, tf_topp3_socp_std) = mean_std(&tf_topp3_socp_vec);
        let (tf_copp3_socp_mean, tf_copp3_socp_std) = mean_std(&tf_copp3_socp_vec);

        println!("\n==== Summary over {} experiments ====", config.n_exp);
        println!("Tc(ms) mean/std:");
        println!("  TOPP2-RA   : mean = {tc_topp2_ra_mean:.6}, std = {tc_topp2_ra_std:.6}");
        println!("  COPP2-SOCP : mean = {tc_copp2_socp_mean:.6}, std = {tc_copp2_socp_std:.6}");
        println!("  TOPP3-LP   : mean = {tc_topp3_lp_mean:.6}, std = {tc_topp3_lp_std:.6}");
        println!("  TOPP3-SOCP : mean = {tc_topp3_socp_mean:.6}, std = {tc_topp3_socp_std:.6}");
        println!("  COPP3-SOCP : mean = {tc_copp3_socp_mean:.6}, std = {tc_copp3_socp_std:.6}");

        println!("Tf(s) mean/std:");
        println!("  TOPP2-RA   : mean = {tf_topp2_ra_mean:.6}, std = {tf_topp2_ra_std:.6}");
        println!("  COPP2-SOCP : mean = {tf_copp2_socp_mean:.6}, std = {tf_copp2_socp_std:.6}");
        println!("  TOPP3-LP   : mean = {tf_topp3_lp_mean:.6}, std = {tf_topp3_lp_std:.6}");
        println!("  TOPP3-SOCP : mean = {tf_topp3_socp_mean:.6}, std = {tf_topp3_socp_std:.6}");
        println!("  COPP3-SOCP : mean = {tf_copp3_socp_mean:.6}, std = {tf_copp3_socp_std:.6}");

        Ok(())
    }

    /// ==== Summary over 100 experiments ====
    /// Tc(ms) mean/std:
    ///   TOPP2-RA   : mean = 0.696362, std = 0.300599
    ///   COPP2-SOCP : mean = 300.428479, std = 71.154448
    ///   TOPP3-LP   : mean = 384.399012, std = 79.626532
    ///   TOPP3-SOCP : mean = 340.468745, std = 59.209470
    ///   COPP3-SOCP : mean = 376.119382, std = 88.865232
    /// Obj mean/std:
    ///   TOPP2-RA   : mean = 223.896965, std = 9.485003
    ///   COPP2-SOCP : mean = 97.746537, std = 2.652869
    ///   TOPP3-LP   : mean = 217.858895, std = 9.324830
    ///   TOPP3-SOCP : mean = 218.026329, std = 9.285519
    ///   COPP3-SOCP : mean = 97.871570, std = 2.645819
    ///
    /// Uses convex objective `Obj = Time + weight_energy * ThermalEnergy`.
    /// TOPP methods in this test still optimize time directly; the reported `Obj`
    /// is computed during post-evaluation for comparison.
    #[test]
    #[ignore = "slow"]
    fn test_copp() -> Result<(), CoppError> {
        let config = TestConfig {
            dim: 7,
            num_waypoints: 50,
            s_len: 1001,
            velocity_max: 1.0,
            acceleration_max: 5.0,
            jerk_max: 50.0,
            seed: 20260414,
            n_exp: 100,
        };
        let normalize = vec![1.0; config.dim];
        let weight_energy = 0.2;
        let objectives = vec![
            CoppObjective::Time(1.0),
            CoppObjective::ThermalEnergy(weight_energy, &normalize),
        ];

        let mut rng = StdRng::seed_from_u64(config.seed);

        let idx_s_interval = (0, config.s_len - 1);
        let a_boundary = (0.0, 0.0);
        let b_boundary = (0.0, 0.0);
        let idx_s_start = 0;

        let mut tc_topp2_ra_vec = Vec::with_capacity(config.n_exp);
        let mut tc_copp2_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tc_topp3_lp_vec = Vec::with_capacity(config.n_exp);
        let mut tc_topp3_socp_vec = Vec::with_capacity(config.n_exp);
        let mut tc_copp3_socp_vec = Vec::with_capacity(config.n_exp);
        let mut obj_topp2_ra_vec = Vec::with_capacity(config.n_exp);
        let mut obj_copp2_socp_vec = Vec::with_capacity(config.n_exp);
        let mut obj_topp3_lp_vec = Vec::with_capacity(config.n_exp);
        let mut obj_topp3_socp_vec = Vec::with_capacity(config.n_exp);
        let mut obj_copp3_socp_vec = Vec::with_capacity(config.n_exp);

        for i_exp in 0..config.n_exp {
            let (mut robot, path) = make_random_robot(&config, &mut rng)?;
            let s = robot.constraints.s_vec(0, robot.constraints.len())?;

            // Test TOPP2-RA
            let start = Instant::now();
            let a_topp2_ra = {
                let problem =
                    Topp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary).build()?;
                let options = ReachSet2OptionsBuilder::new().build()?;
                topp2_ra(&problem, &options)?
            };
            let tc_topp2_ra = start.elapsed();
            let tf_topp2_ra = s_to_t_topp2(&s, &a_topp2_ra, 0.0).0;
            let energy_topp2_ra = thermal_energy_2(&robot, &path, &normalize, &a_topp2_ra)?;
            let obj_topp2_ra = tf_topp2_ra + weight_energy * energy_topp2_ra;

            // Test COPP2-SOCP
            let start = Instant::now();
            let a_copp2_socp = {
                let problem =
                    Copp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary, &objectives)
                        .build()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .build()?;
                copp2_socp(&problem, &options)?
            };
            let tc_copp2_socp = start.elapsed();
            let tf_copp2_socp = s_to_t_topp2(&s, &a_copp2_socp, 0.0).0;
            let energy_copp2_socp = thermal_energy_2(&robot, &path, &normalize, &a_copp2_socp)?;
            let obj_copp2_socp = tf_copp2_socp + weight_energy * energy_copp2_socp;

            // Test TOPP3-LP
            let start = Instant::now();
            let (a_topp3_lp, b_topp3_lp, num_stat_topp3_lp) = {
                let problem = Topp3ProblemBuilder::new(
                    &mut robot,
                    idx_s_start,
                    &a_topp2_ra,
                    a_boundary,
                    b_boundary,
                )
                .build_with_linearization()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .allow_insufficient_progress(true)
                    .build()?;
                topp3_lp(&problem, &options)?
            };
            let tc_topp3_lp = tc_topp2_ra + start.elapsed();
            let tf_topp3_lp = s_to_t_topp3(&s, &a_topp3_lp, &b_topp3_lp, num_stat_topp3_lp, 0.0).0;
            let energy_topp3_lp = thermal_energy_3(
                &robot,
                &path,
                &normalize,
                &a_topp3_lp,
                &b_topp3_lp,
                num_stat_topp3_lp,
            )?;
            let obj_topp3_lp = tf_topp3_lp + weight_energy * energy_topp3_lp;

            // Test TOPP3-SOCP
            let start = Instant::now();
            let (a_topp3_socp, b_topp3_socp, num_stat_topp3_socp) = {
                let problem = Topp3ProblemBuilder::new(
                    &mut robot,
                    idx_s_start,
                    &a_topp2_ra,
                    a_boundary,
                    b_boundary,
                )
                .build_with_linearization()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .allow_insufficient_progress(true)
                    .build()?;
                topp3_socp(&problem, &options)?
            };
            let tc_topp3_socp = tc_topp2_ra + start.elapsed();
            let tf_topp3_socp =
                s_to_t_topp3(&s, &a_topp3_socp, &b_topp3_socp, num_stat_topp3_socp, 0.0).0;
            let energy_topp3_socp = thermal_energy_3(
                &robot,
                &path,
                &normalize,
                &a_topp3_socp,
                &b_topp3_socp,
                num_stat_topp3_socp,
            )?;
            let obj_topp3_socp = tf_topp3_socp + weight_energy * energy_topp3_socp;

            // Test COPP3-SOCP
            let start = Instant::now();
            let (a_copp3_socp, b_copp3_socp, num_stat_copp3_socp) = {
                let problem = Copp3ProblemBuilder::new(
                    &mut robot,
                    &objectives,
                    idx_s_start,
                    &a_topp2_ra,
                    a_boundary,
                    b_boundary,
                )
                .build_with_linearization()?;
                let options = ClarabelOptionsBuilder::new()
                    .allow_almost_solved(true)
                    .allow_max_time(true)
                    .allow_max_iterations(true)
                    .allow_insufficient_progress(true)
                    .build()?;
                copp3_socp(&problem, &options)?
            };
            let tc_copp3_socp = tc_topp2_ra + start.elapsed();
            let tf_copp3_socp =
                s_to_t_topp3(&s, &a_copp3_socp, &b_copp3_socp, num_stat_copp3_socp, 0.0).0;
            let energy_copp3_socp = thermal_energy_3(
                &robot,
                &path,
                &normalize,
                &a_copp3_socp,
                &b_copp3_socp,
                num_stat_copp3_socp,
            )?;
            let obj_copp3_socp = tf_copp3_socp + weight_energy * energy_copp3_socp;

            println!("Experiment {}/{}:", i_exp + 1, config.n_exp);
            println!(
                "Tc: TOPP2-RA = {tc_topp2_ra:?}, COPP2-SOCP = {tc_copp2_socp:?}, TOPP3-LP = {tc_topp3_lp:?}, TOPP3-SOCP = {tc_topp3_socp:?}, COPP3-SOCP = {tc_copp3_socp:?}",
            );
            println!(
                "Obj: TOPP2-RA = {obj_topp2_ra:.4}, COPP2-SOCP = {obj_copp2_socp:.4}, TOPP3-LP = {obj_topp3_lp:.4}, TOPP3-SOCP = {obj_topp3_socp:.4}, COPP3-SOCP = {obj_copp3_socp:.4}",
            );

            tc_topp2_ra_vec.push(tc_topp2_ra.as_secs_f64() * 1E3);
            tc_copp2_socp_vec.push(tc_copp2_socp.as_secs_f64() * 1E3);
            tc_topp3_lp_vec.push(tc_topp3_lp.as_secs_f64() * 1E3);
            tc_topp3_socp_vec.push(tc_topp3_socp.as_secs_f64() * 1E3);
            tc_copp3_socp_vec.push(tc_copp3_socp.as_secs_f64() * 1E3);
            obj_topp2_ra_vec.push(obj_topp2_ra);
            obj_copp2_socp_vec.push(obj_copp2_socp);
            obj_topp3_lp_vec.push(obj_topp3_lp);
            obj_topp3_socp_vec.push(obj_topp3_socp);
            obj_copp3_socp_vec.push(obj_copp3_socp);
        }

        let (tc_topp2_ra_mean, tc_topp2_ra_std) = mean_std(&tc_topp2_ra_vec);
        let (tc_copp2_socp_mean, tc_copp2_socp_std) = mean_std(&tc_copp2_socp_vec);
        let (tc_topp3_lp_mean, tc_topp3_lp_std) = mean_std(&tc_topp3_lp_vec);
        let (tc_topp3_socp_mean, tc_topp3_socp_std) = mean_std(&tc_topp3_socp_vec);
        let (tc_copp3_socp_mean, tc_copp3_socp_std) = mean_std(&tc_copp3_socp_vec);

        let (obj_topp2_ra_mean, obj_topp2_ra_std) = mean_std(&obj_topp2_ra_vec);
        let (obj_copp2_socp_mean, obj_copp2_socp_std) = mean_std(&obj_copp2_socp_vec);
        let (obj_topp3_lp_mean, obj_topp3_lp_std) = mean_std(&obj_topp3_lp_vec);
        let (obj_topp3_socp_mean, obj_topp3_socp_std) = mean_std(&obj_topp3_socp_vec);
        let (obj_copp3_socp_mean, obj_copp3_socp_std) = mean_std(&obj_copp3_socp_vec);

        println!("\n==== Summary over {} experiments ====", config.n_exp);
        println!("Tc(ms) mean/std:");
        println!("  TOPP2-RA   : mean = {tc_topp2_ra_mean:.6}, std = {tc_topp2_ra_std:.6}");
        println!("  COPP2-SOCP : mean = {tc_copp2_socp_mean:.6}, std = {tc_copp2_socp_std:.6}");
        println!("  TOPP3-LP   : mean = {tc_topp3_lp_mean:.6}, std = {tc_topp3_lp_std:.6}");
        println!("  TOPP3-SOCP : mean = {tc_topp3_socp_mean:.6}, std = {tc_topp3_socp_std:.6}");
        println!("  COPP3-SOCP : mean = {tc_copp3_socp_mean:.6}, std = {tc_copp3_socp_std:.6}");

        println!("Obj mean/std:");
        println!("  TOPP2-RA   : mean = {obj_topp2_ra_mean:.6}, std = {obj_topp2_ra_std:.6}");
        println!("  COPP2-SOCP : mean = {obj_copp2_socp_mean:.6}, std = {obj_copp2_socp_std:.6}");
        println!("  TOPP3-LP   : mean = {obj_topp3_lp_mean:.6}, std = {obj_topp3_lp_std:.6}");
        println!("  TOPP3-SOCP : mean = {obj_topp3_socp_mean:.6}, std = {obj_topp3_socp_std:.6}");
        println!("  COPP3-SOCP : mean = {obj_copp3_socp_mean:.6}, std = {obj_copp3_socp_std:.6}");

        Ok(())
    }
}
