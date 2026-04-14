use copp::InterpolationMode;
use copp::diag::CoppError;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::{ReachSet2OptionsBuilder, Topp2ProblemBuilder, topp2_ra};
use copp::solver::topp3_lp::{
    ClarabelOptionsBuilder, Topp3ProblemBuilder, s_to_t_topp3, t_to_s_topp3, topp3_lp,
};
use std::f64::consts::PI;

fn main() -> Result<(), CoppError> {
    // 1) Deterministic 3-axis Lissajous path q(s), s in [0, 1]
    let path = Path::from_parametric(
        |s: Jet3| {
            vec![
                sin(2.0 * PI * s + 0.0),
                sin(3.0 * PI * s + 0.3),
                sin(5.0 * PI * s + 0.7),
            ]
        },
        0.0,
        1.0,
    )?;

    // `n` is the number of path samples (s_i) to build robot constraints on.
    let n = 1001;
    let s: Vec<f64> = (0..n).map(|j| j as f64 / (n - 1) as f64).collect();
    let derivs = path.evaluate_up_to_3rd(&s)?;

    // 2) Build robot constraints (3-axis), then apply symmetric limits v/a/j = 1
    const DIM: usize = 3;
    let mut robot = Robot::with_capacity(DIM, n);
    robot.with_s(s.as_slice())?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().unwrap().as_view(),
        &derivs.ddq.as_ref().unwrap().as_view(),
        derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
        0,
    )?;
    // The axial velocity is -1 <= v <= 1 for each axis in this example
    let vel_max = vec![1.0; DIM];
    let vel_min = vec![-1.0; DIM];
    robot.with_axial_velocity((vel_max.as_slice(), n), (vel_min.as_slice(), n), 0)?;
    // The axial acceleration is -1 <= a <= 1 for each axis in this example.
    let acc_max = vec![1.0; DIM];
    let acc_min = vec![-1.0; DIM];
    robot.with_axial_acceleration((acc_max.as_slice(), n), (acc_min.as_slice(), n), 0)?;
    // The axial jerk is -1 <= j <= 1 for each axis in this example.
    let jerk_max = vec![1.0; DIM];
    let jerk_min = vec![-1.0; DIM];
    robot.with_axial_jerk((jerk_max.as_slice(), n), (jerk_min.as_slice(), n), 0)?;

    // 3) Solve TOPP2-RA TOPP2-RA to get a feasible linearization profile for TOPP3.
    let idx_s_interval = (0, n - 1); // 0 <= k <= n-1
    let a_boundary = (0.0, 0.0); // a(0) = 0, a(1) = 0
    let a_ra0 = {
        let topp2_problem = Topp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary).build()?;
        let options = ReachSet2OptionsBuilder::new().build()?;
        topp2_ra(&topp2_problem, &options)?
    };

    // 4) Solve TOPP3-LP with the linearization profile from TOPP2-RA.
    // The following code is optional. It substitutes the 1st-order constraint `a[k]<=amax[k]` with `a[k]<=a_ra0[k]` to get a less conservative TOPP3-LP result. The user can skip this step and directly use `amax` for TOPP3-LP.
    robot.constraints.amax_substitute(&a_ra0, 0)?;

    let options_lp = ClarabelOptionsBuilder::new()
        .allow_almost_solved(true)
        .build()?;

    let (a_lp1, b_lp1, num_stationary1) = {
        // Note that in TOPP3Problem, the non-convex jerk constraints should be linearized into a convex one.
        // More details can be found in the documentation of `Topp3ProblemBuilder::build_with_linearization`.
        let topp3_problem =
            Topp3ProblemBuilder::new(&mut robot, idx_s_interval.0, &a_ra0, (0.0, 0.0), (0.0, 0.0))
                .build_with_linearization()?;

        topp3_lp(&topp3_problem, &options_lp)?
    };

    // 5) Post-process TOPP3-LP results: (a,b,s) -> t(s) -> s(t)
    // t_final is the traversal time of the path.
    // t_s[i] is the time at which the path parameter s_i is reached.
    let (t_final1, t_s1) = s_to_t_topp3(&s, &a_lp1, &b_lp1, num_stationary1, 0.0);
    // s_t is a uniform time grid of s(t) with dt = 1e-3s. This is useful for plotting and downstream control.
    let dt = 1e-3;
    let s_t1 = t_to_s_topp3(
        &s,
        &a_lp1,
        &b_lp1,
        num_stationary1,
        &t_s1,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    // 6) Print some results. More detailed results and plots can be achieved by the user.
    // (a_lp1, b_lp1) is a feasible but possibly suboptimal profile for TOPP3. It can be directly used for control or as a reference for further optimization unless a more optimal profile is required.
    println!("TOPP3-LP done. (The first-iteration)");
    println!("dim = {DIM}, N = {n}");
    println!("t_final = {t_final1:.6} s");
    println!("a_profile.len() = {}", a_lp1.len());
    println!("b_profile.len() = {}", b_lp1.len());
    println!("s(t) samples = {}", s_t1.len());

    // 7) Solve TOPP3-LP with the linearization profile from TOPP3-LP.
    let (a_lp2, b_lp2, num_stationary2) = {
        // Note that the linearization point is changed from `a_ra0` to `a_lp1`. This is a standard sequential convex programming (SCP) procedure that can be iterated until convergence. Here we just show the 2nd iteration result.
        let topp3_problem =
            Topp3ProblemBuilder::new(&mut robot, idx_s_interval.0, &a_lp1, (0.0, 0.0), (0.0, 0.0))
                .build_with_linearization()?;

        topp3_lp(&topp3_problem, &options_lp)?
    };

    // 8) Post-process TOPP3-LP results: (a,b,s) -> t(s) -> s(t)
    // t_final is the traversal time of the path.
    // t_s[i] is the time at which the path parameter s_i is reached.
    let (t_final2, t_s2) = s_to_t_topp3(&s, &a_lp2, &b_lp2, num_stationary2, 0.0);
    // s_t is a uniform time grid of s(t) with dt = 1e-3s. This is useful for plotting and downstream control.
    let dt = 1e-3;
    let s_t2 = t_to_s_topp3(
        &s,
        &a_lp2,
        &b_lp2,
        num_stationary2,
        &t_s2,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    // 9) Print some results. More detailed results and plots can be achieved by the user.
    // (a_lp2, b_lp2) is a less conservative and more optimal profile for TOPP3 compared with (a_lp1, b_lp1).
    println!("---------\nTOPP3-LP done. (The second-iteration)");
    println!("dim = {DIM}, N = {n}");
    println!("t_final = {t_final2:.6} s <= {t_final1:.6} s");
    println!("a_profile.len() = {}", a_lp2.len());
    println!("b_profile.len() = {}", b_lp2.len());
    println!("s(t) samples = {}", s_t2.len());

    Ok(())
}
