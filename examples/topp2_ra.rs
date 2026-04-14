use copp::InterpolationMode;
use copp::diag::CoppError;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::{
    ReachSet2OptionsBuilder, Topp2ProblemBuilder, s_to_t_topp2, t_to_s_topp2, topp2_ra,
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
    let derivs = path.evaluate_up_to_2nd(&s)?;

    // 2) Build robot constraints (3-axis), then apply symmetric limits v/a = 1
    const DIM: usize = 3;
    let mut robot = Robot::with_capacity(DIM, n);
    robot.with_s(s.as_slice())?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().unwrap().as_view(),
        &derivs.ddq.as_ref().unwrap().as_view(),
        None,
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

    // 3) Solve TOPP2-RA
    let idx_s_interval = (0, n - 1); // 0 <= k <= n-1
    let a_boundary = (0.0, 0.0); // a(0) = 0, a(1) = 0
    let problem = Topp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary).build()?;
    let options = ReachSet2OptionsBuilder::new().build()?;

    let a_ra = topp2_ra(&problem, &options)?;

    // 4) Post-process TOPP2-RA results: a(s) -> -> t(s) -> s(t)
    // t_final is the traversal time of the path.
    // t_s[i] is the time at which the path parameter s_i is reached.
    let (t_final, t_s) = s_to_t_topp2(&s, &a_ra, 0.0);
    // s_t is a uniform time grid of s(t) with dt = 1e-3s. This is useful for plotting and downstream control.
    let dt = 1e-3;
    let s_t = t_to_s_topp2(
        &s,
        &a_ra,
        &t_s,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    // 5) Print some results. More detailed results and plots can be achieved by the user.
    println!("TOPP2-RA done.");
    println!("dim = {DIM}, N = {n}");
    println!("t_final = {t_final:.6} s");
    println!("a_profile.len() = {}", a_ra.len());
    println!("s(t) samples = {}", s_t.len());

    Ok(())
}
