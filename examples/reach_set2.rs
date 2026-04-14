use copp::diag::CoppError;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::reach_set2::{
    ReachSet2OptionsBuilder, Topp2ProblemBuilder, reach_set2_backward, reach_set2_bidirectional,
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

    // 3) Build TOPP2 problem and compute reachable sets
    let idx_s_interval = (0, n - 1); // 0 <= k <= n-1
    let a_boundary = (0.0, 0.0); // a(0) = 0, a(1) = 0
    let problem = Topp2ProblemBuilder::new(&robot, idx_s_interval, a_boundary).build()?;
    let options = ReachSet2OptionsBuilder::new().build()?;

    // backward-only reachability (terminal boundary constrained)
    let reach_back = reach_set2_backward(&problem, &options)?;
    // bidirectional reachability (both start/terminal boundaries constrained)
    let reach_bidir = reach_set2_bidirectional(&problem, &options)?;

    // 4) Print some results. More detailed results and plots can be achieved by the user.
    println!("reach_set2 done.");
    println!("dim = {DIM}, N = {n}");
    println!(
        "backward-only: a_max.len() = {}, a_min.len() = {}",
        reach_back.a_max.len(),
        reach_back.a_min.len()
    );
    println!(
        "bidirectional: a_max.len() = {}, a_min.len() = {}",
        reach_bidir.a_max.len(),
        reach_bidir.a_min.len()
    );

    let k0 = 0;
    let km = n / 2;
    let k1 = n - 1;
    println!(
        "bidirectional bounds @k=0/mid/end: [{:.6}, {:.6}], [{:.6}, {:.6}], [{:.6}, {:.6}]",
        reach_bidir.a_min[k0],
        reach_bidir.a_max[k0],
        reach_bidir.a_min[km],
        reach_bidir.a_max[km],
        reach_bidir.a_min[k1],
        reach_bidir.a_max[k1],
    );

    Ok(())
}
