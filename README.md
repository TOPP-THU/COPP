# COPP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## <font color="#C00000">C</font>onvex-<font color="#C00000">O</font>bjective <font color="#C00000">P</font>ath <font color="#C00000">P</font>arameterization (Rust)

This library targets **Optimal Path Parameterization (OPP)** for robotic trajectory generation. Typical application domains include robotic motion planning and CNC machining.


Given an $n$-dimensional geometric path parameterization

$$
\boldsymbol{q} = \boldsymbol{q}(s),\text{ }s\in[0,s_\text{f}],\text{ (}s_\text{f}\text{ is known)}
$$

the objective is to schedule a dynamically feasible time parameterization

$$
s = s(t),\text{ }t\in[0,t_\text{f}],\text{ (}t_\text{f}\text{ is unknown)}
$$

so that the specified system constraints  are satisfied while an objective $J$ is optimized. In this way, the problem is transformed from geometry-space description to time-space scheduling along a fixed path.

At a high level, this project unifies two system orders and two objective families:

- **2nd-order models**: constraints on velocity, acceleration, torque, etc. The constraint can be written as $\boldsymbol{f}(\boldsymbol{q}(s),\dot{\boldsymbol{q}}(s),\ddot{\boldsymbol{q}}(s);s)\leq\boldsymbol{0}$, and the objective is $\min J=\int_0^{t_\text{f}}L(\boldsymbol{q}(s),\dot{\boldsymbol{q}}(s),\ddot{\boldsymbol{q}}(s);s)\mathrm{d}t$.
- **3rd-order models**: additionally include jerk-related effects. The constraint can be written as $\boldsymbol{f}(\boldsymbol{q}(s),\dot{\boldsymbol{q}}(s),\ddot{\boldsymbol{q}}(s),\dddot{\boldsymbol{q}}(s);s)\leq\boldsymbol{0}$, and the objective is $\min J=\int_0^{t_\text{f}}L(\boldsymbol{q}(s),\dot{\boldsymbol{q}}(s),\ddot{\boldsymbol{q}}(s),\dddot{\boldsymbol{q}}(s);s)\mathrm{d}t$.
- **TOPP** (Time-Optimal Path Parameterization): minimizes traversal time, i.e., $L\equiv1$ and the objective is $J=t_\text{f}$.
- **COPP** (Convex-Objective Path Parameterization): supports broader convex objectives. The objective $L$ should be convex with respect to the state and control: $(\dot{s}^2,\ddot{s})$ in 2nd-order models and $(\dot{s}^2,\ddot{s},\frac{\dddot{s}}{\dot{s}})$ in 3rd-order models.

The resulting taxonomy is summarized below.

| Smoothness order                                     | Time-optimal objective | General convex objective |
| ---------------------------------------------------- | ---------------------- | ------------------------ |
| 2nd-order (velocity/acceleration/torque constraints) | TOPP2                  | COPP2                    |
| 3rd-order (+ jerk constraints)                       | TOPP3                  | COPP3                    |

## Algorithm availability

This section focuses on open-source algorithms. If you need the best possible performance for difficult large-scale problems, please see [Pro](#pro).

| Problem class | Algorithm  | Notes                                                                                                                                                                        |
| ------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TOPP2         | TOPP2-RA   | Ultra-fast reachability-analysis-based method; near-global-optimal in common benchmarks, with relative error typically below $10^{-4}$ versus global optimization baselines. |
| COPP2         | COPP2-SOCP | Solved as an SOCP using `clarabel`; globally optimal under the convex formulation, with moderate-to-high runtime cost.                                                       |
| TOPP3         | TOPP3-SOCP | `clarabel`-based conic formulation; returns KKT solutions with strong optimality quality, and may incur higher computational cost on specific datasets.                      |
| TOPP3         | TOPP3-LP   | Linear-objective approximation of TOPP3-SOCP; usually faster, but can become sub-optimal under tight jerk constraints (recommended mainly when jerk bounds are loose).       |
| COPP3         | COPP3-SOCP | `clarabel`-based conic formulation; returns KKT solutions with strong practical optimality, at relatively high computational cost.                                           |

### Algorithm Selection Guide

| Scenario / primary priority                               | Recommended algorithm                                               | Why this is recommended                                                             | Typical caveat                                         | Alternative                                             |
| --------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| 2nd-order, time-optimal planning with very low runtime    | TOPP2-RA                                                            | Excellent speed-performance trade-off; near-global-optimal in benchmarks            | Objective is fixed to minimum-time style               |                                                         |
| 2nd-order, convex objective with strong global guarantees | COPP2-SOCP                                                          | Convex conic formulation with global optimality under model assumptions             | Higher runtime than RA-style methods                   | See [Pro](#pro) for higher-performance solvers          |
| 3rd-order, best open-source optimality quality            | TOPP3-SOCP (time objective) / COPP3-SOCP (general convex objective) | Strong KKT-quality solutions and broad applicability                                | On specific datasets, computational cost may be higher | TOPP3-LP or higher-performance solvers in [Pro](#pro)   |
| 3rd-order, faster open-source approximation               | TOPP3-LP                                                            | Use this only when your own path-dataset benchmark shows better runtime/performance | May become sub-optimal under tight jerk bounds         | TOPP3-SOCP or higher-performance solvers in [Pro](#pro) |

### Benchmark

The tests are provided in [test_random_spline.rs](tests/test_random_spline.rs). Condition: 

+ `release, --include-ignored`
+ CPU: Intel(R) Core(TM) Ultra 9 285K.
+ Dataset: 100 random 7-DOF spline paths, each discretized into 1000 intervals.

All metrics are listed in the form of "mean ± std".

#### Time-Optimal

| Method     |  Computation time (ms) |   Traversal time (s) |
| ---------- | ---------------------: | -------------------: |
| TOPP2-RA   |    0.665447 ± 0.278833 | 40.903420 ± 1.378671 |
| COPP2-SOCP | 161.544955 ± 13.342861 | 40.900036 ± 1.378611 |
| TOPP3-LP   | 346.354708 ± 38.865584 | 41.422937 ± 1.381852 |
| TOPP3-SOCP | 312.118867 ± 20.544208 | 41.418608 ± 1.381202 |
| COPP3-SOCP | 305.931003 ± 22.910681 | 41.418608 ± 1.381202 |

#### Convex-Objective

In this test, TOPP methods still use traversal time as the optimization objective.

| Method     |  Computation time (ms) |       Objective value |
| ---------- | ---------------------: | --------------------: |
| TOPP2-RA   |    0.696362 ± 0.300599 | 223.896965 ± 9.485003 |
| COPP2-SOCP | 300.428479 ± 71.154448 |  97.746537 ± 2.652869 |
| TOPP3-LP   | 384.399012 ± 79.626532 | 217.858895 ± 9.324830 |
| TOPP3-SOCP | 340.468745 ± 59.209470 | 218.026329 ± 9.285519 |
| COPP3-SOCP | 376.119382 ± 88.865232 |  97.871570 ± 2.645819 |

## Why use COPP instead of re-implementing from scratch?

Unless you are a specialist researcher in TOPP/COPP, we strongly recommend using this library directly. In practical deployment, many critical implementation details are easy to overlook, for example:

+ strict constraint satisfaction rather than soft-constraint relaxation;
+ guaranteed geometric consistency: the executed trajectory $\boldsymbol{q}=\boldsymbol{q}(s(t))$ remains exactly on the original geometric path $\boldsymbol{q}=\boldsymbol{q}(s)$, avoiding additional contour error introduced by the COPP stage;
+ prevention of reverse motion and zero-velocity singularities (i.e., enforcing $\dot{s}>0$ strictly almost everywhere);
+ [certifiable feasibility guarantees, especially for long-path online planning](https://doi.org/10.1016/j.ijmachtools.2025.104355), a challenge only recently addressed in the literature;
+ [boundary acceleration continuity handling](https://doi.org/10.1016/j.ijmachtools.2025.104355), which is frequently problematic in existing methods. Some algorithms bypass this issue by ignoring boundary-acceleration constraints or by disallowing stationary boundary conditions $\dot{s}=0,\ddot{s}=0$, instead requiring $\dot{s}>\delta$.

## Citing

If your work uses the open-source TOPP3/COPP3 functionalities, please cite:

```tex
@article{wang2026online,
  title={Online time-optimal trajectory planning along parametric toolpaths with strict constraint satisfaction and certifiable feasibility guarantee},
  author={Wang, Yunan and Hu, Chuxiong and Li, Yuanshenglong and Yu, Jichuan and Yan, Jizhou and Liang, Yixuan and Jin, Zhao},
  journal={International Journal of Machine Tools and Manufacture},
  volume={215},
  pages={104355},
  year={2026}
}
```

For other use cases, please cite:

```tex
@misc{thu2026copp,
  title = {{C}{O}{P}{P}: Convex-Objective Path Parameterization},
  author = {Wang, Yunan and He, Suqin and Lin, Shize and Hu, Chuxiong},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/TOPP-THU/copp}}
}
```

## Quick Start

### General workflow

+ **Inputs**
    + path grid $0=s_0<s_1<\dots<s_N=s_\text{f}$;
    + path information on the grid (via analytical expressions or sampled values of $\boldsymbol{q}$, $\frac{\mathrm{d}\boldsymbol{q}}{\mathrm{d}s}$, $\frac{\mathrm{d}^2\boldsymbol{q}}{\mathrm{d}s^2}$, and for 3rd-order problems additionally $\frac{\mathrm{d}^3\boldsymbol{q}}{\mathrm{d}s^3}$);
    + constraint forms and bounds, e.g., standard APIs such as `with_axial_velocity`, `with_axial_acceleration`, `with_axial_torque` (plus `with_axial_jerk` for 3rd-order), or more general custom forms. Asymmetric and non-constant limits are supported;
    + for COPP, an objective specification (standard terms or linear combinations, e.g., `Time`, `ThermalEnergy`, `TotalVariationTorque`, `Linear`). In COPP-Pro, user-defined objectives are also supported through trait-based interfaces (algorithm-dependent information such as values and derivatives).
+ **Outputs**
    + the optimal timing map $t=t(s)$, including total traversal time $t_\text{f}$ and grid arrival times $t_k=t(s_k)$;
    + the inverse time parameterization $s=s(t)$, together with interpolation results for user-defined sampling times $\lbrace t_i\rbrace_{i=0}^{I}$, producing $s_i=s(t_i)$ and positions $\boldsymbol{q}(s(t_i))$. These interpolated position/velocity/acceleration references can be streamed to low-level servo controllers in practical deployment.

### Rust

To use the Rust API, add the crate dependency in your Cargo manifest:

```toml
[dependencies]
copp = "*"
```

Complete runnable examples are available in [the examples directory](./examples/). A [quick example](./examples/topp2_ra.rs) is as follows:

```rust
use std::f64::consts::PI;
use copp::InterpolationMode;
use copp::diag::CoppError;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::{
    ReachSet2OptionsBuilder, Topp2ProblemBuilder, s_to_t_topp2, t_to_s_topp2, topp2_ra,
};

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
```

Comprehensive API documentation is also available. The generated docs include mathematical foundations, path/constraint construction methods, logging and output conventions, error definitions, and solver interfaces. You can generate and view the documentation locally with:

```shell
git clone https://github.com/TOPP-THU/copp.git
cd ./copp
cargo doc --no-deps
```

### Other languages

Bindings for other languages are under active development. Planned targets include C, C++, Python, and MATLAB. If you have suggestions for these language interfaces, please feel free to [contact us](#contact-us).

## Pro

### Open-source vs <font color="#C00000">**Pro**</font>

The open-source release and Pro release provide complementary solvers for the above problem classes. Performance evaluations for each method are documented in the corresponding Rust test/example source files and summarized below. For challenging third-order trajectory-planning tasks that require both high solution quality and robust numerical behavior, we recommend the Pro solvers. If you are interested in COPP-Pro licensing or collaboration, please see [Contact Us](#contact-us).

| Problem class | Algorithm  | Availability                         | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------- | ---------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TOPP2         | TOPP2-RA   | Open-source                          | Ultra-fast reachability-analysis-based method; near-global-optimal in common benchmarks, with relative error typically below $10^{-4}$ versus global optimization baselines.                                                                                                                                                                                                                                                                                                                                                                |
| COPP2         | COPP2-SOCP | Open-source                          | Solved as an SOCP using `clarabel`; globally optimal under the convex formulation, with moderate-to-high runtime cost.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| COPP2         | COPP2-ARDP | <font color="#C00000">**Pro**</font> | Ultra-fast original method; globally optimal and substantially faster than COPP2-SOCP.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| TOPP3         | TOPP3-SOCP | Open-source                          | `clarabel`-based conic formulation; returns KKT solutions with strong optimality quality, and may incur higher computational cost on specific datasets.                                                                                                                                                                                                                                                                                                                                                                                     |
| TOPP3         | TOPP3-LP   | Open-source                          | Linear-objective approximation of TOPP3-SOCP; usually faster, but can become sub-optimal under tight jerk constraints (recommended mainly when jerk bounds are loose).                                                                                                                                                                                                                                                                                                                                                                      |
| TOPP3         | TOPP3-RA   | <font color="#C00000">**Pro**</font> | Ultra-fast reachability-analysis-based method; may be sub-optimal under tight jerk constraints (recommended mainly when jerk bounds are loose).                                                                                                                                                                                                                                                                                                                                                                                             |
| COPP3         | COPP3-SOCP | Open-source                          | `clarabel`-based conic formulation; returns KKT solutions with strong practical optimality, at relatively high computational cost.                                                                                                                                                                                                                                                                                                                                                                                                          |
| COPP3         | COPP3-ARDP | <font color="#C00000">**Pro**</font> | Fast original method; returns KKT-quality solutions comparable to TOPP3-SOCP while running substantially faster than TOPP3-SOCP, TOPP3-LP, and COPP3-SOCP. COPP3-ARDP can also be used as a TOPP3 solver, with significantly better time-optimality than TOPP3-RA and TOPP3-LP in many cases. For very long paths, COPP3-ARDP may even exhibit better practical optimality and numerical stability than COPP3-SOCP, since large-scale conic optimization can become limited by convergence behavior and computational-resource constraints. |

### Algorithm Selection Guide

| Scenario / primary priority                                                | Recommended algorithm                                               | Availability                         | Why this is recommended                                                                           | Typical caveat                                         | Alternative                            |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------- |
| 2nd-order, time-optimal planning with very low runtime                     | TOPP2-RA                                                            | Open-source                          | Excellent speed-performance trade-off; near-global-optimal in typical benchmarks                  | Objective is fixed to minimum-time style               |                                        |
| 2nd-order, convex objective with strong global guarantees                  | COPP2-SOCP                                                          | Open-source                          | Convex conic formulation with global optimality under model assumptions                           | Higher runtime than RA/ARDP methods                    | COPP2-ARDP (Pro) for major speed gains |
| 2nd-order, convex objective with maximum efficiency                        | COPP2-ARDP                                                          | <font color="#C00000">**Pro**</font> | Global-optimal quality with substantially improved speed                                          | Pro license required                                   | COPP2-SOCP (Open-source)               |
| 3rd-order, best open-source optimality quality                             | TOPP3-SOCP (time objective) / COPP3-SOCP (general convex objective) | Open-source                          | Strong KKT-quality solutions and broad applicability                                              | On specific datasets, computational cost may be higher | COPP3-ARDP (Pro) for major speed gains |
| 3rd-order, faster open-source approximation                                | TOPP3-LP                                                            | Open-source                          | Use this only when your own path-dataset benchmark shows better runtime/performance               | May become sub-optimal under tight jerk bounds         | COPP3-ARDP (Pro) for major speed gains |
| 3rd-order, ultra-fast RA-style method under loose jerk bounds              | TOPP3-RA                                                            | <font color="#C00000">**Pro**</font> | Very low computational cost                                                                       | Can be sub-optimal when jerk constraints are tight     | COPP3-ARDP or TOPP3-SOCP               |
| 3rd-order, high-quality + high-stability planning for difficult long paths | COPP3-ARDP                                                          | <font color="#C00000">**Pro**</font> | Strong practical optimality with significantly better runtime; often robust on very long horizons | Pro license required                                   | COPP3-SOCP (Open-source)               |

### Benchmark-Pro

All settings are the same as those in [benchmark](#benchmark).

All metrics are listed in the form of "mean ± std".

#### Time-Optimal

| Method                                                  |    Computation time (ms) |       Traversal time (s) |
| ------------------------------------------------------- | -----------------------: | -----------------------: |
| TOPP2-RA                                                |      0.615425 ± 0.244409 |     40.903420 ± 1.378671 |
| COPP2-SOCP                                              |    149.969964 ± 9.364334 |     40.900039 ± 1.378613 |
| <font color="#C00000">**COPP2-ARDP**</font>             |  **5.436142** ± 0.465495 | **40.900135** ± 1.378613 |
| TOPP3-LP                                                |   327.074029 ± 28.893341 |     41.422945 ± 1.381874 |
| TOPP3-SOCP                                              |   289.654071 ± 12.862133 |     41.418608 ± 1.381202 |
| COPP3-SOCP                                              |   285.004302 ± 13.471264 |     41.418608 ± 1.381202 |
| <font color="#C00000">**TOPP3-RA**</font> (Iteration 1) | **10.571045** ± 0.857653 | **41.499200** ± 1.385735 |
| <font color="#C00000">**TOPP3-RA**</font> (Iteration 2) | **20.300932** ± 1.237908 | **41.399867** ± 1.386791 |

#### Convex-Objective, 7-DOF, 100 Random Splines

In this test, TOPP methods still use traversal time as the optimization objective.

| Method                                      |    Computation time (ms) |          Objective value |
| ------------------------------------------- | -----------------------: | -----------------------: |
| TOPP2-RA                                    |      0.534700 ± 0.069296 |   217.444861 ± 12.462360 |
| COPP2-SOCP                                  |   270.059250 ± 52.073677 |     96.517354 ± 3.641154 |
| <font color="#C00000">**COPP2-ARDP**</font> | **12.667700** ± 0.429214 | **96.525785** ± 3.639733 |
| TOPP3-LP                                    |    348.000000 ± 9.326314 |   211.611085 ± 12.367224 |
| TOPP3-SOCP                                  |   301.227000 ± 12.938498 |   211.974066 ± 12.323865 |
| COPP3-SOCP                                  |   301.227000 ± 12.938498 |     96.634962 ± 3.613264 |
| <font color="#C00000">**COPP3-ARDP**</font> | **65.823050** ± 0.087893 | **98.708998** ± 3.354004 |

## Contact Us

For COPP-Pro licensing, commercial collaboration, or technical consulting, please contact:

+ Mr. Yunan Wang: wang-yn22@mails.tsinghua.edu.cn
+ Dr. Suqin He: hsq_thu2012@163.com
+ Dr. Shize Lin: linszthume@gmail.com
+ Prof. Chuxiong Hu: cxhu@tsinghua.edu.cn
