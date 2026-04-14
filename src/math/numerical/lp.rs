//! Incremental linear-programming kernels in 1D/2D.
//!
//! # Method identity
//! The functions in this module are low-level numeric engines used by TOPP/COPP
//! planners. They operate on half-space form constraints and provide:
//! - direct 1D/2D incremental LP solves,
//! - warm-start interfaces,
//! - helper normalization and tiny linear-system primitives.

use crate::math::numerical::cross_product_2d;
use core::f64;

/// Safety multiplier used when propagating tolerance across dimension reductions.
const EPS_SCALE: f64 = 10.0;
/// Near-zero threshold for feasibility/normalization branch decisions.
pub(crate) const EPS_ZERO: f64 = 1e-9;
/// Lower bound for coefficient normalization denominators.
const EPS_NORMALIZE: f64 = 1e-3;
/// Default box bound used to stabilize unbounded LP directions.
pub(crate) const LP_BOUND: f64 = 1e6;

/// LP tolerance bundle for numerical-robustness controls.
///
/// This mirrors the options naming style used in `src/copp` modules
/// (`*Options` + `*OptionsBuilder`) so tolerance policies are explicit and
/// configurable.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LpToleranceOptions {
    /// Base feasibility tolerance used by LP inequality checks.
    pub(crate) feas_tol: f64,
    /// Lower bound used in row normalization to avoid aggressive scaling.
    /// Tolerance scale factor used when reducing dimension (3D->2D->1D).
    pub(crate) reduce_dim_scale: f64,
}

/// Builder for [`LpToleranceOptions`].
pub(crate) struct LpToleranceOptionsBuilder {
    pub feas_tol: f64,
    pub reduce_dim_scale: f64,
}

impl Default for LpToleranceOptionsBuilder {
    fn default() -> Self {
        Self {
            feas_tol: EPS_ZERO,
            reduce_dim_scale: EPS_SCALE,
        }
    }
}

impl LpToleranceOptionsBuilder {
    #[inline(always)]
    pub(crate) fn builder() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub(crate) fn feas_tol(mut self, feas_tol: f64) -> Self {
        self.feas_tol = feas_tol;
        self
    }

    #[inline(always)]
    pub(crate) fn reduce_dim_scale(mut self, reduce_dim_scale: f64) -> Self {
        self.reduce_dim_scale = reduce_dim_scale;
        self
    }

    #[inline(always)]
    pub(crate) fn build(self) -> LpToleranceOptions {
        LpToleranceOptions {
            feas_tol: self.feas_tol,
            reduce_dim_scale: self.reduce_dim_scale,
        }
    }
}

impl LpToleranceOptions {
    #[inline(always)]
    pub(crate) fn with_feas_tol(feas_tol: f64) -> Self {
        LpToleranceOptionsBuilder::builder()
            .feas_tol(feas_tol)
            .reduce_dim_scale(EPS_SCALE)
            .build()
    }
}

/// Linear programming in 2D plane of extreme direction.  
/// max w.0*X+w.1*Y, s.t. Ab[i].0*X+Ab[i].1*Y<=Ab[i].2  
/// Given a x_opt=(x_opt,y_opt), determine the tangent direction v1, v2 on the 2D plane (x, y).  
/// The cone defined by the **anticlockwise** transition from $v_1$ to $v_2$ contains **feasible** directions.
#[cfg(test)]
pub(crate) fn lp_2d_extreme_direction(
    a_b: &[(f64, f64, f64)],
    x_opt: (f64, f64),
    w: (f64, f64),
    tol: &LpToleranceOptions,
) -> ((f64, f64), (f64, f64)) {
    let epsilon = tol.feas_tol;
    let a_b_act = a_b
        .iter()
        .filter_map(|&(a, b, c)| {
            let norm = (a * a + b * b).sqrt();
            if norm > EPS_ZERO && (a * x_opt.0 + b * x_opt.1 - c).abs() < epsilon * norm {
                Some((a / norm, b / norm))
            } else {
                None
            }
        })
        .collect::<Vec<(f64, f64)>>();
    if a_b_act.len() < 2 {
        return ((-w.1, w.0), (w.1, -w.0));
    }
    let w_norm = (w.0 * w.0 + w.1 * w.1).sqrt();
    let v0 = (-w.1 / w_norm, w.0 / w_norm); // 90 degree anticlockwise rotation
    let mut cos_max: f64 = 1.0;
    let mut cos_min: f64 = -1.0;
    let mut v_max = v0;
    let mut v_min = (-v0.0, -v0.1);
    let mut func = |v: (f64, f64), upper_bound: bool| {
        if cross_product_2d(v0, v) > 0.0 {
            // anticlockwise
            let cos_now = (v.0 * v0.0 + v.1 * v0.1) / (v.0 * v.0 + v.1 * v.1).sqrt();
            if upper_bound {
                if cos_now < cos_max {
                    cos_max = cos_now;
                    v_max = v;
                }
            } else if cos_now > cos_min {
                cos_min = cos_now;
                v_min = v;
            }
        }
    };
    for &(a1, b1) in a_b_act.iter() {
        func((-b1, a1), true);
        func((b1, -a1), false);
    }
    let vmax_norm = (v_max.0 * v_max.0 + v_max.1 * v_max.1).sqrt();
    let vmin_norm = (v_min.0 * v_min.0 + v_min.1 * v_min.1).sqrt();

    (
        (v_max.0 / vmax_norm, v_max.1 / vmax_norm),
        (v_min.0 / vmin_norm, v_min.1 / vmin_norm),
    )
}

/// Linear programming in 2D plane based on geometric method.  
/// max J:=w_x*X+w_y*Y, s.t. Ab[i].0*X+Ab[i].1*Y<=Ab[i].2  
/// return (X,Y,J)
#[cfg(test)]
pub(crate) fn lp_2d_incre<W: WarmStartLp2d, const NORMALIZE: bool>(
    a_b: &[(f64, f64, f64)],
    w: (f64, f64),
    warm_start: &W,
    tol: &LpToleranceOptions,
    buffer: &mut Vec<(f64, f64, f64)>,
) -> (f64, f64, f64) {
    let (w_x, w_y) = w;
    if w_x.abs() < EPS_ZERO && w_y.abs() < EPS_ZERO {
        return (f64::NAN, f64::NAN, 0.0);
    }
    let w = (w_x * w_x + w_y * w_y).sqrt();
    let w_inv = 1.0 / w;
    let w_x = w_x * w_inv;
    let w_y = w_y * w_inv;

    buffer.clear();
    buffer.extend(
        a_b.iter()
            .map(|&(ax, ay, b)| (ax * w_y - ay * w_x, ax * w_x + ay * w_y, b)),
    );
    let (x, y) = lp_2d_incre_max_y::<_, NORMALIZE>(buffer, &warm_start.rotate((w_x, w_y)), tol);

    (w_y * x + w_x * y, w_y * y - w_x * x, w * y)
}

#[inline(always)]
/// Normalize 2D half-space rows by normal-vector magnitude.
///
/// Degenerate rows are mapped to zeros.
pub(crate) fn normalize_lp2d(a_b: &mut [(f64, f64, f64)]) {
    for w in a_b.iter_mut() {
        let norm = (w.0 * w.0 + w.1 * w.1).sqrt();
        *w = if norm < EPS_ZERO {
            (0.0, 0.0, 0.0)
        } else {
            let norm_inv = 1.0 / norm.max(EPS_NORMALIZE);
            (w.0 * norm_inv, w.1 * norm_inv, w.2 * norm_inv)
        };
    }
}

/// Linear programming in 2D plane based on incremental method (only maximize Y).  
/// max Y, s.t. Ab[i].0*X+Ab[i].1*Y<=Ab[i].2
/// return (X,Y)
#[inline(always)]
fn lp_2d_incre_max_y_core<C: Lp2dIncCollector, W: WarmStartLp2d, const NORMALIZE: bool>(
    a_b: &mut [(f64, f64, f64)],
    warm_start: &W,
    tol: &LpToleranceOptions,
    mut collector: C,
) -> (f64, f64) {
    let epsilon = tol.feas_tol;
    if a_b.len() <= 1 {
        if a_b.len() == 1 {
            let a_b_0 = a_b.first().unwrap();
            if a_b_0.0.abs() < EPS_ZERO && a_b_0.1 > EPS_ZERO {
                // Unbounded
                return (f64::NAN, a_b_0.2 / a_b_0.1);
            }
        }
        // Infeasible
        return (f64::NAN, f64::NAN);
    }

    if NORMALIZE {
        normalize_lp2d(a_b);
    }

    let (mut x, mut y) = warm_start.get_initial_point();
    let tol_1d = LpToleranceOptions::with_feas_tol(epsilon * tol.reduce_dim_scale);
    for (i, &(a, b, c)) in warm_start.iter_skip(a_b.iter().enumerate()) {
        // a*x + b*y <= c
        if a * x + b * y > c {
            // Infeasible for this constraint
            // Let a*x + b*y == c
            // Apply 1-dim LP
            if a.abs() < EPS_ZERO && b.abs() < EPS_ZERO {
                // 0 <= c
                if c < -epsilon {
                    collector.clear();
                    return (f64::NAN, f64::NAN); // Infeasible
                }
            } else if a.abs() > b.abs() {
                // a*x + b*y == c
                // x == c/a - (b/a)*y == p*y - q
                let a_inv = 1.0 / a;
                let p = -b * a_inv;
                let q = -c * a_inv;
                // a_*x + b_*y <= c_
                // a_ * (p*y - q) + b_*y <= c_
                // (a_*p + b_)*y <= c_ + a_*q
                let (ymax, _) = lp_1d_core::<_, true>(
                    a_b.iter()
                        .take(i)
                        .map(|&(a_, b_, c_)| (a_ * p + b_, c_ + a_ * q)),
                    &tol_1d,
                    &mut collector,
                );
                if ymax.is_nan() {
                    collector.clear();
                    return (f64::NAN, f64::NAN); // Infeasible
                }
                y = ymax.min(y);
                x = p * y - q;
                collector.collect_2d_id1(i);
            } else {
                // a*x + b*y == c
                // y == c/b - (a/b)*x == p*x - q
                let b_inv = 1.0 / b;
                let p = -a * b_inv;
                let q = -c * b_inv;
                // a_*x + b_*y <= c_
                // a_ * x + b_ * (p*x - q) <= c_
                // (a_ + b_*p)*x <= c_ + b_*q
                let (mut xmax, mut xmin) = lp_1d_core::<_, false>(
                    a_b.iter()
                        .take(i)
                        .map(|(a_, b_, c_)| (a_ + b_ * p, c_ + b_ * q)),
                    &tol_1d,
                    &mut collector,
                );
                if xmax < xmin {
                    if p.abs() * (xmin - xmax) > epsilon {
                        collector.clear();
                        collector.collect_2d_id0(i);
                        return (f64::NAN, f64::NAN); // Infeasible
                    } else {
                        xmax = 0.5 * (xmax + xmin);
                        xmin = xmax;
                    }
                }
                x = if p > 0.0 {
                    collector.collect_2d_id1(i);
                    if xmax.is_finite() {
                        xmax
                    } else {
                        xmin.max(LP_BOUND)
                    }
                } else if p < 0.0 {
                    collector.collect_2d_id0(i);
                    if xmin.is_finite() {
                        xmin
                    } else {
                        xmax.min(-LP_BOUND)
                    }
                } else if xmin > 0.0 {
                    collector.collect_2d_id0(i);
                    xmin
                } else if xmax < 0.0 {
                    collector.collect_2d_id1(i);
                    xmax
                } else {
                    collector.clear();
                    collector.collect_2d_id0(i);
                    0.0
                };
                y = p * x - q;
            }
        }
    }

    // y = if (y / BOUND - 1.0).abs() < epsilon {
    //     f64::INFINITY
    // } else {
    //     y
    // };
    (x, y)
}

/// Linear programming in 2D plane based on incremental method (only maximize Y).  
/// max Y, s.t. Ab[i].0*X+Ab[i].1*Y<=Ab[i].2
/// return (X,Y)
pub(crate) fn lp_2d_incre_max_y<W: WarmStartLp2d, const NORMALIZE: bool>(
    a_b: &mut [(f64, f64, f64)],
    warm_start: &W,
    tol: &LpToleranceOptions,
) -> (f64, f64) {
    lp_2d_incre_max_y_core::<_, _, NORMALIZE>(a_b, warm_start, tol, SilentLpIncCollector)
}

/// Linear programming in 1D line.  
/// a_b[i].0 * x <= a_b[i].1  
/// Return (xmax, xmin) if feasible; otherwise (NaN, NaN).
#[inline(always)]
fn lp_1d_core<C: Lp1dIncCollector, const AUTONAN: bool>(
    a_b: impl Iterator<Item = (f64, f64)>,
    tol: &LpToleranceOptions,
    mut collector: C,
) -> (f64, f64) {
    let epsilon = tol.feas_tol;
    // Find the maximum y such that a*x <= b
    let mut xmax = f64::INFINITY;
    let mut xmin = -f64::INFINITY;

    for (k, (a, b)) in C::enumerate(a_b) {
        if a.abs() < b.abs().clamp(EPS_NORMALIZE, 1.0) * EPS_ZERO {
            // 0 <= b
            if b < -epsilon {
                return (f64::NAN, f64::NAN); // Infeasible
            }
        } else if a > 0.0 {
            // x <= x_
            if b < xmax * a {
                xmax = b / a;
                collector.collect_1d_id_max(k.get_key());
            }
        } else {
            // x >= x_
            if b < xmin * a {
                xmin = b / a;
                collector.collect_1d_id_min(k.get_key());
            }
        }
    }
    if AUTONAN {
        if xmin <= xmax + epsilon {
            if xmin <= xmax {
                (xmax, xmin)
            } else {
                let x = 0.5 * (xmax + xmin);
                (x, x)
            }
        } else {
            (f64::NAN, f64::NAN)
        }
    } else {
        (xmax, xmin)
    }
}

/// Linear programming in 1D line.  
/// a_b[i].0 * x <= a_b[i].1  
/// Return (xmax, xmin) if feasible; otherwise (NaN, NaN).
#[inline(always)]
pub(crate) fn lp_1d<const AUTONAN: bool>(
    a_b: impl Iterator<Item = (f64, f64)>,
    tol: &LpToleranceOptions,
) -> (f64, f64) {
    lp_1d_core::<_, AUTONAN>(a_b, tol, SilentLpIncCollector)
}

/// Warm-start interface for 2D incremental LP.
pub(crate) trait WarmStartLp2d {
    /// Initial feasible guess.
    fn get_initial_point(&self) -> (f64, f64);
    /// Iterate constraints while skipping known-prefix constraints if desired.
    fn iter_skip<I>(&self, a_b: I) -> impl Iterator<Item = I::Item>
    where
        I: Iterator;
    #[cfg(test)]
    /// Rotate warm-start state under objective-space rotation.
    fn rotate(&self, w: (f64, f64)) -> impl WarmStartLp2d;
}

/// 2D warm-start policy: always start from default point without skipping.
#[cfg(test)]
pub(crate) struct Lp2dNoWarmStart;
#[cfg(test)]
impl WarmStartLp2d for Lp2dNoWarmStart {
    #[inline(always)]
    fn get_initial_point(&self) -> (f64, f64) {
        (0.0, LP_BOUND)
    }
    #[inline(always)]
    fn iter_skip<I>(&self, a_b: I) -> impl Iterator<Item = I::Item>
    where
        I: Iterator,
    {
        a_b
    }
    #[allow(refining_impl_trait)]
    #[inline(always)]
    #[cfg(test)]
    fn rotate(&self, _w: (f64, f64)) -> Lp2dNoWarmStart {
        Lp2dNoWarmStart
    }
}
/// 2D warm-start policy with explicit initial point and skip length.
pub(crate) struct Lp2dWarmStart {
    /// Initial point in transformed 2D LP coordinates.
    pub x0: (f64, f64),
    /// Number of leading constraints to skip.
    pub skip: usize,
}
impl WarmStartLp2d for Lp2dWarmStart {
    #[inline(always)]
    fn get_initial_point(&self) -> (f64, f64) {
        self.x0
    }
    #[inline(always)]
    fn iter_skip<I>(&self, a_b: I) -> impl Iterator<Item = I::Item>
    where
        I: Iterator,
    {
        a_b.skip(self.skip)
    }
    #[cfg(test)]
    #[allow(refining_impl_trait)]
    #[inline(always)]
    fn rotate(&self, w: (f64, f64)) -> Lp2dWarmStart {
        Lp2dWarmStart {
            x0: (
                w.1 * self.x0.0 - w.0 * self.x0.1,
                w.0 * self.x0.0 + w.1 * self.x0.1,
            ),
            skip: self.skip,
        }
    }
}

/// Lightweight key adapter for LP collector implementations.
trait Lp1dKey: Copy {
    fn get_key(&self) -> usize;
}
impl Lp1dKey for usize {
    #[inline(always)]
    fn get_key(&self) -> usize {
        *self
    }
}
impl Lp1dKey for () {
    #[inline(always)]
    fn get_key(&self) -> usize {
        0
    }
}
/// Collector interface for 1D incremental LP diagnostics (active indices).
trait Lp1dIncCollector {
    type Key: Lp1dKey;
    fn collect_1d_id_max(&mut self, index: usize);
    fn collect_1d_id_min(&mut self, index: usize);
    fn clear(&mut self);
    fn enumerate<D, I>(a_b: I) -> impl Iterator<Item = (Self::Key, D)>
    where
        I: Iterator<Item = D>;
}
impl<T: Lp1dIncCollector> Lp1dIncCollector for &mut T {
    type Key = T::Key;
    #[inline(always)]
    fn collect_1d_id_max(&mut self, k: usize) {
        (**self).collect_1d_id_max(k);
    }
    #[inline(always)]
    fn collect_1d_id_min(&mut self, k: usize) {
        (**self).collect_1d_id_min(k);
    }
    #[inline(always)]
    fn clear(&mut self) {
        (**self).clear();
    }
    #[inline(always)]
    fn enumerate<D, I>(a_b: I) -> impl Iterator<Item = (Self::Key, D)>
    where
        I: Iterator<Item = D>,
    {
        T::enumerate(a_b)
    }
}
/// Collector interface extending 1D collector with 2D active-set events.
trait Lp2dIncCollector: Lp1dIncCollector {
    fn collect_2d_id1(&mut self, index: usize);
    fn collect_2d_id0(&mut self, index: usize);
}
impl<T: Lp2dIncCollector> Lp2dIncCollector for &mut T {
    #[inline(always)]
    fn collect_2d_id1(&mut self, index: usize) {
        (**self).collect_2d_id1(index);
    }
    #[inline(always)]
    fn collect_2d_id0(&mut self, index: usize) {
        (**self).collect_2d_id0(index);
    }
}
/// Silent collector that does nothing.
struct SilentLpIncCollector;
impl Lp1dIncCollector for SilentLpIncCollector {
    type Key = ();
    #[inline(always)]
    fn collect_1d_id_max(&mut self, _: usize) {}
    #[inline(always)]
    fn collect_1d_id_min(&mut self, _: usize) {}
    #[inline(always)]
    fn clear(&mut self) {}
    #[inline(always)]
    fn enumerate<D, I>(a_b: I) -> impl Iterator<Item = (Self::Key, D)>
    where
        I: Iterator<Item = D>,
    {
        a_b.map(|d| ((), d))
    }
}
impl Lp2dIncCollector for SilentLpIncCollector {
    #[inline(always)]
    fn collect_2d_id1(&mut self, _: usize) {}
    #[inline(always)]
    fn collect_2d_id0(&mut self, _: usize) {}
}

/// Solve 2*2 linear equations: a*x=b.
pub(crate) fn solve_2x2(a: ((f64, f64), (f64, f64)), b: (f64, f64)) -> Option<(f64, f64)> {
    let det = cross_product_2d((a.0.0, a.0.1), (a.1.0, a.1.1));
    if det.abs() < f64::EPSILON {
        return None; // Singular matrix
    }
    let det_inv = 1.0 / det;
    let x1 = cross_product_2d((b.0, a.0.1), (b.1, a.1.1)) * det_inv;
    let x2 = cross_product_2d((a.0.0, b.0), (a.1.0, b.1)) * det_inv;
    Some((x1, x2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::panic;
    use rand::RngExt;
    use std::time::Instant;

    #[test]
    fn test_lp_2d_extreme_direction() {
        run_test_lp_2d_extreme_direction_repeated(1, false);
    }

    /// Average ratio over 100000 experiments: 0.4341553260581677
    #[test]
    #[ignore = "slow"]
    fn test_lp_2d_extreme_direction_robust() {
        run_test_lp_2d_extreme_direction_repeated(100000, true);
    }

    #[test]
    fn test_lp_2d_warm_start() {
        run_test_lp_2d_warm_start_repeated(1, false);
    }

    #[test]
    #[ignore = "slow"]
    fn test_lp_2d_warm_start_robust() {
        run_test_lp_2d_warm_start_repeated(100000, true);
    }

    fn run_test_lp_2d_extreme_direction_repeated(n_exp: usize, flag_print_step: bool) {
        let mut rng = rand::rng();

        let mut ratio_sum = 0.0_f64;
        let mut a_b_buffer = Vec::<(f64, f64, f64)>::with_capacity(106);

        for i_exp in 0..n_exp {
            // 1. Generate a_b, ensure (0,0,0) is feasible and bounded
            let n = rng.random_range(0..100);
            let mut time_sum_direction = 0.0;
            let mut time_sum_lp2d = 0.0;
            let mut n_w = 0;
            let mut a_b: Vec<(f64, f64, f64)> = (0..n)
                .map(|_| {
                    (
                        rng.random_range(-1.0..1.0),
                        rng.random_range(-1.0..1.0),
                        rng.random_range(0.1..2.0),
                    )
                })
                .collect();
            let bound = 1.0;
            a_b.push((1.0, 0.0, bound));
            a_b.push((-1.0, 0.0, bound));
            a_b.push((0.0, 1.0, bound));
            a_b.push((0.0, -1.0, bound));

            if flag_print_step && (i_exp + 1) % 1000 == 0 {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Experiment #{}, num_constraints = {}",
                    i_exp + 1,
                    a_b.len()
                );
            }

            // 2. Generate w in 18 directions
            for i in 0..18 {
                let angle =
                    (i as f64) * std::f64::consts::PI * 2.0 / 18.0 + rng.random_range(-0.01..0.01);
                let w_2d = (angle.cos(), angle.sin());

                // Solve LP
                let epsilon = 1e-9;
                let start = std::time::Instant::now();
                let (x, y, j_opt) = if i == 0 {
                    lp_2d_incre::<_, true>(
                        &a_b,
                        w_2d,
                        &Lp2dNoWarmStart,
                        &LpToleranceOptions::with_feas_tol(epsilon),
                        &mut a_b_buffer,
                    )
                } else {
                    lp_2d_incre::<_, false>(
                        &a_b,
                        w_2d,
                        &Lp2dNoWarmStart,
                        &LpToleranceOptions::with_feas_tol(epsilon),
                        &mut a_b_buffer,
                    )
                };
                time_sum_lp2d += start.elapsed().as_secs_f64() * 1e9;
                if j_opt.is_nan() || j_opt.is_infinite() {
                    panic!("LP failed at exp {}, angle {}", i_exp, i);
                }
                let x_opt = (x, y);

                // 3. Project Extremes
                let start = std::time::Instant::now();
                let (v1, v2) = lp_2d_extreme_direction(
                    &a_b,
                    x_opt,
                    w_2d,
                    &LpToleranceOptions::with_feas_tol(1e-6),
                );
                time_sum_direction += start.elapsed().as_secs_f64() * 1e9;
                n_w += 1;

                if v1.0.is_nan() || v2.0.is_nan() || v1.1.is_nan() || v2.1.is_nan() {
                    crate::verbosity_log!(crate::diag::Verbosity::Summary, "a_b = {a_b:?}");
                    crate::verbosity_log!(crate::diag::Verbosity::Summary, "w_2d = {w_2d:?}");
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "v1 = {v1:?}, v2 = {v2:?}"
                    );
                    panic!(
                        "Extreme direction NAN at exp {}, angle {}. x_opt={:?}",
                        i_exp, i, x_opt
                    );
                }

                // 4. Verification
                for (k, &v) in [v1, v2].iter().enumerate() {
                    // (1) Rotate 90 degree as optimization target direction, determine optimal value is same as x_opt
                    let n1 = if k == 0 { (v.1, -v.0) } else { (-v.1, v.0) };
                    let (_, _, j1) = lp_2d_incre::<_, false>(
                        &a_b,
                        n1,
                        &Lp2dNoWarmStart,
                        &LpToleranceOptions::with_feas_tol(epsilon),
                        &mut a_b_buffer,
                    );
                    let val1 = n1.0 * x + n1.1 * y;

                    // Ideally one of them matches exactly. Due to precision, check closeness.
                    let diff = (j1 - val1).abs();
                    if diff > 1e-5 {
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "a_b = {a_b:?}");
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "w_2d = {w_2d:?}");
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "v1 = {v1:?}, v2 = {v2:?}"
                        );
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "x_opt = {x_opt:?}");
                        panic!(
                            "Rotation check failed for v{}: diff={}\nExp {}, angle {}, w={:?}, v={:?}",
                            k, diff, i_exp, i, w_2d, v
                        );
                    }

                    // (2) Use lp_1d to determine this direction is feasible
                    // i.e. not NAN and v3max > 0.0
                    let delta_max = a_b
                        .iter()
                        .map(|&(a, b, c)| c - a * x_opt.0 - b * x_opt.1)
                        .fold(f64::INFINITY, f64::min)
                        .abs();
                    let iter = a_b.iter().map(|&(a, b, c)| {
                        (a * v.0 + b * v.1, (c - a * x_opt.0 - b * x_opt.1).max(0.0))
                    });
                    let (tmax, _tmin) = lp_1d::<true>(
                        iter,
                        &LpToleranceOptions::with_feas_tol((1.1 * delta_max).max(1e-6)),
                    );
                    if !tmax.is_nan() && tmax <= 0.0 {
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "a_b = {a_b:?}");
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "w_2d = {w_2d:?}");
                        let a_b_t12 = a_b
                            .iter()
                            .map(|&(a, b, c)| {
                                (a * v.0 + b * v.1, (c - a * x_opt.0 - b * x_opt.1).max(0.0))
                            })
                            .collect::<Vec<(f64, f64)>>();
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "a_b for t12 = {a_b_t12:?}"
                        );
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "delta for t12 constraints = {:?}",
                            a_b_t12.iter().map(|(_, b)| *b).collect::<Vec<f64>>()
                        );
                        crate::verbosity_log!(crate::diag::Verbosity::Summary, "x_opt = {x_opt:?}");
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "k = {k}, v1 = {v1:?}, v2 = {v2:?}"
                        );
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Summary,
                            "tmin = {_tmin}, tmax = {tmax}"
                        );
                        panic!(
                            "Direction v{} not feasible (lp_2d returned NAN). Exp {}, angle {}",
                            k, i_exp, i
                        )
                    }
                }
            }
            if flag_print_step && (i_exp + 1) % 1000 == 0 {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp {}: cal_time_lp2d: {} ns, cal_time_direction: {} ns, ratio: {}",
                    i_exp + 1,
                    time_sum_lp2d / n_w as f64,
                    time_sum_direction / n_w as f64,
                    time_sum_direction / time_sum_lp2d
                );
            }
            ratio_sum += time_sum_direction / time_sum_lp2d;
        }
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average ratio over {} experiments: {}",
            n_exp,
            ratio_sum / n_exp as f64
        );
    }

    fn run_test_lp_2d_warm_start_repeated(n_exp: usize, flag_print_step: bool) {
        let mut time_sum_warm = 0.0_f64;
        let mut time_sum_cold = 0.0_f64;

        let mut a_b_buffer = Vec::<(f64, f64, f64)>::with_capacity(106);

        let epsilon = 1E-8;
        for i in 0..n_exp {
            let mut rng = rand::rng();
            let n = rng.random_range(0..10); // number of constraints
            let n_warm = rng.random_range(0..=n);
            let a_b: Vec<(f64, f64, f64)> = (0..n)
                .map(|_| {
                    (
                        rng.random_range(-1.0..1.0),
                        rng.random_range(-1.0..1.0),
                        rng.random_range(0.0..2.0),
                    )
                })
                .collect();
            let w = (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));

            let start = Instant::now();
            let (x_cold, y_cold, j_cold) = lp_2d_incre::<_, true>(
                &a_b,
                w,
                &Lp2dNoWarmStart,
                &LpToleranceOptions::with_feas_tol(epsilon),
                &mut a_b_buffer,
            );
            let time_cold = start.elapsed().as_secs_f64() * 1E9; // ns
            assert!(
                j_cold.is_nan()
                    || j_cold.is_infinite()
                    || (w.0 * x_cold + w.1 * y_cold - j_cold).abs() < 1E-6
            );

            let start = Instant::now();
            let (x_mid, y_mid, _j_mid) = lp_2d_incre::<_, false>(
                &a_b[0..n_warm],
                w,
                &Lp2dNoWarmStart,
                &LpToleranceOptions::with_feas_tol(epsilon),
                &mut a_b_buffer,
            );
            let (x_warm, y_warm, j_warm) = if x_mid.is_nan() || y_mid.is_nan() {
                lp_2d_incre::<_, false>(
                    &a_b,
                    w,
                    &Lp2dNoWarmStart,
                    &LpToleranceOptions::with_feas_tol(epsilon),
                    &mut a_b_buffer,
                )
            } else {
                lp_2d_incre::<_, false>(
                    &a_b,
                    w,
                    &Lp2dWarmStart {
                        x0: (x_mid, y_mid),
                        skip: n_warm,
                    },
                    &LpToleranceOptions::with_feas_tol(epsilon),
                    &mut a_b_buffer,
                )
            };
            let time_warm = start.elapsed().as_secs_f64() * 1E9; // ns
            assert!(
                j_warm.is_nan()
                    || j_warm.is_infinite()
                    || (w.0 * x_warm + w.1 * y_warm - j_warm).abs() < 1E-6
            );

            if flag_print_step && (i + 1) % 1000 == 0 {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp #{}: time_cold/n: {:.3} ns, time_warm/n: {:.3} ns, j_cold: {:.3e}, j_warm: {:.3e}",
                    i + 1,
                    time_cold / n.max(1) as f64,
                    time_warm / n.max(1) as f64,
                    j_cold,
                    j_warm,
                );
            }
            time_sum_cold += time_cold / n.max(1) as f64;
            time_sum_warm += time_warm / n.max(1) as f64;

            let cold_failed =
                j_cold.is_nan() || j_cold.is_infinite() || x_cold.abs() > 1E8 || y_cold.abs() > 1E8;
            let warm_failed =
                j_warm.is_nan() || j_warm.is_infinite() || x_warm.abs() > 1E8 || y_warm.abs() > 1E8;
            let flag_assert =
                (warm_failed != cold_failed) || (!warm_failed && ((j_warm - j_cold).abs() >= 1E-3));
            let flag_assert = if flag_assert {
                crate::verbosity_log!(crate::diag::Verbosity::Summary, "a_b = {a_b:?}");
                crate::verbosity_log!(crate::diag::Verbosity::Summary, "w: {w:?}");
                crate::verbosity_log!(crate::diag::Verbosity::Summary, "n_warm: {n_warm}");
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "j_warm-j_cold: {}",
                    j_warm - j_cold,
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "x_warm = {x_warm:<10.3e}, x_inc = {x_cold:<10.3e}"
                );
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "y_warm = {y_warm:<10.3e}, y_inc = {y_cold:<10.3e}"
                );
                if !warm_failed {
                    let delta = a_b.iter().map(|(a, b, c)| c - a * x_cold - b * y_cold);
                    let delta_min = delta.clone().fold(f64::INFINITY, |a, b| a.min(b));
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Summary,
                        "Minimum slackness at incremental solution: {delta_min}"
                    );
                    if delta_min < -epsilon {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "Incremental solution is infeasible!"
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            } else {
                false
            };
            assert!(!flag_assert);
        }
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Average time per constraint: warm = {:.3} ns, cold = {:.3} ns",
            time_sum_warm / n_exp as f64,
            time_sum_cold / n_exp as f64,
        );
    }
}
