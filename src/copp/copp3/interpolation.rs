//! Interpolation and profile-conversion utilities for third-order path parameterization.
//!
//! # Method identity
//! This module serves both:
//! - **Time-Optimal Path Parameterization (TOPP3)** workflows,
//! - **Convex-Objective Path Parameterization (COPP3)** workflows.
//!
//! # Scope
//! This module provides deterministic conversions between:
//! - node profiles `a(s) = \dot{s}^2` and `b(s) = \ddot{s}` sampled on stations,
//! - time mapping `t(s)`,
//! - inverse sampling `s(t)`.
//!
//! # Conventions
//! - Path grid uses station samples `s[0..=n]`.
//! - Both `a` and `b` are node-based in TOPP3/COPP3 (`a.len() == b.len() == s.len()`).
//! - `num_stationary = (head, tail)` indicates stationary boundary counts at start/end.

use crate::copp::InterpolationMode;
use crate::math::numerical::{EPS_ZERO, solve_2x2};
use itertools::izip;

/// Compute cumulative time profile `t(s)` from TOPP3/COPP3 profiles `a(s), b(s)`.
///
/// # Semantics
/// - `t_s[i]` is the time at station `s[i]`.
/// - initial condition is `t_s[0] = t0`.
/// - returns `(t_final, t_s)` where `t_final == *t_s.last().unwrap()`.
///
/// # Input contract
/// - valid when `s.len() >= 2 + num_stationary.0 + num_stationary.1`;
/// - requires `a.len() == s.len()` and `b.len() == s.len()`;
/// - invalid input returns `(NaN, empty)`.
///
/// # Returns
/// Returns `(t_final, t_s)` where `t_s[i]` is cumulative time at `s[i]`.
///
/// # Errors
/// This function does not return `Result`; invalid inputs are mapped to `(NaN, vec![])`.
///
/// # Contract
/// - `t_s.len() == s.len()` on valid input.
/// - `t_s[0] == t0` on valid input.
pub fn s_to_t_topp3(
    s: &[f64],
    a: &[f64],
    b: &[f64],
    num_stationary: (usize, usize),
    t0: f64,
) -> (f64, Vec<f64>) {
    if s.len() < 2 + num_stationary.0 + num_stationary.1 || a.len() != s.len() || b.len() != s.len()
    {
        return (f64::NAN, vec![]);
    }
    let mut t_s = Vec::<f64>::with_capacity(s.len()); // t_s[i] = t(s[i]), begin from t0
    let mut t_prev = t0;
    let n = s.len() - 1;
    t_s.push(t_prev);
    if num_stationary.0 > 0 {
        let s0 = s.first().unwrap();
        t_s.resize(1 + num_stationary.0, t_prev);
        for (t_curr, a_curr, s_curr) in izip!(t_s.iter_mut(), a.iter(), s.iter()).skip(1) {
            *t_curr += 3.0 * (s_curr - s0) / a_curr.sqrt();
        }
        t_prev = *t_s.last().unwrap();
    }
    for (s_pair, b_pair, a_curr) in izip!(s.windows(2), b.windows(2), a.iter())
        .skip(num_stationary.0)
        .take(n - num_stationary.0 - num_stationary.1)
    {
        t_prev += integral_rsrqp(
            *a_curr,
            2.0 * b_pair[0],
            (b_pair[1] - b_pair[0]) / (s_pair[1] - s_pair[0]),
            0.0,
            s_pair[1] - s_pair[0],
        );
        t_s.push(t_prev);
    }
    if num_stationary.1 > 0 {
        let s_final = s.last().unwrap();
        let t_final =
            t_prev + 3.0 * (s_final - s[n - num_stationary.1]) / a[n - num_stationary.1].sqrt();
        t_s.resize(s.len(), t_final);
        if num_stationary.1 > 1 {
            for (t_curr, a_curr, s_curr) in izip!(t_s.iter_mut(), a.iter(), s.iter())
                .rev()
                .skip(1)
                .take(num_stationary.1 - 1)
            {
                *t_curr += 3.0 * (s_curr - s_final) / a_curr.sqrt();
            }
        }
    }

    (*t_s.last().unwrap(), t_s)
}

/// Compute definite integral of reciprocal-square-root quadratic polynomial:
/// $$dt = \int_{x_{left}}^{x_{right}} \frac{dx}{\sqrt{c_0 + c_1 x + c_2 x^2}}.$$
fn integral_rsrqp(c0: f64, c1: f64, c2: f64, x_left: f64, x_right: f64) -> f64 {
    if c2 > f64::EPSILON {
        let func = |x: f64| x + 0.5 * c1 / c2 + (x * x + (c1 * x + c0) / c2).sqrt();
        (func(x_right).abs().ln() - func(x_left).abs().ln()) / c2.sqrt()
    } else if c2 < -f64::EPSILON {
        let delta = c1 * c1 - 4.0 * c2 * c0;
        if delta > 0.0 {
            let func = |x: f64| (-2.0 * c2 * x - c1) / delta.sqrt();
            (func(x_right).asin() - func(x_left).asin()) / (-c2).sqrt()
        } else {
            f64::INFINITY
        }
    } else if c1.abs() > f64::EPSILON {
        // Dt = \int_{xl}^{xr} dx/sqrt(C1*x+C0)
        2.0 / c1 * ((c1 * x_right + c0).sqrt() - (c1 * x_left + c0).sqrt())
    } else if c0.abs() > f64::EPSILON {
        // Dt = \int_{xl}^{xr} dx/sqrt(C0)
        (x_right - x_left) / c0.sqrt()
    } else {
        f64::INFINITY
    }
}

/// Interpolate inverse mapping `s(t)` from `a(s)`, `b(s)`, and sampled `t(s)`.
///
/// # Modes
/// - `UniformTimeGrid(t0, dt, include_final)`: generate uniform time samples;
/// - `NonUniformTimeGrid(t_sample)`: use caller-provided increasing samples.
///
/// # Input contract
/// - requires `s.len() >= 2`, `a.len() == s.len()`, `b.len() == s.len()`, `t_s.len() == s.len()`;
/// - requires `t_s` strictly increasing;
/// - invalid input returns empty vector.
///
/// # Output semantics
/// - output length matches requested sample count in each mode;
/// - for out-of-range time samples, output entries are `NaN`.
///
/// # Returns
/// Returns sampled `s(t)` values under the requested interpolation `mode`.
///
/// # Errors
/// This function does not return `Result`; malformed inputs are mapped to empty output.
///
/// # Contract
/// - preserves caller time-sample ordering.
/// - never panics on invalid user data paths (returns empty vector).
pub fn t_to_s_topp3(
    s: &[f64],
    a: &[f64],
    b: &[f64],
    num_stationary: (usize, usize),
    t_s: &[f64],
    mode: InterpolationMode<'_>,
) -> Vec<f64> {
    if s.len() < 2
        || a.len() != s.len()
        || b.len() != s.len()
        || t_s.len() != s.len()
        || t_s.windows(2).any(|w| w[0] >= w[1])
    {
        return vec![];
    }
    match mode {
        InterpolationMode::UniformTimeGrid(t0, dt, include_final) => {
            if dt <= 0.0 {
                return vec![];
            }
            // num_t * dt + t0 <= t_final
            let num_t = ((t_s.last().unwrap() - t0) / dt).floor() as usize;
            let mut s_t = t_to_s_topp3_core(
                s,
                a,
                b,
                num_stationary,
                t_s,
                (0..num_t).map(|i| t0 + i as f64 * dt),
                num_t,
            );
            if include_final {
                let flag = if s_t.is_empty() {
                    t0 <= *t_s.last().unwrap()
                } else {
                    *s_t.last().unwrap() < *s.last().unwrap()
                };
                if flag {
                    s_t.push(*s.last().unwrap());
                }
            }
            s_t
        }
        InterpolationMode::NonUniformTimeGrid(t_sample) => {
            if t_sample.is_empty() || t_sample.windows(2).any(|w| w[0] >= w[1]) {
                // Exclude the case where t_sample.len() == 1
                return vec![];
            }
            t_to_s_topp3_core(
                s,
                a,
                b,
                num_stationary,
                t_s,
                t_sample.iter().cloned(),
                t_sample.len(),
            )
        }
    }
}

fn t_to_s_topp3_core(
    s: &[f64],
    a: &[f64],
    b: &[f64],
    num_stationary: (usize, usize),
    t_s: &[f64],
    mut t_sample: impl Iterator<Item = f64>,
    len_t_sample: usize,
) -> Vec<f64> {
    // Core inverse interpolation kernel for `t_to_s_topp3`.
    // It consumes increasing `t_sample` values and emits corresponding `s(t)`.
    // Map t to s
    let &t_start = t_s.first().unwrap();
    let &t_final = t_s.last().unwrap();
    let mut s_t = Vec::<f64>::with_capacity(len_t_sample + 1); // s_t[i] = s(t[i])
    let Some(mut t_curr) = t_sample.next() else {
        return vec![];
    };
    while t_curr < t_start {
        s_t.push(f64::NAN);
        let Some(t) = t_sample.next() else {
            return s_t;
        };
        t_curr = t;
    }

    if num_stationary.0 > 0 {
        let s0 = s.first().unwrap();
        let a_stationary = a[num_stationary.0];
        let t_stationary = t_s[num_stationary.0];
        let d3u_over_6 =
            a_stationary.sqrt() * a_stationary / (27.0 * (s[num_stationary.0] - s0).powi(2));
        while t_curr <= t_stationary {
            s_t.push(s0 + d3u_over_6 * (t_curr - t_start).powi(3));
            let Some(t) = t_sample.next() else {
                return s_t;
            };
            t_curr = t;
        }
    }

    for (s_pair, &a_curr, b_pair, t_pair) in
        izip!(s.windows(2), a.iter(), b.windows(2), t_s.windows(2))
            .skip(num_stationary.0)
            .take(s.len() - num_stationary.0 - num_stationary.1 - 1)
    {
        while t_curr <= t_pair[1] {
            s_t.push(
                s_pair[0]
                    + inverse_rsrqp(
                        a_curr,
                        2.0 * b_pair[0],
                        (b_pair[1] - b_pair[0]) / (s_pair[1] - s_pair[0]),
                        0.0,
                        t_curr - t_pair[0],
                    ),
            );
            let Some(t) = t_sample.next() else {
                return s_t;
            };
            t_curr = t;
        }
    }

    if num_stationary.1 > 0 {
        let s_final = s.last().unwrap();
        let a_stationary = a[s.len() - num_stationary.1 - 1];
        let d3u_over_6 = a_stationary.sqrt() * a_stationary
            / (27.0 * (s_final - s[s.len() - num_stationary.1 - 1]).powi(2));
        while t_curr <= t_final {
            s_t.push(s_final + d3u_over_6 * (t_curr - t_final).powi(3));
            let Some(t) = t_sample.next() else {
                return s_t;
            };
            t_curr = t;
        }
    }

    s_t.push(f64::NAN);
    while t_sample.next().is_some() {
        s_t.push(f64::NAN);
    }
    s_t
}

/// Solve `x_right` from
/// $$dt = \int_{x_{left}}^{x_{right}} \frac{dx}{\sqrt{c_0 + c_1 x + c_2 x^2}}.$$
fn inverse_rsrqp(c0: f64, c1: f64, c2: f64, x_left: f64, dt: f64) -> f64 {
    if dt == 0.0 {
        return x_left;
    }
    let delta = c1 * c1 - 4.0 * c2 * c0;
    if c2 > f64::EPSILON {
        let mu = (c2.sqrt() * dt
            + (x_left + 0.5 * c1 / c2 + (x_left * x_left + (c1 * x_left + c0) / c2).sqrt())
                .abs()
                .ln())
        .exp();
        let xr1 = -0.5 * c1 / c2 + 0.5 * (mu + delta / (4.0 * c2 * c2 * mu));
        let xr2 = -0.5 * c1 / c2 - 0.5 * (mu + delta / (4.0 * c2 * c2 * mu));
        let mut flag1 = true;
        let mut flag2 = true;
        if dt > 0.0 {
            flag1 &= xr1 > x_left;
            flag2 &= xr2 > x_left;
        } else {
            flag1 &= xr1 < x_left;
            flag2 &= xr2 < x_left;
        }
        if flag1 && flag2 {
            let dt1 = integral_rsrqp(c0, c1, c2, x_left, xr1);
            let dt2 = integral_rsrqp(c0, c1, c2, x_left, xr2);
            if (dt1 - dt).abs() < (dt2 - dt).abs() {
                xr1
            } else {
                xr2
            }
        } else if flag1 {
            xr1
        } else if flag2 {
            xr2
        } else {
            f64::INFINITY
        }
    } else if c2 < -f64::EPSILON {
        (c1 + delta.sqrt()
            * ((-c2).sqrt() * dt + ((-2.0 * c2 * x_left - c1) / delta.sqrt()).asin()).sin())
            / (-2.0 * c2)
    } else if c1.abs() > f64::EPSILON {
        ((0.5 * c1 * dt + (c1 * x_left + c0).sqrt()).powi(2) - c0) / c1
    } else if c0.abs() > f64::EPSILON {
        c0.sqrt() * dt + x_left
    } else {
        f64::INFINITY
    }
}

/// Post-process `(a, b)` so that interpolated `a(s)` stays strictly positive per interval.
///
/// This is a numerical safety utility for downstream timing integration on
/// profiles that may be very close to zero due to finite precision.
///
/// # Returns
/// Returns `true` when in-place adjustment succeeds, otherwise `false`.
///
/// # Errors
/// This function does not return `Result`; invalid inputs are reported by `false`
/// with diagnostic prints.
///
/// # Contract
/// - requires `a.len() == b.len() == s.len()` and `s.len() >= 4`;
/// - requires endpoint `a` values to be nonnegative.
pub fn force_positive_a(
    a: &mut [f64],
    b: &mut [f64],
    s: &[f64],
    num_stationary: (usize, usize),
    a_min: f64,
) -> bool {
    let n = s.len();
    if a.len() != n || b.len() != n {
        crate::verbosity_log!(
            crate::diag::Verbosity::Debug,
            "force_positive_a: a, b, s should have the same length"
        );
        return false;
    }
    if n < 4 {
        crate::verbosity_log!(
            crate::diag::Verbosity::Debug,
            "force_positive_a: the length of a, b, s should be at least 4"
        );
        return false;
    }
    if a.iter().any(|&a| a < 0.0) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Debug,
            "force_positive_a: a should be non-negative at each end point"
        );
        return false;
    }
    // Now we have a(s[i]) >= 0, and we would like to modify a(s) > 0 for s in (s[i], s[i+1]) if a(s) can be negative for some s in (s[i], s[i+1]).
    let mut flag_succeed = true;
    for i in (num_stationary.0 + 1)..(n - 2 - num_stationary.1) {
        // Consider a[i-1], a[i], a[i+1], a[i+2]
        let b1 = b[i];
        let b2 = b[i + 1];
        if b1 < 0.0 && b2 > 0.0 {
            // a(s) = a[i] + 2 * b[i] * (s - s[i]) + (b[i+1] - b[i]) / ds1 * (s - s[i])^2
            // b[i] ^ 2 < a[i] * (b[i+1] - b[i]) / ds1 should hold
            // b[i] ^ 2 * ds1 < a[i] * (b[i+1] - b[i]) should hold
            let s1 = s[i];
            let s2 = s[i + 1];
            let ds1 = s2 - s1;
            let a1 = a[i];
            let amin = a_min.max(a1.min(a[i + 1]));
            let amin = if amin > 10.0 * EPS_ZERO {
                0.1 * amin
            } else if amin > EPS_ZERO {
                EPS_ZERO
            } else {
                amin
            };
            let da = a1 - amin;
            let db = b2 - b1;
            if b1 * b1 * ds1 >= da * db {
                // a(s) <= 0 holds in (s[i], s[i+1])
                // We add c0 on (s[i-1],s[i+2]), c1 on (s[i],s[i+2]), and c2 on (s[i+1],s[i+2])
                // x[i-1] and x[i+2] should keep the same.
                // (i) --- c0*(s[i+2] - s[i-1]) + c1*(s[i+2] - s[i]) + c2*(s[i+2] - s[i+1]) == 0
                // (ii) --- c0*(s[i+2] - s[i-1])^2 + c1*(s[i+2] - s[i])^2 + c2*(s[i+2] - s[i+1])^2 == 0
                let s0 = s[i - 1];
                let s3 = s[i + 2];
                let delta_s_end = (s3 - s0, s3 - s1, s3 - s2);
                let coeff = match solve_2x2(
                    (
                        (delta_s_end.1, delta_s_end.2),
                        (delta_s_end.1 * delta_s_end.1, delta_s_end.2 * delta_s_end.2),
                    ),
                    (-delta_s_end.0, -delta_s_end.0 * delta_s_end.0),
                ) {
                    Some(coeff) => {
                        // A*[c1;c2] = b*c0
                        coeff
                    }
                    None => {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "coeff is None? A = {:?}, b = {:?}",
                            (
                                (delta_s_end.1, delta_s_end.2),
                                (delta_s_end.1 * delta_s_end.1, delta_s_end.2 * delta_s_end.2)
                            ),
                            (-delta_s_end.0, -delta_s_end.0 * delta_s_end.0)
                        );
                        flag_succeed = false;
                        continue;
                    }
                };
                // c1 = coeff.0 * c0, c2 = coeff.1 * c0
                // Changes: a[i] += c0 * (s1-s0)^2, b[i] += c0 * (s1-s0), b[i+1] += c0 * (s2-s0) + c1 * (s2-s1)
                let ds0 = s1 - s0;
                let coeff_c = (ds0 * ds0, ds0, ds0 + ds1 * (1.0 + coeff.0));
                // Changes: a[i] += c0 * coeff_c.0, b[i] += c0 * coeff_c.1, b[i+1] += c0 * coeff_c.2
                // We hope that a(s) = a[i] + 2 * b[i] * (s - s[i]) + (b[i+1] - b[i]) / ds1 * (s - s[i])^2 >= amin holds in (s[i],s[i+1])
                // b[i] ^ 2 * ds1 == (a[i] - amin) * (b[i+1] - b[i]) should hold for new ones.
                // For old ones: (b[i] + coeff_c.1 * c0) ^ 2 * ds1 == (a[i] - amin + coeff_c.0 * c0) * (b[i+1] - b[i] + (coeff_c.2-coeff_c.1) * c0). Now solve c0.
                // (coeff_c.1^2 * c0^2 + 2 * b1 * coeff_c.1 * c0 + b1 ^ 2) * ds1 == coeff_c.0 * (coeff_c.2-coeff_c.1) * c0^2 + (da * (coeff_c.2-coeff_c.1) + coeff_c.0 * db) * c0 + da * db
                // (coeff_c.1^2 * ds1 - coeff_c.0 * (coeff_c.2-coeff_c.1)) * c0^2 + (2 * b1 * coeff_c.1 * ds1 - da * (coeff_c.2-coeff_c.1) - coeff_c.0 * db) * c0 + (b1 * b1 * ds1 - da * db) == 0
                let coeff_solve = (
                    coeff_c.1 * coeff_c.1 * ds1 - coeff_c.0 * (coeff_c.2 - coeff_c.1),
                    2.0 * b1 * coeff_c.1 * ds1 - da * (coeff_c.2 - coeff_c.1) - coeff_c.0 * db,
                    b1 * b1 * ds1 - da * db,
                );
                let norm = coeff_solve.0.abs() + coeff_solve.1.abs() + coeff_solve.2.abs();
                if norm < EPS_ZERO {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "norm = {norm} < EPS_ZERO, coeff_solve = {coeff_solve:.8?}"
                    );
                    flag_succeed = false;
                    continue;
                }
                let norm_inv = 1.0 / norm;
                let coeff_solve = (
                    coeff_solve.0 * norm_inv,
                    coeff_solve.1 * norm_inv,
                    coeff_solve.2 * norm_inv,
                );
                // coeff_solve.0 * c0^2 + coeff_solve.1 * c0 + coeff_solve.2 == 0
                let c0 = if coeff_solve.0.abs() > EPS_ZERO {
                    // Use quadratic formula to solve for c0
                    let discriminant =
                        coeff_solve.1 * coeff_solve.1 - 4.0 * coeff_solve.0 * coeff_solve.2;
                    if discriminant < 0.0 {
                        if coeff_c.1.abs() > EPS_ZERO && coeff_c.2.abs() > EPS_ZERO {
                            (-b1 / coeff_c.1).min(b2 / coeff_c.2)
                        } else if coeff_c.1.abs() > EPS_ZERO {
                            -b1 / coeff_c.1
                        } else if coeff_c.2.abs() > EPS_ZERO {
                            b2 / coeff_c.2
                        } else {
                            crate::verbosity_log!(
                                crate::diag::Verbosity::Debug,
                                "discriminant = {discriminant:.8} < 0 for c0 (i={i}): coeff_solve = {coeff_solve:.8?}, coeff_c = {coeff_c:.8?}"
                            );
                            flag_succeed = false;
                            continue;
                        }
                    } else {
                        let sqrt_discriminant = discriminant.sqrt();
                        // c0: (max, min)
                        let c0 = if coeff_solve.0 > 0.0 {
                            (
                                (-coeff_solve.1 + sqrt_discriminant) / (2.0 * coeff_solve.0),
                                (-coeff_solve.1 - sqrt_discriminant) / (2.0 * coeff_solve.0),
                            )
                        } else {
                            (
                                (-coeff_solve.1 - sqrt_discriminant) / (2.0 * coeff_solve.0),
                                (-coeff_solve.1 + sqrt_discriminant) / (2.0 * coeff_solve.0),
                            )
                        };
                        if c0.1 >= 0.0 { c0.1 } else { c0.0 }
                    }
                } else {
                    // Linear case
                    -coeff_solve.2 / coeff_solve.1
                };
                a[i] += coeff_c.0 * c0;
                b[i] += coeff_c.1 * c0;
                b[i + 1] += coeff_c.2 * c0;
                a[i + 1] += (coeff_c.0 + (coeff_c.1 + coeff_c.2) * ds1) * c0;
            }
        }
    }

    flag_succeed
}
