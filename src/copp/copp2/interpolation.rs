//! Interpolation and profile-conversion utilities for second-order path parameterization.
//!
//! # Method identity
//! This module serves both:
//! - **Time-Optimal Path Parameterization (TOPP2)** workflows,
//! - **Convex-Objective Path Parameterization (COPP2)** workflows.
//!
//! # Scope
//! This module provides deterministic conversions among:
//! - path-parameter profile `a(s) = \dot{s}^2`,
//! - derivative-like profile `b(s) = \frac{1}{2}\frac{da}{ds}` on segments,
//! - time mapping `t(s)` and inverse sampling `s(t)`.
//!
//! # Conventions
//! - Path grid uses station samples `s[0..=n]`.
//! - State profile `a` is node-based (`a.len() == s.len()`).
//! - Profile `b` is segment-based (`b.len() == s.len() - 1`).

use crate::copp::InterpolationMode;
use itertools::izip;

/// Compute segment profile `b` from node profile `a`.
///
/// # Definition
/// For each segment `[s_k, s_{k+1}]`, this function computes:
/// $b_k = \frac{1}{2}\frac{a_{k+1}-a_k}{s_{k+1}-s_k}$.
///
/// # Input contract
/// - valid when `s.len() >= 2` and `a.len() == s.len()`;
/// - otherwise returns an empty vector.
///
/// # Returns
/// Returns `b` with `b.len() == s.len() - 1`.
///
/// # Errors
/// This function does not return `Result`; invalid input is mapped to empty output.
///
/// # Contract
/// - Output ordering is consistent with segment ordering on `s.windows(2)`.
/// - No allocation beyond returned vector and iterator temporaries.
pub fn a_to_b_topp2(s: &[f64], a: &[f64]) -> Vec<f64> {
    if s.len() < 2 || a.len() != s.len() {
        return vec![];
    }
    s.windows(2)
        .zip(a.windows(2))
        .map(|(s_pair, a_pair)| 0.5 * (a_pair[1] - a_pair[0]) / (s_pair[1] - s_pair[0]))
        .collect::<Vec<f64>>()
}

/// Compute cumulative time profile `t(s)` from `a(s)`.
///
/// # Semantics
/// - `t_s[i]` is the time at station `s[i]`.
/// - initial condition is `t_s[0] = t0`.
/// - returns `(t_final, t_s)` where `t_final == *t_s.last().unwrap()`.
///
/// # Input contract
/// - valid when `s.len() >= 2` and `a.len() == s.len()`;
/// - otherwise returns `(NaN, empty)`.
///
/// # Returns
/// Returns `(t_final, t_s)` with `t_s.len() == s.len()` on valid input.
///
/// # Errors
/// This function does not return `Result`; invalid input is mapped to `(NaN, vec![])`.
///
/// # Contract
/// - `t_s` is monotonically increasing when `a` is nonnegative and `s` is increasing.
/// - `t_s[0] == t0` always holds on valid input.
pub fn s_to_t_topp2(s: &[f64], a: &[f64], t0: f64) -> (f64, Vec<f64>) {
    if s.len() < 2 || a.len() != s.len() {
        return (f64::NAN, vec![]);
    }
    // Map s to t
    let mut t_s = Vec::<f64>::with_capacity(s.len()); // t_s[i] = t(s[i]), begin from t0
    let mut t_prev = t0;
    t_s.push(t_prev);
    for (s_pair, a_pair) in s.windows(2).zip(a.windows(2)) {
        t_prev += 2.0 * (s_pair[1] - s_pair[0]) / (a_pair[0].sqrt() + a_pair[1].sqrt());
        t_s.push(t_prev);
    }
    (t_prev, t_s)
}

/// Interpolate inverse mapping `s(t)` from `a(s)` and sampled `t(s)`.
///
/// # Modes
/// - `UniformTimeGrid(t0, dt, include_final)`: generate uniform time samples;
/// - `NonUniformTimeGrid(t_sample)`: use caller-provided increasing samples.
///
/// # Input contract
/// - requires `s.len() >= 2`, `a.len() == s.len()`, `t_s.len() == s.len()`;
/// - requires `t_s` strictly increasing.
/// - invalid input returns empty vector.
///
/// # Output semantics
/// - output length matches requested sample count in each mode;
/// - for out-of-range time samples, output entries are `NaN`.
///
/// # Returns
/// Returns sampled `s(t)` values according to `mode`.
///
/// # Errors
/// This function does not return `Result`; invalid input or invalid `mode` settings
/// are mapped to empty output.
///
/// # Contract
/// - preserves requested sample order;
/// - never panics for malformed user input (falls back to empty vector).
pub fn t_to_s_topp2(s: &[f64], a: &[f64], t_s: &[f64], mode: InterpolationMode<'_>) -> Vec<f64> {
    if s.len() < 2
        || a.len() != s.len()
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
            let mut s_t =
                t_to_s_topp2_core(s, a, t_s, (0..num_t).map(|i| t0 + i as f64 * dt), num_t);
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
                // Empty or non-increasing sample sequence is invalid.
                return vec![];
            }
            t_to_s_topp2_core(s, a, t_s, t_sample.iter().cloned(), t_sample.len())
        }
    }
}

/// Core inverse interpolation kernel for `t_to_s_topp2`.
///
/// It consumes increasing `t_sample` values and emits corresponding `s(t)` by
/// segment-wise inversion with quadratic-in-`a` local model.
fn t_to_s_topp2_core(
    s: &[f64],
    a: &[f64],
    t_s: &[f64],
    mut t_sample: impl Iterator<Item = f64>,
    len_t_sample: usize,
) -> Vec<f64> {
    let &t_start = t_s.first().unwrap();
    // Map t to s
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

    for (s_pair, a_pair, t_pair) in izip!(s.windows(2), a.windows(2), t_s.windows(2)) {
        while t_curr <= t_pair[1] {
            s_t.push(
                s_pair[0]
                    + inverse_2order(
                        a_pair[0],
                        (a_pair[1] - a_pair[0]) / (s_pair[1] - s_pair[0]),
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

    s_t.push(f64::NAN);
    while t_sample.next().is_some() {
        s_t.push(f64::NAN);
    }
    s_t
}

/// Solve `x_right` from the integral equation
/// $dt = \int_{x_{left}}^{x_{right}} \frac{dx}{\sqrt{c_0 + c_1 x}}$.
#[inline]
fn inverse_2order(c0: f64, c1: f64, x_left: f64, dt: f64) -> f64 {
    if dt == 0.0 {
        x_left
    } else if c1.abs() > f64::EPSILON {
        (((c0 + c1 * x_left).sqrt() + 0.5 * c1 * dt).powi(2) - c0) / c1
    } else if c0.abs() > f64::EPSILON {
        x_left + c0.sqrt() * dt
    } else {
        f64::INFINITY
    }
}
