//! General shared definitions and numeric utilities for TOPP/COPP flows.
//!
//! # Method identity
//! This module hosts lightweight, cross-cutting primitives that are reused by
//! multiple solver/formulation layers, including:
//! - interpolation policy descriptors for output trajectory sampling;
//! - tolerant floating-point comparison helpers used by feasibility and
//!   convergence checks.
//!
//! # Design notes
//! - Types here are intentionally small and dependency-free.
//! - Approximate comparison uses a mixed tolerance
//!   $\max(\text{abs\_tol},\ \text{rel\_tol}\cdot\max(|x_1|,|x_2|))$.
//! - `approx_order()` is preferred over direct equality checks when decisions
//!   depend on floating-point values near boundaries.

/// Time-grid policy used when interpolating path-parameterization outputs.
///
/// This enum describes how target sample times are provided to interpolation
/// routines after a trajectory has been parameterized.
pub enum InterpolationMode<'a> {
    /// Uniform sampling grid.
    ///
    /// # Tuple fields
    /// - `t0`: time stamp of the first path station (`s[0]`).
    /// - `dt`: constant sampling period (`dt > 0` expected by callers).
    /// - `include_final`: handling of non-integer final step.
    ///
    /// # Final-sample policy
    /// If final time `t_final` is not an integer multiple of `dt` from `t0`:
    /// - `include_final = true`: append one final sample exactly at `t_final`.
    /// - `include_final = false`: stop at `t0 + n\,dt`, where `n` is the
    ///   largest integer satisfying `t0 + n\,dt \le t_final`.
    UniformTimeGrid(f64, f64, bool),

    /// User-provided non-uniform sampling grid.
    ///
    /// # Tuple fields
    /// - `t_samples`: strictly increasing time stamps.
    ///
    /// # Contract
    /// Callers should provide an increasing grid and ensure the first sample is
    /// not earlier than the interpolation start time.
    NonUniformTimeGrid(&'a [f64]),
}

/// Approximate ordering relation for two floating-point values.
///
/// Returned by [`approx_order()`] when comparing `x1` and `x2` under mixed
/// absolute/relative tolerance.
pub(crate) enum ApproxOrdering {
    /// `x1 < x2` under tolerance-aware comparison.
    Less,
    /// `x1 \approx x2` within tolerance band.
    Equal,
    /// `x1 > x2` under tolerance-aware comparison.
    Greater,
}

/// Compute the mixed absolute/relative comparison threshold.
///
/// # Formula
/// `threshold = max(abs_tol, rel_tol * max(|x1|, |x2|))`
///
/// # Parameters
/// - `x1`, `x2`: values to be compared.
/// - `abs_tol`: absolute tolerance component.
/// - `rel_tol`: relative tolerance component.
///
/// # Returns
/// A non-negative scalar used as symmetric comparison band around zero for
/// `dx = x1 - x2`.
///
#[inline(always)]
pub(crate) fn threshold_approx(x1: f64, x2: f64, abs_tol: f64, rel_tol: f64) -> f64 {
    abs_tol.max(rel_tol * x1.abs().max(x2.abs()))
}

/// Compare two floating-point values with mixed tolerance and return ordering.
///
/// # Decision rule
/// Let `dx = x1 - x2` and
/// `threshold = threshold_approx(x1, x2, abs_tol, rel_tol)`.
///
/// - return [`ApproxOrdering::Greater`] if `dx > threshold`;
/// - return [`ApproxOrdering::Less`] if `dx < -threshold`;
/// - otherwise return [`ApproxOrdering::Equal`].
///
/// # Parameters
/// - `x1`, `x2`: values to compare.
/// - `abs_tol`: absolute tolerance.
/// - `rel_tol`: relative tolerance.
///
/// # Returns
/// Tolerance-aware ordering relation between `x1` and `x2`.
#[inline(always)]
pub(crate) fn approx_order(x1: f64, x2: f64, abs_tol: f64, rel_tol: f64) -> ApproxOrdering {
    let threshold = threshold_approx(x1, x2, abs_tol, rel_tol);
    let dx = x1 - x2;
    if dx > threshold {
        ApproxOrdering::Greater
    } else if dx < -threshold {
        ApproxOrdering::Less
    } else {
        ApproxOrdering::Equal
    }
}
