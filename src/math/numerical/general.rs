//! Small numerical helpers shared by LP and geometric kernels.
//!
//! # Method identity
//! This module provides deterministic algebraic primitives with no allocation:
//! - cross products in 2D/3D,
//! - convexity predicates for 1D/2D affine supports.

/// 2D cross product under right-hand rule.  
/// If result>0, x2 is **anticlockwise** to x1.
#[inline]
pub(crate) fn cross_product_2d(x1: (f64, f64), x2: (f64, f64)) -> f64 {
    x1.0 * x2.1 - x1.1 * x2.0
}
