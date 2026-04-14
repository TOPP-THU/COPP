//! Demo robot models used by tests and examples.
//!
//! # Included models
//! - `robot_2dof` (test-only): planar 2-link model with closed-form dynamics.

/// Test-only 2-DoF planar robot implementation and checks.
#[cfg(test)]
mod robot_2dof;

/// Re-export test-only 2-DoF robot symbols for internal test modules.
#[cfg(test)]
pub(crate) use robot_2dof::*;
