//! Third-order path-parameterization module group.
//!
//! # Method identity
//! This namespace groups user-facing APIs for third-order planners on `(a, b)` state space:
//! - **TOPP3** (time-optimal),
//! - **COPP3** (convex-objective).
//!
//! # Contents
//! - `opt3`: optimization backends, including LP/SOCP variants on Clarabel.
//! - `formulation`: validated problem data models/builders for TOPP3/COPP3.
//! - `interpolation`: profile conversion and `s(t)` / `t(s)` mapping helpers.
//!
//! # Export policy
//! This module re-exports public APIs from all submodules for user convenience.

pub(crate) mod formulation;
pub(crate) mod interpolation;
pub(crate) mod opt3;

/// Crate-internal stable façade for third-order APIs.
///
/// `lib.rs` should import from this module instead of deep internal paths
/// (`opt3::*`, `formulation::*`, `interpolation::*`) so internal
/// module refactors do not propagate to public façade wiring.
#[allow(unused_imports)]
pub(crate) mod stable {
    pub(crate) mod basic {
        pub use super::super::formulation::{
            Copp3Problem, Copp3ProblemBuilder, Topp3Problem, Topp3ProblemBuilder,
        };
        pub use super::super::interpolation::{s_to_t_topp3, t_to_s_topp3};
    }

    pub(crate) mod copp3_socp {
        pub use super::super::opt3::copp3_socp::*;
    }

    pub(crate) mod topp3_lp {
        pub use super::super::interpolation::force_positive_a;
        pub use super::super::opt3::topp3_lp::*;
    }

    pub(crate) mod topp3_socp {
        pub use super::super::interpolation::force_positive_a;
        pub use super::super::opt3::topp3_socp::*;
    }
}

/// Generic COPP3 result tuple.
///
/// Layout: `(a, b, (num_stationary_start, num_stationary_final))`.
pub type Copp3Result = (Vec<f64>, Vec<f64>, (usize, usize));
