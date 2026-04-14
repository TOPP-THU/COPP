//! Second-order path-parameterization module group.
//!
//! # Method identity
//! This namespace groups:
//! - **Time-Optimal Path Parameterization (TOPP2)** APIs,
//! - **Convex-Objective Path Parameterization (COPP2)** APIs.
//!
//! # Contents
//! - `formulation`: validated problem data models and builders.
//! - `dp2`: **Dynamic Programming (DP)** family backend for
//!   **Reachability Analysis (RA)**.
//! - `opt2`: optimization backends, including
//!   **Second-Order Cone Programming (SOCP)** formulations on Clarabel.
//! - `interpolation`: profile conversion and `s(t)` / `t(s)` mapping utilities.
//!
//! # Export policy
//! This module re-exports public APIs from all submodules for user-facing convenience.

pub(crate) mod dp2;
pub(crate) mod formulation;
pub(crate) mod interpolation;
pub(crate) mod opt2;

/// Crate-internal stable façade for second-order APIs.
///
/// `lib.rs` should import from this module instead of deep internal paths
/// (e.g. `dp2::reach_set2`, `opt2::copp2_socp`) to avoid breakage when
/// internal `mod.rs` layouts are refactored.
#[allow(unused_imports)]
pub(crate) mod stable {
    pub(crate) mod basic {
        pub use super::super::formulation::{
            Copp2Problem, Copp2ProblemBuilder, Topp2Problem, Topp2ProblemBuilder,
        };
        pub use super::super::interpolation::{a_to_b_topp2, s_to_t_topp2, t_to_s_topp2};
    }

    pub(crate) mod reach_set2 {
        pub use super::super::dp2::reach_set2::{
            ReachSet2, ReachSet2Options, ReachSet2OptionsBuilder, reach_set2_backward,
            reach_set2_bidirectional,
        };
    }

    pub(crate) mod topp2_ra {
        pub use super::super::dp2::reach_set2::{ReachSet2Options, ReachSet2OptionsBuilder};
        pub use super::super::dp2::topp2_ra::*;
    }

    pub(crate) mod copp2_socp {
        pub use super::super::opt2::copp2_socp::*;
    }
}
