//! Dynamic-programming family backends for second-order path parameterization.
//!
//! # Method identity
//! This module provides **Dynamic Programming (DP)** style solvers for
//! **Time-Optimal Path Parameterization (TOPP2)**.
//!
//! # Contents
//! - `reach_set2`: bidirectional reachable-set construction.
//! - `topp2_ra`: **Reachability Analysis (RA)** solver for TOPP2.
//!
//! RA is included here as a specialized DP-style method on the same TOPP2 state space.

pub(crate) mod reach_set2;
pub(crate) mod topp2_ra;
