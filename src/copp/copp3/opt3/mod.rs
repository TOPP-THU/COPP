//! Optimization backends for third-order path parameterization (Clarabel-based).
//!
//! # Method identity
//! This module provides optimization formulations for third-order problems:
//! - **TOPP3** (time-optimal),
//! - **COPP3** (convex-objective).
//!
//! # Contents
//! - `clarabel_constraints`: shared standard TOPP3/COPP3 constraint assembly.
//! - `topp3_lp`: TOPP3 LP backend.
//! - `topp3_socp`: TOPP3 SOCP backend.
//! - `copp3_socp`: COPP3 SOCP backend with normal/expert/core layering.

pub(crate) mod clarabel_constraints;
pub(crate) mod copp3_socp;
pub(crate) mod topp3_lp;
pub(crate) mod topp3_socp;
