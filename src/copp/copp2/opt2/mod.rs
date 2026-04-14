//! Optimization backends for second-order path parameterization (Clarabel-based).
//!
//! # Method identity
//! This module provides optimization formulations for:
//! - **Time-Optimal Path Parameterization (TOPP2)**,
//! - **Convex-Objective Path Parameterization (COPP2)**.
//!
//! # Contents
//! - `clarabel_constraints`: shared standard TOPP2 constraint assembly.
//! - `copp2_socp`: COPP2 backend via **Second-Order Cone Programming (SOCP)**
//!   with normal/expert/core layering.

pub(crate) mod clarabel_constraints;
pub(crate) mod copp2_socp;
