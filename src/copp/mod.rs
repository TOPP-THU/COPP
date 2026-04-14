//! Internal COPP/TOPP namespace and crate-level façade.
//!
//! # Method identity
//! This module is the root namespace for:
//! - **TOPP2 / TOPP3** (time-optimal path parameterization),
//! - **COPP2 / COPP3** (convex-objective path parameterization),
//! - shared numerical backends and utility abstractions used by planners.
//!
//! Users should not import from this module directly; use [`crate::solver`] and [`crate::prelude`]
//! for stable public entry points.
//!
//! # Submodule responsibilities
//! - `constraints` (**public**): station-indexed circular storage and constraint access layer.
//! - `copp2` (**crate-private**): second-order formulation + DP/optimization internals.
//! - `copp3` (**crate-private**): third-order formulation + DP/optimization internals.
//! - `clarabel_backend` (crate-private module, selectively re-exported): Clarabel options/types.
//! - `general` (crate-private module, selectively re-exported): shared enums/helpers used by APIs.
//! - `objectives` (crate-private module, selectively re-exported): objective descriptors for COPP.
//!
//! # Export policy (façade design)
//! This module acts as a **thin public façade**:
//! - Stable user-facing concepts are available as `copp::copp::{TypeOrFn}` via `pub use`.
//! - User-facing solver entry points are provided by [`crate::solver`] and [`crate::prelude`].
//! - Internal implementation modules stay `pub(crate)` to reduce API surface and semver burden.
//!
//! In other words, `pub mod` exposes a named namespace, while `pub use` flattens selected items
//! into this root module for ergonomic import paths.

pub(crate) mod clarabel_backend;
pub mod constraints;
pub(crate) mod copp2;
pub(crate) mod copp3;
pub(crate) mod general;
pub(crate) mod objectives;

pub(crate) use clarabel_backend::*;
pub use general::*;
pub use objectives::*;
