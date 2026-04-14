//! `copp` crate public API.
//!
//! # What problem does this library solve?
//! This library targets **OPP** (Optimal Path Parameterization) problems.
//! Given a geometric path parameterization
//! $q = q(s)$, the solver finds a time law $s = s(t)$ such that constraints are
//! satisfied while an objective is optimized.
//!
//! In short, we convert geometry-space motion into time-space scheduling:
//! $$
//! q = q(s) \quad \Longrightarrow \quad s = s(t),
//! $$
//! and solve for globally optimal / KKT-satisfying / heuristic near-optimal
//! trajectories under the selected backend and objective model.
//!
//! # Objective families: TOPP vs COPP
//! - **TOPP**: Time-Optimal Path Parameterization (typical objective is minimum
//!   traversal time).
//! - **COPP**: Convex-Objective Path Parameterization (supports richer convex
//!   objectives beyond pure time-optimality).
//!
//! For non-convex objectives, a practical approach is **SCP**:
//! perform DC decomposition and iterative objective linearization. Constraint
//! linearization (notably for third-order models) is already implemented in this
//! crate. For objective modeling details, see COPP solver families and
//! `CoppObjective`-related APIs.
//!
//! # Constraint-order families: 2nd vs 3rd order
//! - **2nd-order** (TOPP2/COPP2): commonly covers velocity, acceleration,
//!   and torque-related constraints.
//! - **3rd-order** (TOPP3/COPP3): additionally models jerk-level effects.
//!
//! User-defined constraints are supported at different abstraction levels:
//! - prefer [`robot::Robot`] for physically meaningful high-level ingestion;
//! - use [`constraints::Constraints`] directly for maximum low-level flexibility.
//!
//! # Supported problem classes
//! This crate provides solvers for four major classes:
//! - TOPP2
//! - COPP2
//! - TOPP3
//! - COPP3
//!
//! # Recommended onboarding path
//! 1. Build constraints with [`robot::Robot`] (or directly with
//!    [`constraints::Constraints`] for advanced customization).
//! 2. Choose a solver family from [`solver`] based on objective/accuracy/runtime.
//! 3. Start quickly with [`prelude`] for common imports.

pub mod robot;

pub(crate) mod copp;
pub mod diag;
pub(crate) mod math;
pub mod path;

pub use crate::copp::constraints;

/// Solver entry namespace (TOPP2/TOPP3/COPP2/COPP3).
///
/// Each submodule exposes a solver family with stable public paths.
/// Read the submodule summary first, then choose by objective and runtime budget.
pub mod solver {
    /// TOPP2 reachable-set construction.
    ///
    /// Input: TOPP2 problem + reach-set options.  
    /// Output: feasible acceleration range representation along the path.  
    /// Scenario: feasibility analysis or as a precursor for TOPP2/TOPP3 pipelines.
    ///
    /// # Example
    /// ```rust, no_run
    #[doc = include_str!("../examples/reach_set2.rs")]
    /// ```
    pub mod reach_set2 {
        pub use crate::copp::copp2::stable::reach_set2::*;
        pub use crate::topp2_basic::*;
    }

    /// TOPP2 reachability-analysis solver.
    ///
    /// Input: TOPP2 problem + RA options.  
    /// Output: time-optimal feasible `a` profile under second-order constraints.  
    /// Scenario: fast and reliable TOPP2 baseline for production pipelines.
    ///
    /// # Example
    /// ```rust, no_run
    #[doc = include_str!("../examples/topp2_ra.rs")]
    /// ```
    pub mod topp2_ra {
        pub use crate::copp::copp2::stable::topp2_ra::*;
        pub use crate::topp2_basic::*;
    }

    /// COPP2 SOCP backend (Clarabel).
    ///
    /// Input: COPP2 problem + convex objective + SOCP/Clarabel options.  
    /// Output: conic-optimization based solution and conversion helpers.  
    /// Scenario: when conic formulation is preferred over DP-style solvers.
    ///
    /// # Example
    /// ```rust, no_run
    #[doc = include_str!("../examples/copp2_socp.rs")]
    /// ```
    pub mod copp2_socp {
        pub use crate::copp::clarabel_backend::{
            ClarabelOptions, ClarabelOptionsBuilder, clarabel_to_copp2_solution,
        };
        pub use crate::copp::copp2::stable::copp2_socp::*;
        pub use crate::copp2_basic::*;
    }

    /// COPP3 SOCP backend (Clarabel).
    ///
    /// Input: COPP3 problem + convex objective + SOCP/Clarabel options.  
    /// Output: conic-program solution and conversion helpers.  
    /// Scenario: third-order convex optimization via conic programming.
    ///
    /// # Example
    /// ```rust, no_run
    #[doc = include_str!("../examples/copp3_socp.rs")]
    /// ```
    pub mod copp3_socp {
        pub use crate::copp::clarabel_backend::{
            ClarabelOptions, ClarabelOptionsBuilder, clarabel_to_copp3_solution,
        };
        pub use crate::copp::copp3::stable::copp3_socp::*;
        pub use crate::copp3_basic::*;
    }

    /// TOPP3 LP backend (Clarabel).
    ///
    /// Input: TOPP3 problem + LP/Clarabel options.  
    /// Output: LP-based time-optimal solution in third-order setting.  
    /// Scenario: linear-programming formulation for TOPP3.
    ///
    /// # Example
    /// ```rust, no_run
    #[doc = include_str!("../examples/topp3_lp.rs")]
    /// ```
    pub mod topp3_lp {
        pub use crate::copp::clarabel_backend::{
            ClarabelOptions, ClarabelOptionsBuilder, clarabel_to_copp3_solution,
        };
        pub use crate::copp::copp3::stable::topp3_lp::*;
        pub use crate::topp3_basic::*;
    }

    /// TOPP3 SOCP backend (Clarabel).
    ///
    /// Input: TOPP3 problem + SOCP/Clarabel options.  
    /// Output: SOCP-based time-optimal trajectory profile.  
    /// Scenario: conic alternative to LP/RA for TOPP3.
    ///
    /// # Example
    /// ```rust, no_run
    #[doc = include_str!("../examples/topp3_socp.rs")]
    /// ```
    pub mod topp3_socp {
        pub use crate::copp::clarabel_backend::{
            ClarabelOptions, ClarabelOptionsBuilder, clarabel_to_copp3_solution,
        };
        pub use crate::copp::copp3::stable::topp3_socp::*;
        pub use crate::topp3_basic::*;
    }
}

/// Time-grid interpolation policy used when converting trajectory profiles to time-domain samples.
///
/// Import via `use copp::InterpolationMode;` — this is the single canonical path.
/// All solver submodules (`solver::topp2_ra`, `solver::topp3_lp`, etc.) accept this type
/// in their interpolation helpers (`t_to_s_topp2`, `t_to_s_topp3`, …).
pub use crate::copp::general::InterpolationMode;

mod topp2_basic {
    pub use crate::copp::copp2::stable::basic::{
        Topp2Problem, Topp2ProblemBuilder, a_to_b_topp2, s_to_t_topp2, t_to_s_topp2,
    };
}

mod copp2_basic {
    pub use crate::copp::copp2::stable::basic::{
        Copp2Problem, Copp2ProblemBuilder, a_to_b_topp2, s_to_t_topp2, t_to_s_topp2,
    };
    pub use crate::copp::objectives::CoppObjective;
}

mod topp3_basic {
    pub use crate::copp::copp3::stable::basic::{
        Topp3Problem, Topp3ProblemBuilder, s_to_t_topp3, t_to_s_topp3,
    };
}

mod copp3_basic {
    pub use crate::copp::copp3::stable::basic::{
        Copp3Problem, Copp3ProblemBuilder, s_to_t_topp3, t_to_s_topp3,
    };
    pub use crate::copp::objectives::CoppObjective;
}

/// Commonly used public imports for application code.
///
/// Input: none (import-only convenience module).  
/// Output: unified symbols for robot, constraints, solvers, and utility types.
/// Scenario: rapid prototyping and application-layer code with minimal import boilerplate.  
///
/// Typical usage: `use copp::prelude::*;`
///
/// # Import ordering (mirrors recommended onboarding path)
/// 1. **Robot & constraints** — [`robot::Robot`], [`robot::RobotBasic`], [`robot::RobotTorque`], [`constraints::Constraints`].
/// 2. **Solver builders** — Problem/options builders for each solver family.
/// 3. **Utility types** — [`InterpolationMode`], [`copp::objectives::CoppObjective`], [`diag::CoppError`], [`diag::Verbosity`].
/// 4. **Path & AD** — [`path::Path`], [`path::Jet3`], math helpers.
/// 5. **Solver submodules** — for calling solver entry functions directly.
pub mod prelude {
    // 1. Robot model traits and constraint container
    pub use crate::constraints::Constraints;
    pub use crate::robot::*;

    // 2. Solver builders (Problem builders and options builders)
    pub use crate::solver::copp2_socp::{ClarabelOptions, ClarabelOptionsBuilder};
    pub use crate::solver::copp2_socp::{Copp2Problem, Copp2ProblemBuilder};
    pub use crate::solver::copp3_socp::{Copp3Problem, Copp3ProblemBuilder};
    pub use crate::solver::topp2_ra::{ReachSet2OptionsBuilder, Topp2Problem, Topp2ProblemBuilder};
    pub use crate::solver::topp3_lp::{Topp3Problem, Topp3ProblemBuilder};

    // 3. Shared utility types
    pub use crate::InterpolationMode;
    pub use crate::copp::objectives::CoppObjective;
    pub use crate::diag::{
        CoppError, Verbosity, VerbosityOutput, set_verbosity_log_file, set_verbosity_output,
        verbosity_output,
    };

    // 4. Path building and automatic differentiation
    pub use crate::path::{
        Jet3, Parametrization, Path, PathDerivatives, SplineConfig, cos, exp, ln, powi, sin, sqrt,
    };

    // 5. Solver submodule namespaces (for calling solver entry functions)
    pub use crate::solver::{copp2_socp, copp3_socp, reach_set2, topp2_ra, topp3_lp, topp3_socp};
}
