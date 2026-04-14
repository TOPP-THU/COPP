//! Problem data models and builders for second-order path parameterization.
//!
//! # Method identity
//! This module defines validated formulation objects for:
//! - **Time-Optimal Path Parameterization (TOPP2)**,
//! - **Convex-Objective Path Parameterization (COPP2)**.
//!
//! # Discrete variables (shared notation)
//! On a station grid with closed index interval `[idx_s_start, idx_s_final]`:
//! - state profile is `a(s)=\dot{s}^2`;
//! - boundary tuple is `a_boundary = (a_start, a_final)`;
//! - station count is `s_len = idx_s_final - idx_s_start + 1`.
//!
//! # High-level pipeline
//! 1. Construct `Topp2ProblemBuilder` or `Copp2ProblemBuilder` from caller data.
//! 2. Run builder validation (index interval, bounds, objective compatibility).
//! 3. Build immutable problem objects used by DP/optimization backends.

use crate::copp::constraints::Constraints;
use crate::copp::{CoppObjective, validate_copp2_objectives};
use crate::diag::{CoppError, check_non_negative, check_s_interval_valid};
use crate::robot::robot_core::{Robot, RobotBasic, RobotTorque};

/// Formulated TOPP2 problem data.
///
/// # Fields
/// - `constraints`: path-dependent kinematic/dynamic bounds on the selected interval;
/// - `idx_s_interval`: closed station-index interval `[idx_s_start, idx_s_final]`;
/// - `a_boundary`: endpoint state tuple `(a_start, a_final)`.
pub struct Topp2Problem<'a> {
    pub(crate) constraints: &'a Constraints,
    pub(crate) idx_s_interval: (usize, usize),
    pub(crate) a_boundary: (f64, f64),
}

impl<'a> Topp2Problem<'a> {
    /// Return station count on the closed interval.
    ///
    /// For `[idx_s_start, idx_s_final]`, this returns
    /// `idx_s_final - idx_s_start + 1`.
    #[inline]
    pub fn s_len(&self) -> usize {
        self.idx_s_interval.1 - self.idx_s_interval.0 + 1
    }
}

/// Builder for [`Topp2Problem`].
pub struct Topp2ProblemBuilder<'a> {
    /// Reference to path constraints (extracted from robot at construction time).
    pub constraints: &'a Constraints,
    /// Closed station-index interval `(idx_s_start, idx_s_final)`.
    pub idx_s_interval: (usize, usize),
    /// Endpoint state tuple `(a_start, a_final)`.
    pub a_boundary: (f64, f64),
}

impl<'a> Topp2ProblemBuilder<'a> {
    /// Create a TOPP2 builder from a [`Robot`] reference.
    ///
    /// # Parameters
    /// - `robot`: robot wrapper with the trait [`RobotBasic`] whose constraint buffer defines the problem domain.
    /// - `idx_s_interval`: closed station-index interval `(idx_s_start, idx_s_final)`.
    /// - `a_boundary`: endpoint state tuple `(a_start, a_final)`.
    #[inline]
    pub fn new<M: RobotBasic>(
        robot: &'a Robot<M>,
        idx_s_interval: (usize, usize),
        a_boundary: (f64, f64),
    ) -> Self {
        Self {
            constraints: &robot.constraints,
            idx_s_interval,
            a_boundary,
        }
    }

    /// Create a TOPP2 builder with all required fields.
    ///
    /// # Parameters
    /// - `constraints`: reference to path constraints defining the problem domain.
    /// - `idx_s_interval`: closed station-index interval `(idx_s_start, idx_s_final)`.
    /// - `a_boundary`: endpoint state tuple `(a_start, a_final)`.
    #[inline]
    pub fn with_constraint(
        constraints: &'a Constraints,
        idx_s_interval: (usize, usize),
        a_boundary: (f64, f64),
    ) -> Self {
        Self {
            constraints,
            idx_s_interval,
            a_boundary,
        }
    }

    /// Build a validated [`Topp2Problem`].
    #[inline]
    pub fn build(&self) -> Result<Topp2Problem<'a>, CoppError> {
        self.validate()?;
        Ok(Topp2Problem {
            constraints: self.constraints,
            idx_s_interval: self.idx_s_interval,
            a_boundary: self.a_boundary,
        })
    }

    /// Validate builder fields and consistency.
    #[inline]
    pub fn validate(&self) -> Result<(), CoppError> {
        check_s_interval_valid(
            "Topp2ProblemBuilder",
            self.idx_s_interval.0,
            self.idx_s_interval.1,
        )?;
        self.constraints.check_s_in_bounds(
            self.idx_s_interval.0,
            self.idx_s_interval.1 - self.idx_s_interval.0 + 1,
        )?;
        check_non_negative(
            "Topp2ProblemBuilder",
            "a_start (a_boundary.0)",
            self.a_boundary.0,
        )?;
        check_non_negative(
            "Topp2ProblemBuilder",
            "a_final (a_boundary.1)",
            self.a_boundary.1,
        )?;
        Ok(())
    }
}

/// Formulated COPP2 problem data.
///
/// # Fields
/// - `robot`: robot model supplying constraints and torque-related terms;
/// - `objectives`: objective list for COPP2 optimization;
/// - `idx_s_interval`: closed station-index interval `[idx_s_start, idx_s_final]`;
/// - `a_boundary`: endpoint state tuple `(a_start, a_final)`.
pub struct Copp2Problem<'a, M: RobotTorque> {
    pub(crate) robot: &'a Robot<M>,
    pub(crate) objectives: &'a [CoppObjective<'a>],
    pub(crate) idx_s_interval: (usize, usize),
    pub(crate) a_boundary: (f64, f64),
}

impl<'a, M: RobotTorque> Copp2Problem<'a, M> {
    /// Return station count on the closed interval.
    ///
    /// For `[idx_s_start, idx_s_final]`, this returns
    /// `idx_s_final - idx_s_start + 1`.
    #[inline]
    pub fn s_len(&self) -> usize {
        self.idx_s_interval.1 - self.idx_s_interval.0 + 1
    }
}

/// Builder for [`Copp2Problem`].
pub struct Copp2ProblemBuilder<'a, M: RobotTorque> {
    /// Reference to robot model defining constraints and dynamics.
    pub robot: &'a Robot<M>,
    /// Closed station-index interval `(idx_s_start, idx_s_final)`.
    pub idx_s_interval: (usize, usize),
    /// Endpoint state tuple `(a_start, a_final)`.
    pub a_boundary: (f64, f64),
    /// Objectives for COPP2 optimization.
    pub objectives: &'a [CoppObjective<'a>],
}

impl<'a, M: RobotTorque> Copp2ProblemBuilder<'a, M> {
    /// Create a COPP2 builder with all required fields.
    #[inline]
    pub fn new(
        robot: &'a Robot<M>,
        idx_s_interval: (usize, usize),
        a_boundary: (f64, f64),
        objectives: &'a [CoppObjective<'a>],
    ) -> Self {
        Self {
            robot,
            idx_s_interval,
            a_boundary,
            objectives,
        }
    }

    /// Build a validated [`Copp2Problem`].
    #[inline]
    pub fn build(&self) -> Result<Copp2Problem<'a, M>, CoppError> {
        self.validate()?;
        Ok(Copp2Problem {
            robot: self.robot,
            idx_s_interval: self.idx_s_interval,
            a_boundary: self.a_boundary,
            objectives: self.objectives,
        })
    }

    /// Validate builder fields and objective compatibility.
    #[inline]
    pub fn validate(&self) -> Result<(), CoppError> {
        check_s_interval_valid(
            "Copp2ProblemBuilder",
            self.idx_s_interval.0,
            self.idx_s_interval.1,
        )?;
        self.robot.constraints.check_s_in_bounds(
            self.idx_s_interval.0,
            self.idx_s_interval.1 - self.idx_s_interval.0 + 1,
        )?;
        check_non_negative(
            "Copp2ProblemBuilder",
            "a_start (a_boundary.0)",
            self.a_boundary.0,
        )?;
        check_non_negative(
            "Copp2ProblemBuilder",
            "a_final (a_boundary.1)",
            self.a_boundary.1,
        )?;

        let s_len = self.idx_s_interval.1 - self.idx_s_interval.0 + 1;
        validate_copp2_objectives(
            "Copp2ProblemBuilder",
            self.objectives,
            self.robot.dim(),
            s_len,
        )?;
        Ok(())
    }
}

impl<'a, M: RobotTorque> Copp2Problem<'a, M> {
    /// Convert to the TOPP2 view that shares interval and boundary fields.
    ///
    /// This is used by internal stages that only need standard TOPP2 constraints.
    pub(crate) fn as_topp2_problem(&self) -> Topp2Problem<'a> {
        Topp2Problem {
            constraints: &self.robot.constraints,
            idx_s_interval: self.idx_s_interval,
            a_boundary: self.a_boundary,
        }
    }
}
