//! Robot abstractions and constraint-ingestion utilities for TOPP/COPP.
//!
//! # Method identity
//! This module defines:
//! - model-side traits (`RobotBasic`, `RobotTorque`),
//! - generic wrapper [`Robot`] that owns a constraint buffer,
//! - helper trait [`UpperBound`] for broadcasting bound inputs,
//! - conversion methods from robot kinematics/dynamics constraints to
//!   first/second/third-order inequalities consumed by solvers.
//!
//! # Layering
//! - Model traits provide pure robot semantics (dimension, inverse dynamics).
//! - [`Robot`] maps user constraints (`velocity/acceleration/jerk/torque`) into
//!   [`Constraints`](crate::constraints::Constraints).
//! - Solvers read only normalized constraints, independent of concrete robot
//!   model type.
//!
//! # When to use this module
//! - For most users, prefer [`Robot`] instead of operating on
//!   [`Constraints`](crate::constraints::Constraints) directly. This enables
//!   physically meaningful high-level APIs such as [`Robot::with_axial_velocity`],
//!   [`Robot::with_axial_acceleration`], [`Robot::with_axial_jerk`], and torque constraints.
//! - `Topp*Problem` workflows only require [`RobotBasic`].
//! - `Copp*Problem` workflows require [`RobotTorque`].
//! - If no real dynamics are involved but API integration expects
//!   [`RobotTorque`], you can use `usize` as a trivial placeholder
//!   (`tau = ddq`).
//! - For physical robots, implement [`RobotTorque`] with your own inverse
//!   dynamics.
//!
//! # Feasibility contract
//! Bound pairs must satisfy strict signed limits per station:
//! - upper bound `> 0`, lower bound `< 0`.
//!   This guarantees the zero-state neighborhood remains strictly feasible after
//!   normalization.

use crate::copp::constraints::{AsInputMatrix1D, Constraints, InputMatrix};
use crate::diag::ConstraintError;
use nalgebra::{Const, DMatrix, Dyn, Matrix, ViewStorage};

/// Borrowable upper-bound input accepted by robot constraint APIs.
///
/// This trait abstracts two common user inputs:
/// - broadcast vectors `(&[f64], ncols)`;
/// - explicit matrix views `&InputMatrix`.
///
/// Implementations must expose a matrix view of shape `(dim, ncols)`.
pub trait UpperBound {
    /// Validate that input row count is compatible with robot dimension `dim`.
    fn check_valid(&self, dim: usize) -> bool;

    /// Number of station columns represented by this bound input.
    fn ncols(&self) -> usize;

    /// Borrow input as a matrix view (`dim x ncols`).
    fn as_matrix(&self) -> InputMatrix<'_>;
}

impl UpperBound for (&[f64], usize) {
    #[inline(always)]
    fn check_valid(&self, dim: usize) -> bool {
        self.0.len() == dim
    }

    #[inline(always)]
    fn ncols(&self) -> usize {
        self.1
    }

    #[inline(always)]
    fn as_matrix(&self) -> InputMatrix<'_> {
        let dim = self.0.len();
        let ncols = self.1;
        // Zero-copy broadcast.
        unsafe {
            // Construct the matrix view directly using ViewStorage::from_raw_parts.
            // Parameters:
            // - data: Pointer to the original slice.
            // - shape: (Rows: dim, Columns: ncols).
            // - stride: (Row stride: 1, Column stride: 0).
            // Setting the column stride to 0 achieves horizontal broadcasting
            // without copying data, as every column starts at the same memory address.
            let storage = ViewStorage::from_raw_parts(
                self.0.as_ptr(),
                (Dyn(dim), Dyn(ncols)),
                (Const::<1>, Dyn(0)),
            );
            Matrix::from_data(storage)
        }
    }
}

impl UpperBound for &InputMatrix<'_> {
    #[inline(always)]
    fn check_valid(&self, dim: usize) -> bool {
        self.nrows() == dim
    }

    #[inline(always)]
    fn ncols(&self) -> usize {
        (*self).ncols()
    }

    #[inline(always)]
    fn as_matrix(&self) -> InputMatrix<'_> {
        // Already a view; no conversion/allocation needed.
        self.as_view()
    }
}

/// Minimal robot metadata required by the planner.
///
/// A `usize` variable can serve as a trivial `RobotBasic` implementation representing the robot dimension, but users can also implement this trait for their own robot models.
pub trait RobotBasic {
    /// Return robot dimension / DoF.
    fn dim(&self) -> usize;
}

impl RobotBasic for usize {
    #[inline(always)]
    fn dim(&self) -> usize {
        *self
    }
}

/// User-facing robot wrapper that owns constraint storage and conversion logic.
///
/// # Design role
/// `Robot<M>` bridges robot-side physical constraints and solver-side normalized
/// inequalities. Internally it owns [`Constraints`](crate::constraints::Constraints),
/// but exposes higher-level APIs with physical semantics.
///
/// # Why prefer this over direct `Constraints`
/// For most applications, `Robot` is the recommended entry because it provides
/// domain-meaningful methods ([`Robot::with_axial_velocity`], [`Robot::with_axial_acceleration`],
/// [`Robot::with_axial_jerk`], torque constraints) and enforces common contracts.
///
/// # Trait requirements by solver family
/// - `Topp*Problem`: model type `M` only needs [`RobotBasic`].
/// - `Copp*Problem`: model type `M` must implement [`RobotTorque`].
///
/// If you do not have a real inverse-dynamics model yet, use `usize`
/// as a placeholder implementing [`RobotTorque`] (`tau = ddq`).
pub struct Robot<M: RobotBasic> {
    /// Concrete robot model implementation.
    model: M,

    /// Shared station-indexed constraint buffer used by TOPP/COPP solvers.
    pub constraints: Constraints,
}

impl<M: RobotBasic> Robot<M> {
    /// Enforce strict signed contract for upper/lower bounds.
    ///
    /// # Contract
    /// For every element in the provided matrices:
    /// - `upper > 0`
    /// - `lower < 0`
    ///
    /// # Errors
    /// Returns `ConstraintError::InvalidSignedBounds` when contract is violated.
    #[inline(always)]
    fn check_strict_signed_limits(
        upper: &InputMatrix,
        lower: &InputMatrix,
        bound_name: &'static str,
    ) -> Result<(), ConstraintError> {
        let upper_valid = upper.iter().all(|&u| u > 0.0);
        let lower_valid = lower.iter().all(|&l| l < 0.0);
        if upper_valid && lower_valid {
            Ok(())
        } else {
            Err(ConstraintError::InvalidSignedBounds { bound_name })
        }
    }

    /// Construct a robot wrapper with default constraint-buffer capacity.
    ///
    /// # Parameters
    /// - `model`: concrete robot model implementing [`RobotBasic`].
    pub fn new(model: M) -> Self {
        let dim = model.dim();
        Self {
            model,
            constraints: Constraints::new(dim),
        }
    }

    /// Construct a robot wrapper with explicit initial constraint capacity.
    ///
    /// # Parameters
    /// - `model`: concrete robot model implementing [`RobotBasic`].
    ///   Pass a `usize` value for a dimension-only placeholder that also satisfies
    ///   [`RobotTorque`] with a trivial identity dynamics (`tau = ddq`), which is
    ///   convenient for testing or applications without real inverse dynamics.
    /// - `capacity`: pre-allocated number of station columns in the internal circular
    ///   constraint buffer.  Setting this to the expected number of path samples (e.g.
    ///   `n`) avoids re-allocations during constraint ingestion.  Use [`Robot::new`]
    ///   when the size is unknown up-front.
    pub fn with_capacity(model: M, capacity: usize) -> Self {
        let dim = model.dim();
        Self {
            model,
            constraints: Constraints::with_capacity(dim, capacity),
        }
    }

    /// Get robot dimension / DoF.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.constraints.dim()
    }

    /// Append a new station segment into the internal constraint buffer.
    ///
    /// # Parameters
    /// - `s_new`: station samples accepted as 1D slice or matrix view.
    ///
    /// # Errors
    /// Propagates monotonicity/range errors from
    /// `Constraints::with_s`.
    #[inline(always)]
    pub fn with_s<T: AsInputMatrix1D + ?Sized>(
        &mut self,
        s_new: &T,
    ) -> Result<(), ConstraintError> {
        self.constraints.with_s(s_new)
    }

    /// Write path derivatives over interval starting at `idx_s`.
    ///
    /// # Parameters
    /// - `q_new`, `dq_new`, `ddq_new`: required derivative matrices.
    /// - `dddq_new`: optional third derivative matrix.
    /// - `idx_s`: global start station id.
    ///
    /// # Errors
    /// Forwards shape/range errors from
    /// `Constraints::with_q`.
    #[inline(always)]
    pub fn with_q(
        &mut self,
        q_new: &InputMatrix,
        dq_new: &InputMatrix,
        ddq_new: &InputMatrix,
        dddq_new: Option<&InputMatrix>,
        idx_s: usize,
    ) -> Result<(), ConstraintError> {
        self.constraints
            .with_q(q_new, dq_new, ddq_new, dddq_new, idx_s)
    }

    /// Add axial velocity limits on interval starting at `start_idx_s`.
    ///
    /// # Input semantics
    /// Enforces per-axis bounds:
    /// `axial_velocity_min < \dot{q} < axial_velocity_max`.
    ///
    /// # Mapping
    /// Converts velocity bounds into first-order path-speed limits on
    /// `a = \dot{s}^2`, then fuses into `amax`.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions`
    /// - `ConstraintError::OutOfSBounds`
    /// - `ConstraintError::NoGivenQInfo`
    /// - `ConstraintError::InvalidSignedBounds` when max/min signs are invalid
    pub fn with_axial_velocity<T1, T2>(
        &mut self,
        axial_velocity_max: T1,
        axial_velocity_min: T2,
        start_idx_s: usize,
    ) -> Result<(), ConstraintError>
    where
        T1: UpperBound,
        T2: UpperBound,
    {
        // Check dimensions
        if !axial_velocity_max.check_valid(self.dim())
            || !axial_velocity_min.check_valid(self.dim())
            || axial_velocity_max.ncols() != axial_velocity_min.ncols()
        {
            return Err(ConstraintError::NoMatchDimensions);
        }
        // Check bounds
        self.constraints
            .check_s_in_bounds(start_idx_s, axial_velocity_max.ncols())?;
        // Check given dq
        if !self
            .constraints
            .check_given_q(start_idx_s, start_idx_s + axial_velocity_max.ncols())
        {
            return Err(ConstraintError::NoGivenQInfo);
        }
        if axial_velocity_max.ncols() == 0 {
            return Ok(());
        }
        let axial_velocity_max = axial_velocity_max.as_matrix();
        let axial_velocity_min = axial_velocity_min.as_matrix();
        Self::check_strict_signed_limits(
            &axial_velocity_max,
            &axial_velocity_min,
            "axial_velocity",
        )?;
        // Add new axial velocity constraints
        let mut amax_new =
            DMatrix::<f64>::from_element(self.dim(), axial_velocity_max.ncols(), f64::INFINITY);
        let func = |start_idx: usize, ncols: usize, offset: usize| {
            let amax_ = self.constraints.dq.columns(start_idx, ncols).zip_zip_map(
                &axial_velocity_max.columns(offset, ncols),
                &axial_velocity_min.columns(offset, ncols),
                |dq, vmax, vmin| {
                    if dq > 0.0 {
                        (vmax / dq).powi(2)
                    } else if dq < 0.0 {
                        (vmin / dq).powi(2)
                    } else {
                        f64::INFINITY
                    }
                },
            );
            amax_new.columns_mut(offset, ncols).copy_from(&amax_);
        };
        let ncols_mat = self.constraints.capacity();
        let start_idx = self.constraints.col_at_idx_s_unchecked(start_idx_s);
        Constraints::circular_process(ncols_mat, start_idx, axial_velocity_max.ncols(), func);

        self.constraints
            .with_constraint_1order(&amax_new.as_view(), start_idx_s)?;

        Ok(())
    }

    /// Add axial acceleration limits on interval starting at `start_idx_s`.
    ///
    /// # Input semantics
    /// Enforces per-axis bounds:
    /// `axial_acceleration_min < \ddot{q} < axial_acceleration_max`.
    ///
    /// # Mapping
    /// Generates second-order rows:
    /// `acc_a * a + acc_b * b <= acc_max`,
    /// where `(a,b)` are path-speed variables.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions`
    /// - `ConstraintError::OutOfSBounds`
    /// - `ConstraintError::NoGivenQInfo`
    /// - `ConstraintError::InvalidSignedBounds`
    pub fn with_axial_acceleration<T1, T2>(
        &mut self,
        axial_acceleration_max: T1,
        axial_acceleration_min: T2,
        start_idx_s: usize,
    ) -> Result<(), ConstraintError>
    where
        T1: UpperBound,
        T2: UpperBound,
    {
        // Check dimensions
        if !axial_acceleration_max.check_valid(self.dim())
            || !axial_acceleration_min.check_valid(self.dim())
            || axial_acceleration_max.ncols() != axial_acceleration_min.ncols()
        {
            return Err(ConstraintError::NoMatchDimensions);
        }
        // Check bounds
        self.constraints
            .check_s_in_bounds(start_idx_s, axial_acceleration_max.ncols())?;
        // Check given dq, ddq
        if !self
            .constraints
            .check_given_q(start_idx_s, start_idx_s + axial_acceleration_max.ncols())
        {
            return Err(ConstraintError::NoGivenQInfo);
        }
        if axial_acceleration_max.ncols() == 0 {
            return Ok(());
        }
        let axial_acceleration_max = axial_acceleration_max.as_matrix();
        let axial_acceleration_min = axial_acceleration_min.as_matrix();
        Self::check_strict_signed_limits(
            &axial_acceleration_max,
            &axial_acceleration_min,
            "axial_acceleration",
        )?;
        // Add new axial acceleration constraints
        let mut acc_a_new = DMatrix::<f64>::zeros(self.dim(), axial_acceleration_max.ncols());
        let mut acc_b_new = DMatrix::<f64>::zeros(self.dim(), axial_acceleration_max.ncols());
        let func = |start_idx: usize, ncols: usize, offset: usize| {
            acc_a_new
                .columns_mut(offset, ncols)
                .copy_from(&self.constraints.ddq.columns(start_idx, ncols));
            acc_b_new
                .columns_mut(offset, ncols)
                .copy_from(&self.constraints.dq.columns(start_idx, ncols));
        };
        let ncols_mat = self.constraints.capacity();
        let start_idx = self.constraints.col_at_idx_s_unchecked(start_idx_s);
        Constraints::circular_process(ncols_mat, start_idx, axial_acceleration_max.ncols(), func);
        self.constraints.with_constraint_2order(
            &acc_a_new.as_view(),
            &acc_b_new.as_view(),
            &axial_acceleration_max.as_view(),
            start_idx_s,
            false,
        )?;
        self.constraints.with_constraint_2order(
            &acc_a_new.as_view(),
            &acc_b_new.as_view(),
            &axial_acceleration_min.as_view(),
            start_idx_s,
            true,
        )?;

        Ok(())
    }

    /// Add axial jerk limits on interval starting at `start_idx_s`.
    ///
    /// # Input semantics
    /// Enforces per-axis bounds:
    /// `axial_jerk_min < \dddot{q} < axial_jerk_max`.
    ///
    /// # Mapping
    /// Generates third-order rows used by TOPP3/COPP3:
    /// `sqrt(a) * (jerk_a*a + jerk_b*b + jerk_c*c + jerk_d) <= jerk_max`.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions`
    /// - `ConstraintError::OutOfSBounds`
    /// - `ConstraintError::NoGivenQInfo` (needs `q/dq/ddq/dddq`)
    /// - `ConstraintError::InvalidSignedBounds`
    pub fn with_axial_jerk<T1, T2>(
        &mut self,
        axial_jerk_max: T1,
        axial_jerk_min: T2,
        start_idx_s: usize,
    ) -> Result<(), ConstraintError>
    where
        T1: UpperBound,
        T2: UpperBound,
    {
        // Check dimensions
        if !axial_jerk_max.check_valid(self.dim())
            || !axial_jerk_min.check_valid(self.dim())
            || axial_jerk_max.ncols() != axial_jerk_min.ncols()
        {
            return Err(ConstraintError::NoMatchDimensions);
        }
        // Check bounds
        self.constraints
            .check_s_in_bounds(start_idx_s, axial_jerk_max.ncols())?;
        // Check given dq, ddq, dddq
        if !self
            .constraints
            .check_given_q(start_idx_s, start_idx_s + axial_jerk_max.ncols())
            || !self
                .constraints
                .check_given_dddq(start_idx_s, start_idx_s + axial_jerk_max.ncols())
        {
            return Err(ConstraintError::NoGivenQInfo);
        }
        if axial_jerk_max.ncols() == 0 {
            return Ok(());
        }
        let axial_jerk_max = axial_jerk_max.as_matrix();
        let axial_jerk_min = axial_jerk_min.as_matrix();
        Self::check_strict_signed_limits(&axial_jerk_max, &axial_jerk_min, "axial_jerk")?;
        // Add new axial jerk constraints
        let mut jerk_a_new = DMatrix::<f64>::zeros(self.dim(), axial_jerk_max.ncols());
        let mut jerk_b_new = DMatrix::<f64>::zeros(self.dim(), axial_jerk_max.ncols());
        let mut jerk_c_new = DMatrix::<f64>::zeros(self.dim(), axial_jerk_max.ncols());
        let jerk_d_new = DMatrix::<f64>::zeros(self.dim(), axial_jerk_max.ncols());
        let func = |start_idx: usize, ncols: usize, offset: usize| {
            jerk_a_new
                .columns_mut(offset, ncols)
                .copy_from(&self.constraints.dddq.columns(start_idx, ncols));
            jerk_b_new
                .columns_mut(offset, ncols)
                .copy_from(&self.constraints.ddq.columns(start_idx, ncols));
            jerk_c_new
                .columns_mut(offset, ncols)
                .copy_from(&self.constraints.dq.columns(start_idx, ncols));
        };
        let ncols_mat = self.constraints.capacity();
        let start_idx = self.constraints.col_at_idx_s_unchecked(start_idx_s);
        Constraints::circular_process(ncols_mat, start_idx, axial_jerk_max.ncols(), func);
        jerk_b_new.scale_mut(3.0);
        self.constraints.with_constraint_3order(
            &jerk_a_new.as_view(),
            &jerk_b_new.as_view(),
            &jerk_c_new.as_view(),
            &jerk_d_new.as_view(),
            &axial_jerk_max,
            start_idx_s,
            false,
        )?;
        self.constraints.with_constraint_3order(
            &jerk_a_new.as_view(),
            &jerk_b_new.as_view(),
            &jerk_c_new.as_view(),
            &jerk_d_new.as_view(),
            &axial_jerk_min,
            start_idx_s,
            true,
        )?;

        Ok(())
    }
}

/// Robot trait with inverse-dynamics capability.
///
/// This trait is mainly required when building COPP2/COPP3 problems with
/// torque/dynamics constraints. For TOPP-only use cases, a direct
/// [`Constraints`](crate::copp::constraints::Constraints) workflow is usually
/// enough.
///
/// A `usize` variable can serve as a trivial `RobotTorque` implementation representing a point-mass model, where `tau = ddq`. For physical robots, users should implement this trait with their own inverse dynamics.
pub trait RobotTorque: RobotBasic {
    /// Evaluate inverse dynamics.
    ///
    /// # Model
    /// `tau = M(q) * ddq + C(q, dq) * dq + g(q)`
    ///
    /// # Parameters
    /// - `q`: joint positions (`dim`).
    /// - `dq`: joint velocities (`dim`).
    /// - `ddq`: joint accelerations (`dim`).
    /// - `tau`: output required torques/forces (`dim`).
    fn inverse_dynamics(&self, q: &[f64], dq: &[f64], ddq: &[f64], tau: &mut [f64]);
}

impl<M: RobotTorque> Robot<M> {
    /// Compute torque profile from path-domain `(a,b)` samples.
    ///
    /// # Notes
    /// This is a test helper used to evaluate dynamic feasibility of a profile.
    ///
    /// # Errors
    /// Returns shape/range/data-availability errors when prerequisites are not met.
    #[cfg(test)]
    pub(crate) fn get_torque_with_ab(
        &self,
        a_profile: &[f64],
        b_profile: &[f64],
        start_idx_s: usize,
    ) -> Result<DMatrix<f64>, ConstraintError> {
        if a_profile.len() != b_profile.len() {
            return Err(ConstraintError::NoMatchDimensions);
        }
        if a_profile.is_empty() {
            return Ok(DMatrix::zeros(self.dim(), 0));
        }
        self.constraints
            .check_s_in_bounds(start_idx_s, a_profile.len())?;
        if !self
            .constraints
            .check_given_q(start_idx_s, start_idx_s + a_profile.len())
        {
            return Err(ConstraintError::NoGivenQInfo);
        }
        let (mut coeff_a, mut coeff_b, mut coeff_g) =
            self.torque_coeff(start_idx_s, a_profile.len());
        for (mut coeff_a_col, &a_curr) in coeff_a.column_iter_mut().zip(a_profile.iter()) {
            coeff_a_col.scale_mut(a_curr);
        }
        for (mut coeff_b_col, &b_curr) in coeff_b.column_iter_mut().zip(b_profile.iter()) {
            coeff_b_col.scale_mut(b_curr);
        }
        coeff_g += coeff_a;
        coeff_g += coeff_b;
        Ok(coeff_g)
    }

    /// Build affine torque coefficients in path variables `(a,b)`.
    ///
    /// # Output
    /// Returns `(coeff_a, coeff_b, coeff_g)` such that
    /// `tau = coeff_a * a + coeff_b * b + coeff_g` column-wise.
    ///
    /// # Shape
    /// Each returned matrix has shape `(dim, ncols)`.
    ///
    /// # Preconditions
    /// Caller ensures target station interval is available.
    #[allow(clippy::type_complexity)]
    pub(crate) fn torque_coeff(
        &self,
        start_idx_s: usize,
        ncols: usize,
    ) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let mut coeff_a = DMatrix::<f64>::zeros(self.dim(), ncols);
        let mut coeff_b = DMatrix::<f64>::zeros(self.dim(), ncols);
        let mut coeff_g = DMatrix::<f64>::zeros(self.dim(), ncols);
        let vec_zero_dim = vec![0.0; self.dim()];

        let func = |start_idx: usize, ncols: usize, offset: usize| {
            for (((((mut a, mut b), mut g), q), dq), ddq) in coeff_a
                .columns_mut(offset, ncols)
                .column_iter_mut()
                .zip(coeff_b.columns_mut(offset, ncols).column_iter_mut())
                .zip(coeff_g.columns_mut(offset, ncols).column_iter_mut())
                .zip(self.constraints.q.columns(start_idx, ncols).column_iter())
                .zip(self.constraints.dq.columns(start_idx, ncols).column_iter())
                .zip(self.constraints.ddq.columns(start_idx, ncols).column_iter())
            {
                let q_slice = q.as_slice();
                let dq_slice = dq.as_slice();
                let ddq_slice = ddq.as_slice();
                // tau(q, dq/dt, ddq/ddt) = M(q) * ddq/ddt + C(q, dq/dt) * dq/dt + g(q).
                // tau = M(q) * (ddq/dds * a + dq/ds * b) + C(q, dq/ds * sqrt(a)) * dq/ds * sqrt(a) + g(q)
                // tau = (M * ddq/dds + C * dq/ds) * a + M * dq/ds * b + g(q)
                // Step 1. coeff_g = g(q) = tau(q, 0, 0)
                self.model.inverse_dynamics(
                    q_slice,
                    &vec_zero_dim,
                    &vec_zero_dim,
                    g.as_mut_slice(),
                );
                // Step 2. coeff_b = M(q) * dq = tau(q, 0, dq) - g(q)
                self.model
                    .inverse_dynamics(q_slice, &vec_zero_dim, dq_slice, b.as_mut_slice());
                b.iter_mut()
                    .zip(g.iter())
                    .for_each(|(b_i, g_i)| *b_i -= *g_i);
                // Step 3. coeff_a = M(q) * ddq + C(q, dq) * dq = tau(q, dq, ddq) - g(q)
                self.model
                    .inverse_dynamics(q_slice, dq_slice, ddq_slice, a.as_mut_slice());
                a.iter_mut()
                    .zip(g.iter())
                    .for_each(|(a_i, g_i)| *a_i -= *g_i);
            }
        };
        let ncols_mat = self.constraints.capacity();
        Constraints::circular_process(ncols_mat, start_idx_s, ncols, func);
        (coeff_a, coeff_b, coeff_g)
    }

    /// Build edge-coupled affine torque coefficients over `a[k], a[k+1]`.
    ///
    /// # Output
    /// Returns `(coeff_a_curr, coeff_a_next, coeff_g)` such that
    /// `tau[k] = coeff_a_curr * a[k] + coeff_a_next * a[k+1] + coeff_g`.
    ///
    /// # Shape
    /// Each returned matrix has shape `(dim, ncols)`.
    ///
    /// # Preconditions
    /// Requires station window `[start_idx_s, start_idx_s + ncols]` to be valid.
    #[allow(clippy::type_complexity)]
    pub(crate) fn torque2_coeff_a(
        &self,
        start_idx_s: usize,
        ncols: usize,
    ) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let s = self
            .constraints
            .s_vec(start_idx_s, start_idx_s + ncols + 1)
            .expect("torque2_coeff_a: s interval must be in bounds");
        let ds_double_down = s
            .windows(2)
            .map(|s_pair| 0.5 / (s_pair[1] - s_pair[0]))
            .collect::<Vec<f64>>();

        // tau[k] = coeff_a * a[k] + coeff_b * b[k] + coeff_g
        let (mut coeff_a, mut coeff_b, coeff_g) = self.torque_coeff(start_idx_s, ncols);
        // tau[k] = coeff_a * a[k] + coeff_b * (a[k+1] - a[k]) * ds_double_down + coeff_g

        // tau[k] = coeff_a * a[k] + coeff_b * (a[k+1] - a[k]) + coeff_g
        for (mut v_b, &ds_double_down) in coeff_b.column_iter_mut().zip(ds_double_down.iter()) {
            v_b.scale_mut(ds_double_down);
        }
        // tau[k] = (coeff_a - coeff_b) * a[k] + coeff_b * a[k+1] + coeff_g

        // tau[k] = coeff_a * a[k] + coeff_b * a[k+1] + coeff_g
        coeff_a -= &coeff_b;

        (coeff_a, coeff_b, coeff_g)
    }

    /// Add axial torque limits on interval starting at `start_idx_s`.
    ///
    /// # Input semantics
    /// Enforces per-axis bounds:
    /// `axial_torque_min < tau < axial_torque_max`.
    ///
    /// # Mapping
    /// Using inverse dynamics, torque limits are transformed into second-order
    /// rows on `(a,b)` and appended to the constraint buffer.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions`
    /// - `ConstraintError::OutOfSBounds`
    /// - `ConstraintError::NoGivenQInfo`
    /// - `ConstraintError::InvalidSignedBounds`
    pub fn with_axial_torque<T1, T2>(
        &mut self,
        axial_torque_max: T1,
        axial_torque_min: T2,
        start_idx_s: usize,
    ) -> Result<(), ConstraintError>
    where
        T1: UpperBound,
        T2: UpperBound,
    {
        // Check dimensions
        if !axial_torque_max.check_valid(self.dim())
            || !axial_torque_min.check_valid(self.dim())
            || axial_torque_max.ncols() != axial_torque_min.ncols()
        {
            return Err(ConstraintError::NoMatchDimensions);
        }
        // Check bounds
        self.constraints
            .check_s_in_bounds(start_idx_s, axial_torque_max.ncols())?;
        // Check given dq
        if !self
            .constraints
            .check_given_q(start_idx_s, start_idx_s + axial_torque_max.ncols())
        {
            return Err(ConstraintError::NoGivenQInfo);
        }
        if axial_torque_max.ncols() == 0 {
            return Ok(());
        }
        let axial_torque_max = axial_torque_max.as_matrix();
        let axial_torque_min = axial_torque_min.as_matrix();
        Self::check_strict_signed_limits(&axial_torque_max, &axial_torque_min, "axial_torque")?;

        // torque_min <= tau = coeff_a * a + coeff_b * b + coeff_g <= torque_max
        let (coeff_a, coeff_b, coeff_g) = self.torque_coeff(start_idx_s, axial_torque_max.ncols());
        // coeff_a * a + coeff_b * b <= torque_max - coeff_g
        self.constraints.with_constraint_2order(
            &coeff_a.as_view(),
            &coeff_b.as_view(),
            &(axial_torque_max - &coeff_g).as_view(),
            start_idx_s,
            false,
        )?;
        // torque_min - coeff_g <= coeff_a * a + coeff_b * b
        self.constraints.with_constraint_2order(
            &coeff_a.as_view(),
            &coeff_b.as_view(),
            &(axial_torque_min - coeff_g).as_view(),
            start_idx_s,
            true,
        )?;

        Ok(())
    }
}

impl RobotTorque for usize {
    /// Evaluate inverse dynamics for point-mass model.
    ///
    /// Since `tau = ddq`, this function copies `ddq` directly into `tau`.
    #[inline(always)]
    fn inverse_dynamics(&self, _q: &[f64], _dq: &[f64], ddq: &[f64], tau: &mut [f64]) {
        tau.copy_from_slice(ddq);
    }
}
