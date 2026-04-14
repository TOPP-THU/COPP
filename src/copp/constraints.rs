//! Constraint storage and access layer for TOPP/COPP pipelines.
//!
//! # Method identity
//! This module provides a circular-buffer based constraint container shared by:
//! - **TOPP2 / COPP2** (first-order + second-order constraints),
//! - **TOPP3 / COPP3** (plus nonlinear / linearized third-order constraints).
//!
//! # Data model (math + discrete code view)
//! Continuous/discrete state definition:
//! - $a(s) = \dot{s}^2$;
//! - $b(s) = \ddot{s} = \frac{1}{2}\frac{\mathrm{d}a}{\mathrm{d}s}$;
//! - $c(s) = \frac{\dddot{s}}{\dot{s}} = \frac{\mathrm{d}b}{\mathrm{d}s}$.
//!
//! Continuous/discrete state definition at station $s_k$:
//! - $a_k = \dot{s}_k^2$;
//! - $b_k = \ddot{s}_k$;
//! - $c_k = \frac{\dddot{s}_k}{\dot{s}_k}$.
//!
//! Discrete code symbols in this module:
//! - `a[k]` corresponds to $a_k$,
//! - `b[k]` corresponds to $b_k$,
//! - `c[k]` corresponds to $c_k$.
//!
//! Continuous constraint families:
//! - first-order rows: $a(s) \le a\_{\text{max}}(s)$;
//! - second-order rows: $f\_a(s) a(s) + f\_b(s) b(s) \le f\_{\text{max}}(s)$;
//! - third-order rows: $\sqrt{a(s)}(g\_a(s) a(s) + g\_b(s) b(s) + g\_c(s) c(s) + g\_d(s)) \le g\_{\text{max}}(s)$;
//! - linearized third-order rows: $h\_a(s) a(s) + h\_b(s) b(s) + h\_c(s) c(s) \le h\_{\text{max}}(s)$.
//!
//! # API layering
//! - Public safe getters `get_*` (e.g. [`get_s`](`Constraints::get_s`), [`get_acc_constraints`](`Constraints::get_acc_constraints`), [`get_jerk_constraints`](`Constraints::get_jerk_constraints`))
//!   return `Result<_, ConstraintError>` with explicit bounds contract.
//! - Internal fast getters `*_unchecked` are `pub(crate)` and require caller-side precondition guarantees.
//!
//! # User guidance
//! - For most users, prefer [`Robot`](crate::robot::Robot) as the entry point so
//!   constraints can be expressed with physical semantics (`with_axial_velocity`,
//!   `with_axial_acceleration`, torque-related APIs).
//! - Direct manipulation of [`Constraints`] is recommended for advanced users who
//!   need maximum flexibility and custom low-level constraint composition.
//!
//! # Contract summary
//! - Public APIs validate station range before indexing.
//! - Internal unchecked APIs are for hot paths and guarded by debug assertions.
//! - Linearized jerk access requires builders to call
//!   [`build_with_linearization`](`crate::copp::copp3::formulation::Topp3ProblemBuilder::build_with_linearization`) or
//!   [`build_with_linearization`](`crate::copp::copp3::formulation::Copp3ProblemBuilder::build_with_linearization`) beforehand.
//! - For robust solver behavior, keep zero-state `a=b=c=0` strictly feasible at every station.
//!   In practice this means every active scalar RHS must stay strictly positive:
//!   `amax > 0`, `acc_max > 0`, and `jerk_max > 0` (after sign normalization).

use crate::diag::ConstraintError;
use core::f64;
use itertools::{Itertools, izip};
use nalgebra::{Const, DMatrix, DMatrixView, Dyn, Matrix, RowDVector, ViewStorage};
use std::cmp::{max, min};
use std::collections::BTreeMap;
use std::ops::Bound::{Excluded, Included, Unbounded};

use crate::copp::copp2::stable::basic::a_to_b_topp2;

/// Small numerical threshold used by feasibility and bound computations.
///
/// The value is intentionally conservative and only serves as a tolerance
/// around near-zero comparisons (for example, branch decisions on inequality
/// coefficients). It is **not** a global optimization tolerance.
const EPSILON_NUMERIC: f64 = 1E-10;

/// Constraint storage and query object used by TOPP/COPP solvers.
///
/// # Mathematical symbols (with code mapping)
/// For each path station $s_k$:
/// - $a_k = \dot{s}_k^2$ (squared path speed), mapped to code symbol `a[k]`.
/// - $b_k = \ddot{s}_k = \frac{1}{2}\frac{\mathrm{d}a}{\mathrm{d}s}(s_k)$ (path acceleration), mapped to `b[k]`.
/// - $c_k = \frac{\dddot{s}_k}{\dot{s}_k} = \frac{\mathrm{d}b}{\mathrm{d}s}(s_k)$ (normalized jerk term), mapped to `c[k]`.
///
/// # Constraint families
/// - First-order: `0 <= a[k] <= amax[k]`
/// - Second-order: `acc_a[k]*a[k] + acc_b[k]*b[k] <= acc_max[k]`
/// - Third-order (nonlinear):
///   `sqrt(a[k])*(jerk_a[k]*a[k] + jerk_b[k]*b[k] + jerk_c[k]*c[k] + jerk_d[k]) <= jerk_max[k]`
/// - Third-order (linearized):
///   `jerk_a_linear[k]*a[k] + jerk_b[k]*b[k] + jerk_c[k]*c[k] <= jerk_max_linear[k]`
///
/// # Storage model
/// All station-wise arrays are stored as circular column-major matrices. Logical
/// station index range is `[idx_s, idx_s + len)`, and logical column `i` maps to
/// physical column `(head_col + i) % capacity_col`.
///
/// # API contract
/// - `get_*` methods are safe public accessors and return `Result<_, ConstraintError>`.
/// - `*_unchecked` methods are internal fast-path helpers. Callers must satisfy
///   preconditions; debug builds assert them.
pub struct Constraints {
    /// Allocated circular-buffer capacity in **columns**.
    pub(crate) capacity_col: usize,
    /// Path dimension / degrees of freedom (`DoF`).
    dim: usize,
    /// Path station grid values (`1 x capacity_col`).
    s: DMatrix<f64>,
    /// Configuration values `q(s)` (`dim x capacity_col`).
    pub(crate) q: DMatrix<f64>,
    /// First derivative `dq/ds` (`dim x capacity_col`).
    pub(crate) dq: DMatrix<f64>,
    /// Second derivative `d2q/ds2` (`dim x capacity_col`).
    pub(crate) ddq: DMatrix<f64>,
    /// Third derivative `d3q/ds3` (`dim x capacity_col`).
    pub(crate) dddq: DMatrix<f64>,
    /// First-order upper bound `amax` (`1 x capacity_col`).
    amax: DMatrix<f64>,
    /// Second-order coefficient `acc_a`.
    acc_a: DMatrix<f64>,
    /// Second-order coefficient `acc_b`.
    acc_b: DMatrix<f64>,
    /// Second-order right-hand side bound.
    acc_max: DMatrix<f64>,
    /// Nonlinear third-order coefficient for `a`.
    jerk_a: DMatrix<f64>,
    /// Nonlinear third-order coefficient for `b`.
    jerk_b: DMatrix<f64>,
    /// Nonlinear third-order coefficient for `c`.
    jerk_c: DMatrix<f64>,
    /// Nonlinear third-order constant term.
    jerk_d: DMatrix<f64>,
    /// Nonlinear third-order right-hand side bound.
    jerk_max: DMatrix<f64>,
    /// Linearized third-order coefficient for `a`.
    jerk_a_linear: DMatrix<f64>,
    /// Linearized third-order right-hand side bound.
    jerk_max_linear: DMatrix<f64>,
    /// Piecewise-constant valid row counts for `q`, `dq`, `ddq`.
    valid_rows_q: ValidRows,
    /// Piecewise-constant valid row counts for `dddq`.
    valid_rows_dddq: ValidRows,
    /// Piecewise-constant valid row counts for second-order constraints.
    valid_rows_acc: ValidRows,
    /// Piecewise-constant valid row counts for third-order constraints.
    valid_rows_jerk: ValidRows,
    /// Valid station-id interval for linearized jerk constraints `[left, right)`.
    valid_ids_linear_jerk: (usize, usize),
    /// Physical column in circular buffer that corresponds to logical offset `0`.
    head_col: usize,
    /// Number of valid logical columns currently stored.
    len: usize,
    /// Global station id of the first logical column (`head_col`).
    idx_s: usize,
}

/// Piecewise-constant row-validity map.
///
/// - Key (`usize`): right boundary `idx_s_right` (exclusive upper station id).
/// - Value: `(idx_s_left, n_rows)` meaning stations in
///   `[idx_s_left, idx_s_right)` have exactly `n_rows` valid rows.
///
/// The map is maintained as a contiguous partition without overlaps.
type ValidRows = BTreeMap<usize, (usize, usize)>;

/// Borrowed 2D matrix view type accepted by constraint-ingestion APIs.
///
/// Internally this is a dynamic nalgebra matrix view with column stride support,
/// allowing callers to pass slices, vectors, matrix columns, or full views
/// without allocation.
pub type InputMatrix<'a> = Matrix<f64, Dyn, Dyn, ViewStorage<'a, f64, Dyn, Dyn, Const<1>, Dyn>>;

/// Conversion helper trait for 1D-like first-order inputs.
///
/// This trait normalizes different containers into a `1 x N` `InputMatrix` view.
/// It is used by APIs such as `with_s()` and `with_constraint_1order()`.
pub trait AsInputMatrix1D {
    /// Borrow input data as a `1 x N` matrix view.
    fn as_input_matrix(&self) -> InputMatrix<'_>;
}

impl AsInputMatrix1D for InputMatrix<'_> {
    fn as_input_matrix(&self) -> InputMatrix<'_> {
        *self
    }
}

impl AsInputMatrix1D for [f64] {
    fn as_input_matrix(&self) -> InputMatrix<'_> {
        DMatrixView::from_slice(self, 1, self.len())
    }
}

impl AsInputMatrix1D for Vec<f64> {
    fn as_input_matrix(&self) -> InputMatrix<'_> {
        DMatrixView::from_slice(self, 1, self.len())
    }
}

impl Constraints {
    pub const DEFAULT_CAPACITY: usize = 1000;

    /// Construct a new container with default column capacity.
    ///
    /// # Parameters
    /// - `dim`: path dimension / DoF.
    ///
    /// # Notes
    /// Equivalent to `with_capacity(dim, DEFAULT_CAPACITY)`.
    pub(crate) fn new(dim: usize) -> Self {
        Self::with_capacity(dim, Self::DEFAULT_CAPACITY)
    }

    /// Construct a new container with explicit column capacity.
    ///
    /// # Parameters
    /// - `dim`: path dimension / DoF.
    /// - `capacity_col`: initial number of allocated columns.
    ///
    /// # Initialization policy
    /// - Bound matrices are initialized to neutral values (`infinity` where applicable).
    /// - Valid-row maps start empty and are progressively populated by `with_s()`.
    pub(crate) fn with_capacity(dim: usize, capacity_col: usize) -> Self {
        Constraints {
            capacity_col,
            dim,
            s: DMatrix::<f64>::zeros(1, capacity_col),
            q: DMatrix::<f64>::zeros(dim, capacity_col),
            dq: DMatrix::<f64>::zeros(dim, capacity_col),
            ddq: DMatrix::<f64>::zeros(dim, capacity_col),
            dddq: DMatrix::<f64>::zeros(dim, capacity_col),
            amax: DMatrix::<f64>::from_element(1, capacity_col, f64::INFINITY),
            acc_a: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            acc_b: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            acc_max: DMatrix::<f64>::from_element(2 * dim, capacity_col, f64::INFINITY),
            jerk_a: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            jerk_b: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            jerk_c: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            jerk_d: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            jerk_max: DMatrix::<f64>::from_element(2 * dim, capacity_col, f64::INFINITY),
            jerk_a_linear: DMatrix::<f64>::zeros(2 * dim, capacity_col),
            jerk_max_linear: DMatrix::<f64>::from_element(2 * dim, capacity_col, f64::INFINITY),
            valid_rows_acc: BTreeMap::new(),
            valid_rows_jerk: BTreeMap::new(),
            valid_rows_q: BTreeMap::new(),
            valid_rows_dddq: BTreeMap::new(),
            valid_ids_linear_jerk: (0, 0),
            head_col: 0,
            len: 0,
            idx_s: 0,
        }
    }

    /// Calculate the physical column index in the circular buffer given a logical offset `col`.  
    /// This uses `head_col` as the memory offset (bias) for the circular buffer.
    #[inline(always)]
    fn idx(&self, col: usize) -> usize {
        (self.head_col + col) % self.capacity_col
    }

    /// Current number of logical stations stored in the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Path dimension / DoF.
    #[inline(always)]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Allocated circular-buffer capacity in columns.
    #[inline(always)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity_col
    }

    /// Whether no logical stations are currently stored.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the column index in the circular buffer for a given `idx_s` in the original path.
    ///
    /// # Preconditions
    /// Caller must guarantee `idx_s` is within `[self.idx_s, self.idx_s + self.len)`.
    #[inline(always)]
    pub(crate) fn col_at_idx_s_unchecked(&self, idx_s: usize) -> usize {
        debug_assert!(
            self.check_s_in_bounds(idx_s, 1).is_ok(),
            "col_at_idx_s_unchecked called with out-of-bounds idx_s={idx_s}"
        );
        self.idx(idx_s - self.idx_s)
    }

    /// Get the path s value at index `idx_s` without bounds checking.
    ///
    /// # Preconditions
    /// Caller must guarantee `idx_s` is within `[self.idx_s, self.idx_s + self.len)`.
    #[inline(always)]
    pub(crate) fn s_unchecked(&self, idx_s: usize) -> f64 {
        debug_assert!(
            self.check_s_in_bounds(idx_s, 1).is_ok(),
            "s_unchecked called with out-of-bounds idx_s={idx_s}"
        );
        self.s[(0, self.col_at_idx_s_unchecked(idx_s))]
    }

    /// Get station value `s[idx_s]` with bounds validation.
    ///
    /// # Errors
    /// Returns `ConstraintError::OutOfSBounds` if `idx_s` is outside
    /// `[idx_s_start(), idx_s_end())`.
    #[inline(always)]
    pub fn get_s(&self, idx_s: usize) -> Result<f64, ConstraintError> {
        self.check_s_in_bounds(idx_s, 1)?;
        Ok(self.s_unchecked(idx_s))
    }

    /// Number of rows currently allocated for first-order constraints.
    ///
    /// Normally this is `1`, but the method is intentionally generic.
    #[inline(always)]
    pub fn amax_rows(&self) -> usize {
        self.amax.nrows()
    }

    /// Number of rows currently allocated for second-order constraints.
    #[inline(always)]
    pub fn acc_rows(&self) -> usize {
        self.acc_a.nrows()
    }

    /// Number of rows currently allocated for third-order constraints.
    #[inline(always)]
    pub fn jerk_rows(&self) -> usize {
        self.jerk_a.nrows()
    }

    /// Export station values in half-open interval `[idx_s_from, idx_s_to)`.
    ///
    /// # Parameters
    /// - `idx_s_from`: global start station id (inclusive).
    /// - `idx_s_to`: global end station id (exclusive).
    ///
    /// # Returns
    /// A contiguous vector of station values with length `idx_s_to - idx_s_from`.
    ///
    /// # Errors
    /// - `ConstraintError::EmptyInterval` when `idx_s_from >= idx_s_to`.
    /// - `ConstraintError::OutOfSBounds` if the interval is outside stored data.
    pub fn s_vec(&self, idx_s_from: usize, idx_s_to: usize) -> Result<Vec<f64>, ConstraintError> {
        if idx_s_from >= idx_s_to {
            return Err(ConstraintError::EmptyInterval {
                start: idx_s_from,
                end: idx_s_to,
            });
        }
        self.check_s_in_bounds(idx_s_from, idx_s_to - idx_s_from)?;
        let s_raw_slice = self.s.as_slice();
        let start_idx = self.idx(idx_s_from - self.idx_s);
        let ncols_mat = self.s.ncols();
        let ncols_data = idx_s_to - idx_s_from;
        let mut result = vec![0.0; ncols_data];
        let func = |start_idx_: usize, ncols_: usize, offset: usize| {
            result[offset..(offset + ncols_)]
                .copy_from_slice(&s_raw_slice[start_idx_..(start_idx_ + ncols_)]);
        };
        Self::circular_process(ncols_mat, start_idx, ncols_data, func);
        Ok(result)
    }

    /// Export first-order upper bounds in `[idx_s_from, idx_s_to)`.
    ///
    /// Semantics and error behavior are identical to `s_vec()`.
    pub fn amax_vec(
        &self,
        idx_s_from: usize,
        idx_s_to: usize,
    ) -> Result<Vec<f64>, ConstraintError> {
        if idx_s_from >= idx_s_to {
            return Err(ConstraintError::EmptyInterval {
                start: idx_s_from,
                end: idx_s_to,
            });
        }
        self.check_s_in_bounds(idx_s_from, idx_s_to - idx_s_from)?;
        let amax_raw_slice = self.amax.as_slice();
        let start_idx = self.idx(idx_s_from - self.idx_s);
        let ncols_mat = self.amax.ncols();
        let ncols_data = idx_s_to - idx_s_from;
        let mut result = vec![0.0; ncols_data];
        let func = |start_idx_: usize, ncols_: usize, offset: usize| {
            result[offset..(offset + ncols_)]
                .copy_from_slice(&amax_raw_slice[start_idx_..(start_idx_ + ncols_)]);
        };
        Self::circular_process(ncols_mat, start_idx, ncols_data, func);
        Ok(result)
    }

    /// Get the amax value at index `idx_s` without bounds checking.
    ///
    /// # Preconditions
    /// Caller must guarantee `idx_s` is within `[self.idx_s, self.idx_s + self.len)`.
    #[inline(always)]
    pub(crate) fn amax_unchecked(&self, idx_s: usize) -> f64 {
        debug_assert!(
            self.check_s_in_bounds(idx_s, 1).is_ok(),
            "amax_unchecked called with out-of-bounds idx_s={idx_s}"
        );
        self.amax[(0, self.idx(idx_s - self.idx_s))]
    }

    /// Get first-order upper bound `amax[idx_s]` with bounds validation.
    ///
    /// # Errors
    /// Returns `ConstraintError::OutOfSBounds` if `idx_s` is invalid.
    #[inline(always)]
    pub fn get_amax(&self, idx_s: usize) -> Result<f64, ConstraintError> {
        self.check_s_in_bounds(idx_s, 1)?;
        Ok(self.amax_unchecked(idx_s))
    }

    /// Global start station id (inclusive) of current logical window.
    #[inline(always)]
    pub fn idx_s_start(&self) -> usize {
        self.idx_s
    }

    /// Global end station id (exclusive) of current logical window.
    #[inline(always)]
    pub fn idx_s_end(&self) -> usize {
        self.idx_s + self.len
    }

    /// Get the second-order constraints at index `idx_s` without bounds checking.
    ///
    /// # Preconditions
    /// Caller must guarantee `idx_s` is within `[self.idx_s, self.idx_s + self.len)`.
    pub(crate) fn acc_constraints_unchecked<'a>(
        &'a self,
        idx_s: usize,
    ) -> (InputMatrix<'a>, InputMatrix<'a>, InputMatrix<'a>) {
        debug_assert!(
            self.check_s_in_bounds(idx_s, 1).is_ok(),
            "acc_constraints_unchecked called with out-of-bounds idx_s={idx_s}"
        );
        // Find the valid_rows for idx_s
        let (_, &(_, valid_rows)) = self
            .valid_rows_acc
            .range((Excluded(idx_s), Unbounded))
            .next()
            .unwrap_or((&0, &(0, 0)));
        // Return the constraints
        let idx = self.idx(idx_s - self.idx_s);
        (
            self.acc_a.view((0, idx), (valid_rows, 1)),
            self.acc_b.view((0, idx), (valid_rows, 1)),
            self.acc_max.view((0, idx), (valid_rows, 1)),
        )
    }

    /// Get second-order row views at station `idx_s`.
    ///
    /// # Returns
    /// `(acc_a_col, acc_b_col, acc_max_col)` where each matrix is a `valid_rows x 1`
    /// view into internal storage.
    ///
    /// # Errors
    /// Returns `ConstraintError::OutOfSBounds` if `idx_s` is invalid.
    pub fn get_acc_constraints<'a>(
        &'a self,
        idx_s: usize,
    ) -> Result<(InputMatrix<'a>, InputMatrix<'a>, InputMatrix<'a>), ConstraintError> {
        self.check_s_in_bounds(idx_s, 1)?;
        Ok(self.acc_constraints_unchecked(idx_s))
    }

    /// Get the third-order constraints at index `idx_s` without bounds checking.
    ///
    /// # Preconditions
    /// Caller must guarantee `idx_s` is within `[self.idx_s, self.idx_s + self.len)`.
    pub(crate) fn jerk_constraints_unchecked<'a>(
        &'a self,
        idx_s: usize,
    ) -> (
        InputMatrix<'a>,
        InputMatrix<'a>,
        InputMatrix<'a>,
        InputMatrix<'a>,
        InputMatrix<'a>,
    ) {
        debug_assert!(
            self.check_s_in_bounds(idx_s, 1).is_ok(),
            "jerk_constraints_unchecked called with out-of-bounds idx_s={idx_s}"
        );
        // Find the valid_rows for idx_s
        let (_, &(_, valid_rows)) = self
            .valid_rows_jerk
            .range((Excluded(idx_s), Unbounded))
            .next()
            .unwrap_or((&0, &(0, 0)));
        // Return the constraints
        let idx = self.idx(idx_s - self.idx_s);
        (
            self.jerk_a.view((0, idx), (valid_rows, 1)),
            self.jerk_b.view((0, idx), (valid_rows, 1)),
            self.jerk_c.view((0, idx), (valid_rows, 1)),
            self.jerk_d.view((0, idx), (valid_rows, 1)),
            self.jerk_max.view((0, idx), (valid_rows, 1)),
        )
    }

    /// Get nonlinear third-order row views at station `idx_s`.
    ///
    /// # Returns
    /// `(jerk_a, jerk_b, jerk_c, jerk_d, jerk_max)`, each a `valid_rows x 1` view.
    ///
    /// # Errors
    /// Returns `ConstraintError::OutOfSBounds` if `idx_s` is invalid.
    pub fn get_jerk_constraints<'a>(
        &'a self,
        idx_s: usize,
    ) -> Result<
        (
            InputMatrix<'a>,
            InputMatrix<'a>,
            InputMatrix<'a>,
            InputMatrix<'a>,
            InputMatrix<'a>,
        ),
        ConstraintError,
    > {
        self.check_s_in_bounds(idx_s, 1)?;
        Ok(self.jerk_constraints_unchecked(idx_s))
    }

    /// Get the linearized third-order constraints at index `idx_s` without bounds checking.
    ///
    /// This accessor assumes linearized jerk constraints are prepared by
    /// `Topp3ProblemBuilders` / `Copp3ProblemBuilders` via `build_with_linearization`.
    ///
    /// # Preconditions
    /// Caller must guarantee:
    /// - `idx_s` is within `[self.idx_s, self.idx_s + self.len)`
    /// - `idx_s` is within `self.valid_ids_linear_jerk`
    pub(crate) fn jerk_linear_constraints_unchecked<'a>(
        &'a self,
        idx_s: usize,
    ) -> (
        InputMatrix<'a>,
        InputMatrix<'a>,
        InputMatrix<'a>,
        InputMatrix<'a>,
    ) {
        debug_assert!(
            self.check_s_in_bounds(idx_s, 1).is_ok(),
            "jerk_linear_constraints_unchecked called with out-of-bounds idx_s={idx_s}"
        );
        debug_assert!(
            idx_s >= self.valid_ids_linear_jerk.0 && idx_s < self.valid_ids_linear_jerk.1,
            "jerk_linear_constraints_unchecked called outside linearized range: idx_s={idx_s}, valid=[{}, {})",
            self.valid_ids_linear_jerk.0,
            self.valid_ids_linear_jerk.1
        );
        // Find the valid_rows for idx_s
        let (_, &(_, valid_rows)) = self
            .valid_rows_jerk
            .range((Excluded(idx_s), Unbounded))
            .next()
            .unwrap_or((&0, &(0, 0)));
        // Return the constraints
        let idx = self.idx(idx_s - self.idx_s);
        (
            self.jerk_a_linear.view((0, idx), (valid_rows, 1)),
            self.jerk_b.view((0, idx), (valid_rows, 1)),
            self.jerk_c.view((0, idx), (valid_rows, 1)),
            self.jerk_max_linear.view((0, idx), (valid_rows, 1)),
        )
    }

    /// Get linearized third-order row views at station `idx_s`.
    ///
    /// # Returns
    /// `(jerk_a_linear, jerk_b, jerk_c, jerk_max_linear)`, each a `valid_rows x 1`
    /// view into internal storage.
    ///
    /// # Errors
    /// - `ConstraintError::OutOfSBounds` if `idx_s` is outside current station window.
    /// - `ConstraintError::LinearJerkNotAvailable` if `idx_s` is not covered by the
    ///   latest linearization interval.
    pub fn get_jerk_linear_constraints<'a>(
        &'a self,
        idx_s: usize,
    ) -> Result<
        (
            InputMatrix<'a>,
            InputMatrix<'a>,
            InputMatrix<'a>,
            InputMatrix<'a>,
        ),
        ConstraintError,
    > {
        self.check_s_in_bounds(idx_s, 1)?;
        if idx_s < self.valid_ids_linear_jerk.0 || idx_s >= self.valid_ids_linear_jerk.1 {
            return Err(ConstraintError::LinearJerkNotAvailable {
                idx_s,
                valid_range: self.valid_ids_linear_jerk,
            });
        }
        Ok(self.jerk_linear_constraints_unchecked(idx_s))
    }

    /// Ensure buffer capacity is at least `new_capacity` columns.
    ///
    /// # Growth strategy
    /// If expansion is required, target capacity is
    /// `max(new_capacity, 2 * current_capacity + 1)`.
    ///
    /// # Guarantees
    /// - Logical order of existing data is preserved.
    /// - `head_col` is reset to `0` after re-layout.
    /// - All backing matrices (`s`, derivative buffers, and constraint buffers)
    ///   are expanded consistently.
    #[inline(always)]
    pub fn expand_capacity(&mut self, new_capacity: usize) {
        if new_capacity <= self.capacity_col {
            return;
        }
        let new_capacity = max(new_capacity, 2 * self.capacity_col + 1);

        // Copy data from old matrices to new matrices.
        let id_from = self.head_col;
        let len_copy = min(self.len, self.capacity_col - self.head_col);
        let matrices_to_move = [
            (&mut self.s, 0.0),
            (&mut self.q, 0.0),
            (&mut self.dq, 0.0),
            (&mut self.ddq, 0.0),
            (&mut self.dddq, 0.0),
            (&mut self.amax, f64::INFINITY),
            (&mut self.acc_a, 0.0),
            (&mut self.acc_b, 0.0),
            (&mut self.acc_max, f64::INFINITY),
            (&mut self.jerk_a, 0.0),
            (&mut self.jerk_b, 0.0),
            (&mut self.jerk_c, 0.0),
            (&mut self.jerk_d, 0.0),
            (&mut self.jerk_max, f64::INFINITY),
            (&mut self.jerk_a_linear, 0.0),
            (&mut self.jerk_max_linear, f64::INFINITY),
        ];
        for (mat, default_val) in matrices_to_move.into_iter() {
            let mut new_mat = DMatrix::<f64>::from_element(mat.nrows(), new_capacity, default_val);
            // Copy the data in two parts to handle the circular buffer wrap-around.
            new_mat
                .columns_mut(0, len_copy)
                .copy_from(&mat.columns(id_from, len_copy));
            // Copy the remaining part if needed.
            if len_copy < self.len {
                new_mat
                    .columns_mut(len_copy, self.len - len_copy)
                    .copy_from(&mat.columns(0, self.len - len_copy));
            }
            *mat = new_mat;
        }

        self.capacity_col = new_capacity;
        self.head_col = 0;
    }

    /// Append strictly increasing station samples to the logical tail.
    ///
    /// # Parameters
    /// - `s_new`: a `1 x N` station segment; accepted via `AsInputMatrix1D`.
    ///
    /// # Behavior
    /// - Rejects non-increasing input.
    /// - Rejects overlap with existing tail station (`s_new[0]` must be greater
    ///   than current last station when the buffer is non-empty).
    /// - Expands capacity proactively.
    /// - Appends zero-valid-row segments into all validity maps so downstream
    ///   `with_q` / `with_constraint_*` calls can progressively fill data.
    ///
    /// # Errors
    /// Returns `ConstraintError::NonIncreasingS` on monotonicity violations.
    pub(crate) fn with_s<T: AsInputMatrix1D + ?Sized>(
        &mut self,
        s_new: &T,
    ) -> Result<(), ConstraintError> {
        let s_new = s_new.as_input_matrix();
        if s_new.ncols() == 0 {
            return Ok(());
        }
        // Check if s_new[0] > self.s[last]
        if self.len > 0 && s_new[0] <= self.s[self.idx(self.len - 1)] {
            return Err(ConstraintError::NonIncreasingS { index: 0 });
        }
        // Check if s_new is strictly increasing
        let check_s_new_increasing = s_new
            .iter()
            .zip(s_new.iter().skip(1))
            .enumerate()
            .find(|(_, (prev, curr))| prev >= curr);
        // let check_s_new_increasing = s_new
        //     .as_slice()
        //     .windows(2)
        //     .enumerate()
        //     .find(|(_, w)| w[0] >= w[1]);
        if let Some((index, _)) = check_s_new_increasing {
            return Err(ConstraintError::NonIncreasingS { index });
        }
        // Expand capacity if needed
        self.expand_capacity(self.len + s_new.ncols() * 2);
        // Add new s values
        let start_idx = self.idx(self.len);
        Self::copy_from_matrix_to_cirmatrix(&mut self.s, &s_new, start_idx);
        // Update valid_rows
        Self::push_back_valid_rows(&mut self.valid_rows_q, s_new.ncols(), 0);
        Self::push_back_valid_rows(&mut self.valid_rows_dddq, s_new.ncols(), 0);
        Self::push_back_valid_rows(&mut self.valid_rows_acc, s_new.ncols(), 0);
        Self::push_back_valid_rows(&mut self.valid_rows_jerk, s_new.ncols(), 0);

        self.len += s_new.ncols();

        Ok(())
    }

    /// Copy a dense matrix block into a circular matrix starting at `start_idx`.
    ///
    /// # Parameters
    /// - `mat`: destination circular matrix.
    /// - `data`: source matrix view (`rows x cols`).
    /// - `start_idx`: destination physical start column in `mat`.
    ///
    /// # Notes
    /// - Automatically grows destination row count if needed.
    /// - Handles wrap-around by splitting into at most two contiguous writes.
    fn copy_from_matrix_to_cirmatrix(mat: &mut DMatrix<f64>, data: &InputMatrix, start_idx: usize) {
        if data.ncols() == 0 {
            return;
        }
        if data.nrows() > mat.nrows() {
            mat.resize_vertically_mut(data.nrows(), 0.0);
        }
        let ncols_mat = mat.ncols();
        let func = |start_idx: usize, ncols: usize, offset: usize| {
            mat.view_mut((0, start_idx), (data.nrows(), ncols))
                .copy_from(&data.columns(offset, ncols));
        };
        Self::circular_process(ncols_mat, start_idx, data.ncols(), func);
    }

    /// Execute a callback over a circular column interval as up to two segments.
    ///
    /// # Parameters
    /// - `ncols_mat`: total columns in circular matrix.
    /// - `start_idx`: physical start column.
    /// - `ncols_data`: logical number of columns to process.
    /// - `func`: callback invoked as `(segment_start, segment_len, source_offset)`.
    ///
    /// # Contract
    /// - Exactly one callback if no wrap occurs.
    /// - Exactly two callbacks if wrap occurs.
    /// - `source_offset` is suitable for indexing source arrays/views.
    #[inline(always)]
    pub(crate) fn circular_process<F>(
        ncols_mat: usize,
        start_idx: usize,
        ncols_data: usize,
        mut func: F,
    ) where
        F: FnMut(usize, usize, usize),
    {
        if ncols_mat - start_idx >= ncols_data {
            // The range is within the current matrix
            func(start_idx, ncols_data, 0);
        } else {
            // The range is out of the current matrix
            let len_first = ncols_mat - start_idx;
            func(start_idx, len_first, 0);
            let len_second = ncols_data - len_first;
            func(0, len_second, len_first);
        }
    }

    /// Append per-column row blocks into an existing circular matrix region.
    ///
    /// # Intended usage
    /// This helper is used by `with_constraint_2order()` / `with_constraint_3order()`
    /// after `update_valid_rows()` has already increased row counts for the target
    /// station interval.
    ///
    /// # Parameters
    /// - `mat`: destination circular matrix containing stacked rows.
    /// - `valid_rows`: updated row-partition map for the destination interval.
    /// - `data`: newly added row block to be appended in each covered station.
    /// - `start_idx`: physical destination column for `start_idx_s`.
    /// - `start_idx_s`: global station id corresponding to `start_idx`.
    /// - `is_negative`: if `true`, copied block is sign-flipped after insertion.
    ///
    /// # Layout rule
    /// Newly inserted rows are written at the bottom of each station column, i.e.
    /// row range `[num_valid_rows - data.nrows(), num_valid_rows)`.
    fn concat_from_matrix_to_cirmatrix(
        mat: &mut DMatrix<f64>,
        valid_rows: &ValidRows,
        data: &InputMatrix,
        start_idx: usize,
        start_idx_s: usize,
        is_negative: bool,
    ) {
        if data.ncols() == 0 || data.nrows() == 0 {
            return;
        }
        let ncols_mat = mat.ncols();
        for (&idx_s_right, &(idx_s_left, num_valid_rows)) in
            valid_rows.range((Excluded(start_idx_s), Included(start_idx_s + data.ncols())))
        {
            let idx_left = idx_s_left - start_idx_s;
            let idx_right = idx_s_right - start_idx_s;
            let start_idx_here = start_idx + idx_left;
            let ncols_data = idx_right - idx_left;
            if mat.nrows() < num_valid_rows {
                mat.resize_vertically_mut(num_valid_rows, 0.0);
            }
            let func = |start_idx_: usize, ncols: usize, offset: usize| {
                mat.view_mut(
                    (num_valid_rows - data.nrows(), start_idx_),
                    (data.nrows(), ncols),
                )
                .copy_from(&data.view((0, idx_left + offset), (data.nrows(), ncols)));
                if is_negative {
                    mat.view_mut(
                        (num_valid_rows - data.nrows(), start_idx_),
                        (data.nrows(), ncols),
                    )
                    .scale_mut(-1.0);
                }
            };
            Self::circular_process(ncols_mat, start_idx_here, ncols_data, func);
        }
    }

    /// Append a right-side interval with constant valid-row count.
    ///
    /// # Parameters
    /// - `valid_rows`: piecewise-constant map to update.
    /// - `n_cols`: number of appended stations.
    /// - `num_valid_rows`: row count assigned to each appended station.
    ///
    /// If the new interval has the same row count as the current tail interval,
    /// both are merged to keep the map compact.
    fn push_back_valid_rows(valid_rows: &mut ValidRows, n_cols: usize, num_valid_rows: usize) {
        if n_cols == 0 {
            return;
        }
        if valid_rows.is_empty() {
            valid_rows.insert(n_cols, (0, num_valid_rows));
            return;
        }
        let (&last_idx_right, &(last_idx_left, last_n_rows)) = valid_rows.iter().last().unwrap();
        let (idx_left, n_cols_) = if last_n_rows == num_valid_rows {
            // Remove the last entry
            valid_rows.pop_last();
            (last_idx_left, n_cols + last_idx_right - last_idx_left)
        } else {
            (last_idx_right, n_cols)
        };
        // Insert a new entry
        valid_rows.insert(idx_left + n_cols_, (idx_left, num_valid_rows));
    }

    /// Validate whether `[idx_left, idx_left + len)` is inside current station window.
    ///
    /// # Notes
    /// This check uses `checked_add` to avoid potential `usize` overflow in range-end
    /// computations. If overflow occurs, the interval is treated as out-of-bounds.
    #[inline(always)]
    pub(crate) fn check_s_in_bounds(
        &self,
        idx_left: usize,
        len: usize,
    ) -> Result<(), ConstraintError> {
        let idx_right = idx_left.checked_add(len);
        let valid_right = self.idx_s.checked_add(self.len);
        if let (Some(idx_right), Some(valid_right)) = (idx_right, valid_right)
            && idx_left >= self.idx_s
            && idx_right <= valid_right
        {
            Ok(())
        } else {
            Err(ConstraintError::OutOfSBounds {
                idx_s: idx_left,
                len,
            })
        }
    }

    /// Write path geometry derivatives on a station interval.
    ///
    /// # Parameters
    /// - `q_new`: configuration values (`dim x N`).
    /// - `dq_new`: first derivatives (`dim x N`).
    /// - `ddq_new`: second derivatives (`dim x N`).
    /// - `dddq_new`: optional third derivatives (`dim x N`).
    /// - `idx_s`: global start station id (inclusive).
    ///
    /// # Behavior
    /// - Performs shape checks and bounds checks.
    /// - Overwrites corresponding circular-buffer ranges.
    /// - Marks derivative data as fully valid (`n_rows = dim`) over the updated range.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions` on shape mismatch.
    /// - `ConstraintError::OutOfSBounds` if target interval is invalid.
    pub(crate) fn with_q(
        &mut self,
        q_new: &InputMatrix,
        dq_new: &InputMatrix,
        ddq_new: &InputMatrix,
        dddq_new: Option<&InputMatrix>,
        idx_s: usize,
    ) -> Result<(), ConstraintError> {
        // Check dimensions and bounds
        if q_new.nrows() != self.dim
            || dq_new.shape() != q_new.shape()
            || ddq_new.shape() != q_new.shape()
            || dddq_new.is_some_and(|d| d.shape() != q_new.shape())
        {
            return Err(ConstraintError::NoMatchDimensions);
        }
        self.check_s_in_bounds(idx_s, dq_new.ncols())?;
        if dq_new.ncols() == 0 {
            return Ok(());
        }
        // Add new derivatives
        let start_idx = self.idx(idx_s - self.idx_s);

        Self::copy_from_matrix_to_cirmatrix(&mut self.q, q_new, start_idx);
        Self::copy_from_matrix_to_cirmatrix(&mut self.dq, dq_new, start_idx);
        Self::copy_from_matrix_to_cirmatrix(&mut self.ddq, ddq_new, start_idx);
        // Update valid_rows_dq
        Self::update_valid_rows(
            &mut self.valid_rows_q,
            idx_s,
            idx_s + dq_new.ncols(),
            self.dim,
            ModeUpdateValidRows::SetValues,
        );
        Self::merge_valid_rows(&mut self.valid_rows_q, idx_s, idx_s + dq_new.ncols());

        if let Some(dddq_new) = dddq_new {
            Self::copy_from_matrix_to_cirmatrix(&mut self.dddq, dddq_new, start_idx);
            Self::update_valid_rows(
                &mut self.valid_rows_dddq,
                idx_s,
                idx_s + dddq_new.ncols(),
                self.dim,
                ModeUpdateValidRows::SetValues,
            );
            Self::merge_valid_rows(&mut self.valid_rows_dddq, idx_s, idx_s + dddq_new.ncols());
        }

        Ok(())
    }

    /// Add / tighten first-order bound `amax` over an interval.
    ///
    /// # Parameters
    /// - `amax_new`: candidate upper bounds as `R x N`; each column is reduced
    ///   to its minimum before being fused into storage.
    /// - `idx_s`: global start station id.
    ///
    /// # Fusion rule
    /// Stored value is updated as `self.amax = min(self.amax, amax_new_reduced)`.
    ///
    /// # Errors
    /// - `ConstraintError::OutOfSBounds` if interval is invalid.
    /// - `ConstraintError::NonPositiveA` if any reduced bound is non-positive.
    pub fn with_constraint_1order<T: AsInputMatrix1D + ?Sized>(
        &mut self,
        amax_new: &T,
        idx_s: usize,
    ) -> Result<(), ConstraintError> {
        let amax_new = amax_new.as_input_matrix();
        // Check bounds
        self.check_s_in_bounds(idx_s, amax_new.ncols())?;
        if amax_new.ncols() == 0 || amax_new.nrows() == 0 {
            return Ok(());
        }
        let amax_row: RowDVector<f64> = RowDVector::from_iterator(
            amax_new.ncols(),
            amax_new.column_iter().map(|col| col.min()),
        );
        if amax_row.min() <= 0.0 {
            return Err(ConstraintError::NonPositiveA);
        }
        // Update amax
        let start_idx = self.idx(idx_s - self.idx_s);
        let ncols_mat = self.amax.ncols();
        let ncols_data = amax_row.ncols();
        let func = |start_idx_: usize, ncols: usize, offset: usize| {
            self.amax
                .columns_mut(start_idx_, ncols)
                .iter_mut()
                .zip(amax_row.columns(offset, ncols).iter())
                .for_each(|(a_self, &a_new)| {
                    *a_self = a_self.min(a_new);
                });
        };
        Self::circular_process(ncols_mat, start_idx, ncols_data, func);

        Ok(())
    }

    /// Append second-order inequality rows over station interval starting at `idx_s`.
    ///
    /// # Model
    /// For each station column, rows satisfy:
    /// `acc_a * a + acc_b * b <= acc_max`.
    ///
    /// # Parameters
    /// - `acc_a_new`, `acc_b_new`, `acc_max_new`: same-shape matrices (`R x N`).
    /// - `idx_s`: global start station id.
    /// - `is_negative`: whether to negate inserted rows (used to build symmetric
    ///   upper/lower bounds from one physical expression).
    ///
    /// # Behavior
    /// - Increases row counts by `R` on affected stations.
    /// - Appends new rows below existing rows per station.
    /// - Merges adjacent validity intervals when row counts match.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions` for shape mismatch.
    /// - `ConstraintError::OutOfSBounds` for invalid interval.
    pub fn with_constraint_2order(
        &mut self,
        acc_a_new: &InputMatrix,
        acc_b_new: &InputMatrix,
        acc_max_new: &InputMatrix,
        idx_s: usize,
        is_negative: bool,
    ) -> Result<(), ConstraintError> {
        // Check dimensions and bounds
        if acc_a_new.shape() != acc_b_new.shape() || acc_a_new.shape() != acc_max_new.shape() {
            return Err(ConstraintError::NoMatchDimensions);
        }
        self.check_s_in_bounds(idx_s, acc_a_new.ncols())?;
        if acc_a_new.ncols() == 0 || acc_a_new.nrows() == 0 {
            return Ok(());
        }
        // Add new second-order constraints
        let start_idx = self.idx(idx_s - self.idx_s);
        Self::update_valid_rows(
            &mut self.valid_rows_acc,
            idx_s,
            idx_s + acc_a_new.ncols(),
            acc_a_new.nrows(),
            ModeUpdateValidRows::AddValues,
        );

        [
            (&mut self.acc_a, &acc_a_new),
            (&mut self.acc_b, &acc_b_new),
            (&mut self.acc_max, &acc_max_new),
        ]
        .into_iter()
        .for_each(|(mat_self, mat_new)| {
            Self::concat_from_matrix_to_cirmatrix(
                mat_self,
                &self.valid_rows_acc,
                mat_new,
                start_idx,
                idx_s,
                is_negative,
            )
        });

        Self::merge_valid_rows(&mut self.valid_rows_acc, idx_s, idx_s + acc_a_new.ncols());
        Ok(())
    }

    /// Append third-order nonlinear inequality rows over station interval.
    ///
    /// # Model
    /// `sqrt(a) * (jerk_a*a + jerk_b*b + jerk_c*c + jerk_d) <= jerk_max`
    ///
    /// # Parameters
    /// - `jerk_*_new`: same-shape matrices (`R x N`).
    /// - `idx_s`: global start station id.
    /// - `is_negative`: if `true`, inserted rows are sign-flipped.
    ///
    /// # Side effects
    /// If the inserted interval overlaps current `valid_ids_linear_jerk`, the
    /// linearization-valid interval is cleared because source nonlinear rows changed.
    ///
    /// # Errors
    /// - `ConstraintError::NoMatchDimensions` for shape mismatch.
    /// - `ConstraintError::OutOfSBounds` for invalid interval.
    #[allow(clippy::too_many_arguments)]
    pub fn with_constraint_3order(
        &mut self,
        jerk_a_new: &InputMatrix,
        jerk_b_new: &InputMatrix,
        jerk_c_new: &InputMatrix,
        jerk_d_new: &InputMatrix,
        jerk_max_new: &InputMatrix,
        idx_s: usize,
        is_negative: bool,
    ) -> Result<(), ConstraintError> {
        // Check dimensions and bounds
        if [&jerk_b_new, &jerk_c_new, &jerk_d_new, &jerk_max_new]
            .iter()
            .any(|mat| mat.shape() != jerk_a_new.shape())
        {
            return Err(ConstraintError::NoMatchDimensions);
        }
        self.check_s_in_bounds(idx_s, jerk_a_new.ncols())?;
        if jerk_a_new.ncols() == 0 || jerk_a_new.nrows() == 0 {
            return Ok(());
        }
        if self.valid_ids_linear_jerk.1 > idx_s
            && self.valid_ids_linear_jerk.0 < idx_s + jerk_a_new.ncols()
        {
            // clear the valid linear jerk constraints
            self.valid_ids_linear_jerk = (0, 0);
        }
        // Add new third-order constraints
        let start_idx = self.idx(idx_s - self.idx_s);

        Self::update_valid_rows(
            &mut self.valid_rows_jerk,
            idx_s,
            idx_s + jerk_a_new.ncols(),
            jerk_a_new.nrows(),
            ModeUpdateValidRows::AddValues,
        );

        [
            (&mut self.jerk_a, &jerk_a_new),
            (&mut self.jerk_b, &jerk_b_new),
            (&mut self.jerk_c, &jerk_c_new),
            (&mut self.jerk_d, &jerk_d_new),
            (&mut self.jerk_max, &jerk_max_new),
        ]
        .into_iter()
        .for_each(|(mat_self, mat_new)| {
            Self::concat_from_matrix_to_cirmatrix(
                mat_self,
                &self.valid_rows_jerk,
                mat_new,
                start_idx,
                idx_s,
                is_negative,
            )
        });

        Self::merge_valid_rows(&mut self.valid_rows_jerk, idx_s, idx_s + jerk_a_new.ncols());

        Ok(())
    }

    /// Linearize third-order jerk constraints around a reference profile `a_linear`.
    ///
    /// # Purpose
    /// The original third-order inequality contains a nonlinear factor `1/sqrt(a)`:
    /// `sqrt(a) * (jerk_a*a + jerk_b*b + jerk_c*c + jerk_d) <= jerk_max`, i.e.,
    /// `jerk_a*a + jerk_b*b + jerk_c*c + jerk_d <= jerk_max / sqrt(a)`.
    ///
    /// This method performs a first-order affine approximation around reference `a_linear`,
    /// and writes the result into internal linearized buffers:
    /// - `jerk_a_linear`
    /// - `jerk_max_linear`
    ///
    /// so downstream LP/SOCP/RA stages can read linear constraints through
    /// `get_jerk_linear_constraints()`.
    ///
    /// # Numerical safety
    /// - Negative `a_linear` is rejected.
    /// - `a_linear == 0` is allowed.
    /// - To avoid singularity of `1/sqrt(a)` near zero, this method uses
    ///   `1.0 / max(a_linear, a_linearization_floor).sqrt()`.
    ///
    /// # Parameters
    /// - `a_linear`: reference profile for linearization.
    /// - `start_idx_s`: global start index of `a_linear`.
    /// - `a_linearization_floor`: strictly positive denominator floor for
    ///   `1/sqrt(a)` evaluation.
    pub(crate) fn linearize_constraint_3order_with_floor(
        &mut self,
        a_linear: &[f64],
        start_idx_s: usize,
        a_linearization_floor: f64,
    ) -> Result<(), ConstraintError> {
        if a_linear.is_empty() {
            return Ok(());
        }
        self.check_s_in_bounds(start_idx_s, a_linear.len())?;
        if a_linearization_floor <= 0.0 {
            return Err(ConstraintError::NonPositiveLinearizationFloor);
        }
        if a_linear.iter().any(|&a| a < 0.0) {
            return Err(ConstraintError::NonPositiveA);
        }

        let a_linear_half = a_linear
            .iter()
            .map(|a| 1.0 / a.max(a_linearization_floor).sqrt())
            .collect_vec(); // 1 / sqrt(a_linear)
        let a_linear_one_half = a_linear_half.iter().map(|a| *a * *a * *a).collect_vec(); // 1 / (a_linear)^(3/2)

        let start_idx = self.idx(start_idx_s - self.idx_s);
        for (&idx_s_right, &(idx_s_left, num_valid_rows)) in self
            .valid_rows_jerk
            .range((Excluded(start_idx_s), Unbounded))
        {
            if idx_s_left >= start_idx_s + a_linear.len() {
                break;
            }
            let idx_left = idx_s_left.max(start_idx_s) - start_idx_s;
            let idx_right = idx_s_right.min(start_idx_s + a_linear.len()) - start_idx_s;
            let start_idx_here = start_idx + idx_left;
            let ncols_data = idx_right - idx_left;
            if self.jerk_a_linear.nrows() < num_valid_rows {
                self.jerk_a_linear
                    .resize_vertically_mut(num_valid_rows, 0.0);
                self.jerk_max_linear
                    .resize_vertically_mut(num_valid_rows, f64::INFINITY);
            }
            // jerk_a * a[k] + jerk_b * b[k] + jerk_c * c[k] + jerk_d <= jerk_max / sqrt(a[k])
            // jerk_a * a[k] + jerk_b * b[k] + jerk_c * c[k] <= jerk_max * (1.5 * a_linear_half - 0.5 * a_linear_one_half * a[k]) - jerk_d
            // (jerk_a + 0.5 * jerk_max * a_linear_one_half) * a[k] + jerk_b * b[k] + jerk_c * c[k] <= 1.5 * jerk_max * a_linear_half - jerk_d
            let func = |start_idx_: usize, ncols: usize, offset: usize| {
                // jerk_a_linear = jerk_a + 0.5 * jerk_max * a_linear_one_half
                self.jerk_a_linear
                    .view_mut((0, start_idx_), (num_valid_rows, ncols))
                    .copy_from(&self.jerk_a.view((0, start_idx_), (num_valid_rows, ncols)));
                self.jerk_a_linear
                    .column_iter_mut()
                    .zip(self.jerk_max.column_iter())
                    .skip(start_idx_)
                    .take(ncols)
                    .zip(a_linear_one_half[(idx_left + offset)..(idx_left + offset + ncols)].iter())
                    .for_each(|((mut jerk_a_linear, jerk_max), &a_lin_o_h)| {
                        jerk_a_linear.axpy(0.5 * a_lin_o_h, &jerk_max, 1.0);
                    });
                // jerk_max_linear = 1.5 * jerk_max * a_linear_half - jerk_d
                self.jerk_max_linear
                    .view_mut((0, start_idx_), (num_valid_rows, ncols))
                    .copy_from(&self.jerk_d.view((0, start_idx_), (num_valid_rows, ncols)));
                self.jerk_max_linear
                    .view_mut((0, start_idx_), (num_valid_rows, ncols))
                    .neg_mut();
                self.jerk_max_linear
                    .column_iter_mut()
                    .zip(self.jerk_max.column_iter())
                    .skip(start_idx_)
                    .take(ncols)
                    .zip(a_linear_half[(idx_left + offset)..(idx_left + offset + ncols)].iter())
                    .for_each(|((mut jerk_max_linear, jerk_max), &a_lin_h)| {
                        jerk_max_linear.axpy(1.5 * a_lin_h, &jerk_max, 1.0);
                    });
            };
            Self::circular_process(self.jerk_max.ncols(), start_idx_here, ncols_data, func);
        }

        self.valid_ids_linear_jerk = (start_idx_s, start_idx_s + a_linear.len());
        Ok(())
    }

    /// Update valid-row map on interval `[idx_from, idx_to)`.
    ///
    /// # Parameters
    /// - `valid_rows`: map to modify.
    /// - `idx_from`: inclusive start station id.
    /// - `idx_to`: exclusive end station id.
    /// - `n_rows`: row count argument applied per `mode`.
    /// - `mode`: `SetValues` to overwrite, `AddValues` to accumulate.
    ///
    /// # Invariant handling
    /// The function splits leaves at boundaries when necessary so update is exact
    /// on the requested half-open interval.
    fn update_valid_rows(
        valid_rows: &mut ValidRows,
        idx_from: usize,
        idx_to: usize,
        n_rows: usize,
        mode: ModeUpdateValidRows,
    ) {
        // Locate idx_to and split the leaf
        if let Some((&idx_right, (idx_left, n_rows_right))) = valid_rows.range_mut(idx_to..).next()
            && idx_right != idx_to
        {
            // Split the entry at idx_to
            let idx_left_old = *idx_left;
            let n_rows_old = *n_rows_right;
            *idx_left = idx_to;
            valid_rows.insert(idx_to, (idx_left_old, n_rows_old));
        }
        // Locate idx_from and split the leaf
        if let Some((_, (idx_left, n_rows_left))) =
            valid_rows.range_mut((Excluded(idx_from), Unbounded)).next()
            && *idx_left != idx_from
        {
            // Split the entry at idx_from
            let idx_left_old = *idx_left;
            let n_rows_old = *n_rows_left;
            *idx_left = idx_from;
            valid_rows.insert(idx_from, (idx_left_old, n_rows_old));
        }
        // Update entries in [idx_from, idx_to)
        let iter = valid_rows.range_mut((Excluded(idx_from), Included(idx_to)));
        match mode {
            ModeUpdateValidRows::SetValues => {
                for (_, (_, n_rows_entry)) in iter {
                    *n_rows_entry = n_rows;
                }
            }
            ModeUpdateValidRows::AddValues => {
                for (_, (_, n_rows_entry)) in iter {
                    *n_rows_entry += n_rows;
                }
            }
        }
    }

    /// Merge adjacent map leaves with identical row counts near update boundaries.
    ///
    /// This post-processing keeps `ValidRows` compact after splits and updates.
    fn merge_valid_rows(valid_rows: &mut ValidRows, idx_from: usize, idx_to: usize) {
        // Merge those near idx_from
        if let Some((&idx_right, &(idx_left, nrows))) =
            valid_rows.range((Excluded(idx_from), Unbounded)).next()
        {
            // Merge those after idx_from
            let mut idx_right = idx_right;
            while let Some((&idx_right_new, &(_, n_rows_new))) =
                valid_rows.range((Excluded(idx_right), Unbounded)).next()
            {
                if n_rows_new != nrows {
                    break;
                }
                valid_rows.remove(&idx_right);
                idx_right = idx_right_new;
            }
            if let Some((idx_left_final, _)) = valid_rows.get_mut(&idx_right) {
                *idx_left_final = idx_left;
            }

            // Merge those before idx_from
            let mut idx_left = idx_left;
            while let Some((&idx_right_new, &(idx_left_new, n_rows_new))) = valid_rows
                .range((Unbounded, Included(idx_left)))
                .next_back()
            {
                if n_rows_new != nrows {
                    break;
                }
                valid_rows.remove(&idx_right_new);
                idx_left = idx_left_new;
            }
            if let Some((idx_left_final, _)) = valid_rows.get_mut(&idx_right) {
                *idx_left_final = idx_left;
            }
        }

        // Merge those near idx_to
        if let Some((&idx_right, &(idx_left, nrows))) =
            valid_rows.range((Included(idx_to), Unbounded)).next()
        {
            // Merge those before idx_to
            let mut id_left = idx_left;
            while let Some((&idx_right_new, &(idx_left_new, n_rows_new))) =
                valid_rows.range((Unbounded, Included(id_left))).next_back()
            {
                if n_rows_new != nrows {
                    break;
                }
                valid_rows.remove(&idx_right_new);
                id_left = idx_left_new;
            }
            if let Some((idx_left_final, _)) = valid_rows.get_mut(&idx_right) {
                *idx_left_final = id_left;
            }
            // Merge those after idx_to
            let mut idx_right = idx_right;
            while let Some((&idx_right_new, &(_, n_rows_new))) =
                valid_rows.range((Excluded(idx_right), Unbounded)).next()
            {
                if n_rows_new != nrows {
                    break;
                }
                valid_rows.remove(&idx_right);
                idx_right = idx_right_new;
            }
            if let Some((idx_left_final, _)) = valid_rows.get_mut(&idx_right) {
                *idx_left_final = id_left;
            }
        }
    }

    /// Check whether `q`, `dq`, and `ddq` are fully available in `[start_idx_s, end_idx_s)`.
    ///
    /// Returns `true` only when each covered station has `n_rows == dim` in
    /// `valid_rows_q`.
    pub(crate) fn check_given_q(&self, start_idx_s: usize, end_idx_s: usize) -> bool {
        let (&end_idx_s_new, _) = self.valid_rows_q.range(end_idx_s..).next().unwrap();
        self.valid_rows_q
            .range((Excluded(start_idx_s), Included(end_idx_s_new)))
            .into_iter()
            .all(|(&_, &(_, n_rows))| n_rows == self.dim)
    }

    /// Check whether `dddq` is fully available in `[start_idx_s, end_idx_s)`.
    ///
    /// Returns `true` only when each covered station has `n_rows == dim` in
    /// `valid_rows_dddq`.
    pub(crate) fn check_given_dddq(&self, start_idx_s: usize, end_idx_s: usize) -> bool {
        let (&end_idx_s_new, _) = self.valid_rows_dddq.range(end_idx_s..).next().unwrap();
        self.valid_rows_dddq
            .range((Excluded(start_idx_s), Included(end_idx_s_new)))
            .into_iter()
            .all(|(&_, &(_, n_rows))| n_rows == self.dim)
    }

    /// Remove a prefix of logical stations from the front.
    ///
    /// # Modes
    /// - `ModePopConstraints::CutAtIdxS(cut)`: keep stations with `id >= cut`.
    /// - `ModePopConstraints::PopNCols(n)`: remove first `n` logical stations.
    ///
    /// # Notes
    /// - `amax` values in removed columns are reset to `+inf`.
    /// - Valid-row maps are trimmed and re-anchored.
    /// - `idx_s` increases and `head_col` advances accordingly.
    pub fn pop_front(&mut self, mode: ModePopConstraints) {
        // Determine ncols to pop and idx_s_cut
        let idx_s_cut = match mode {
            ModePopConstraints::CutAtIdxS(idx_s_cut) => min(idx_s_cut, self.idx_s + self.len),
            ModePopConstraints::PopNCols(n_cols) => self.idx_s + min(n_cols, self.len),
        };
        if idx_s_cut <= self.idx_s {
            return;
        }
        let ncols = idx_s_cut - self.idx_s;
        if ncols >= self.len {
            self.clear(true);
            return;
        }
        // Update amax
        let ncols_mat = self.amax.ncols();
        let start_idx = self.idx(0);
        let clear_amax = |start_idx: usize, ncols: usize, _: usize| {
            self.amax.columns_mut(start_idx, ncols).fill(f64::INFINITY)
        };
        Self::circular_process(ncols_mat, start_idx, ncols, clear_amax);
        // Update valid_rows
        let pop_front_valid_rows = |valid_rows: &mut ValidRows, key_cut: usize| {
            let tree_to_keep = valid_rows.split_off(&(key_cut + 1));
            *valid_rows = tree_to_keep;
            if let Some((&_, (idx_s_left, _))) = valid_rows.iter_mut().next() {
                *idx_s_left = key_cut;
            }
        };
        pop_front_valid_rows(&mut self.valid_rows_acc, idx_s_cut);
        pop_front_valid_rows(&mut self.valid_rows_jerk, idx_s_cut);
        pop_front_valid_rows(&mut self.valid_rows_q, idx_s_cut);
        pop_front_valid_rows(&mut self.valid_rows_dddq, idx_s_cut);
        // Update parameters
        self.head_col = self.idx(ncols);
        self.idx_s += ncols;
        self.len -= ncols;
    }

    /// Remove a suffix of logical stations from the back.
    ///
    /// # Modes
    /// - `ModePopConstraints::CutAtIdxS(cut)`: keep stations with `id < cut`.
    /// - `ModePopConstraints::PopNCols(n)`: remove last `n` logical stations.
    ///
    /// # Notes
    /// - `amax` values in removed columns are reset to `+inf`.
    /// - Valid-row maps are trimmed to new right boundary.
    /// - `idx_s` is unchanged; only `len` shrinks.
    pub fn pop_back(&mut self, mode: ModePopConstraints) {
        // Determine ncols to pop
        let idx_s_cut = match mode {
            ModePopConstraints::CutAtIdxS(idx_s_cut) => max(idx_s_cut, self.idx_s),
            ModePopConstraints::PopNCols(n_cols) => self.idx_s + self.len - min(n_cols, self.len),
        };
        if idx_s_cut >= self.idx_s + self.len {
            return;
        }
        let ncols = self.idx_s + self.len - idx_s_cut;
        if ncols >= self.len {
            self.clear(true);
            return;
        }
        // Update amax
        let ncols_mat = self.amax.ncols();
        let start_idx = self.idx(self.len - ncols);
        let clear_amax = |start_idx: usize, ncols: usize, _: usize| {
            self.amax.columns_mut(start_idx, ncols).fill(f64::INFINITY)
        };
        Self::circular_process(ncols_mat, start_idx, ncols, clear_amax);
        // Update valid_rows
        let pop_back_valid_rows = |valid_rows: &mut ValidRows, key_cut: usize| {
            valid_rows.split_off(&key_cut);
            if let Some((&idx_s_right, &(idx_s_left, n_rows))) = valid_rows.iter().last() {
                valid_rows.remove(&idx_s_right);
                valid_rows.insert(key_cut, (idx_s_left, n_rows));
            }
        };
        pop_back_valid_rows(&mut self.valid_rows_acc, idx_s_cut);
        pop_back_valid_rows(&mut self.valid_rows_jerk, idx_s_cut);
        pop_back_valid_rows(&mut self.valid_rows_q, idx_s_cut);
        pop_back_valid_rows(&mut self.valid_rows_dddq, idx_s_cut);
        // Update parameters
        self.len -= ncols;
    }

    /// Reset logical content and validity maps.
    ///
    /// # Parameters
    /// - `keep_idx_s`: when `true`, preserve current global station origin;
    ///   otherwise reset it to `0`.
    ///
    /// # Notes
    /// `amax` is reinitialized to `+∞`; other matrices are kept allocated and may
    /// retain old values outside the active logical window.
    pub fn clear(&mut self, keep_idx_s: bool) {
        self.head_col = 0;
        self.len = 0;
        if !keep_idx_s {
            self.idx_s = 0;
        }
        self.valid_rows_acc.clear();
        self.valid_rows_jerk.clear();
        self.valid_rows_q.clear();
        self.valid_rows_dddq.clear();
        self.amax.fill(f64::INFINITY);
    }

    /// Compute total row count over station interval in a `ValidRows` map.
    ///
    /// Returns:
    /// `sum_{k in [idx_from, idx_to)} valid_rows(k)`.
    fn count_rows(valid_rows: &ValidRows, idx_from: usize, idx_to: usize) -> usize {
        let mut n_rows_total: usize = 0;
        for (&idx_right, &(idx_left, n_rows)) in
            valid_rows.range((Excluded(idx_from), Included(idx_to)))
        {
            n_rows_total += (idx_right.min(idx_to) - idx_left.max(idx_from)) * n_rows;
        }
        n_rows_total
    }

    /// Total second-order row count in `[idx_from, idx_to)`.
    pub(crate) fn count_rows_acc(&self, idx_from: usize, idx_to: usize) -> usize {
        Self::count_rows(&self.valid_rows_acc, idx_from, idx_to)
    }

    /// Total third-order row count in `[idx_from, idx_to)`.
    pub(crate) fn count_rows_jerk(&self, idx_from: usize, idx_to: usize) -> usize {
        Self::count_rows(&self.valid_rows_jerk, idx_from, idx_to)
    }

    /// Overwrite first-order bounds in `[idx_from, idx_from + amax_new.len())`.
    ///
    /// # Parameters
    /// - `amax_new`: replacement values.
    /// - `idx_from`: global start station id.
    ///
    /// # Errors
    /// Returns `ConstraintError::OutOfSBounds` if target range is invalid.
    pub fn amax_substitute(
        &mut self,
        amax_new: &[f64],
        idx_from: usize,
    ) -> Result<(), ConstraintError> {
        let ncols_data = amax_new.len();
        self.check_s_in_bounds(idx_from, ncols_data)?;
        if ncols_data == 0 {
            return Ok(());
        }
        // Step 1. copy old amax
        let start_idx = self.idx(idx_from - self.idx_s);
        let ncols_mat = self.amax.ncols();
        // Step 2. substitute new amax
        let func = |start_idx_: usize, ncols: usize, offset: usize| {
            self.amax
                .columns_mut(start_idx_, ncols)
                .copy_from_slice(&amax_new[offset..(offset + ncols)]);
        };
        Self::circular_process(ncols_mat, start_idx, ncols_data, func);
        Ok(())
    }

    /// Derive equivalent scalar bounds for stationary boundary handling in TOPP3.
    ///
    /// # Purpose
    /// Condenses first/second/third-order constraints over a stationary boundary
    /// stencil into a single interval bound on one representative `a` variable.
    ///
    /// # Returns
    /// `(amax_stationary, amin_stationary)` such that:
    /// - forward (`REV = false`): `amin <= a[num_stationary] <= amax`
    /// - reverse (`REV = true`): `amin <= a[n - num_stationary] <= amax`
    ///
    /// Returns `(∞, 0)` when prerequisite station range is unavailable.
    pub(crate) fn stationary_constraint_topp3<const REV: bool>(
        &self,
        a_linear: &[f64],
        idx_s_start: usize,
        num_stationary: usize,
    ) -> (f64, f64) {
        if num_stationary == 0 {
            return (f64::INFINITY, 0.0);
        }
        let n = a_linear.len() - 1;
        let (id_start, id_a) = if REV {
            if self
                .check_s_in_bounds(idx_s_start + n - num_stationary, num_stationary + 1)
                .is_err()
            {
                return (f64::INFINITY, 0.0);
            }
            (n, n - num_stationary)
        } else {
            if self
                .check_s_in_bounds(idx_s_start, num_stationary + 1)
                .is_err()
            {
                return (f64::INFINITY, 0.0);
            }
            (0, num_stationary)
        };
        // Preconditions already verified above, so `*_unchecked` access is valid here.
        let s_start = self.s_unchecked(idx_s_start + id_start);
        let ds_stationary = self.s_unchecked(idx_s_start + id_a) - s_start;

        // ds2start_alpha is the collection of (ds_to_start=s_curr-s_start, alpha) for each stationary interval
        let map_ds2start_alpha = |i_s: (usize, &f64)| {
            // Input: (i_s: (i, &a_linear_curr))
            let s = self.s_unchecked(idx_s_start + i_s.0);
            let ds_to_start = s - s_start;
            let mut alpha = ds_to_start / ds_stationary;
            alpha *= alpha.cbrt();
            let a_linear_half = 1.0 / i_s.1.sqrt().max(EPSILON_NUMERIC);
            let a_linear_one_half = a_linear_half * a_linear_half * a_linear_half;
            (i_s.0, ds_to_start, alpha, a_linear_half, a_linear_one_half)
        };
        // (i, ds_to_start, alpha, a_linear_half, a_linear_one_half)
        let vec_prepare: Vec<(usize, f64, f64, f64, f64)> = if REV {
            a_linear
                .iter()
                .enumerate()
                .rev()
                .skip(1)
                .take(num_stationary)
                .map(map_ds2start_alpha)
                .collect()
        } else {
            a_linear
                .iter()
                .enumerate()
                .skip(1)
                .take(num_stationary)
                .map(map_ds2start_alpha)
                .collect()
        };
        let mut amax_stationary = f64::INFINITY;
        let mut amin_stationary: f64 = 0.0;
        // 1st-order constraints at the start
        vec_prepare.iter().for_each(|(i, _, alpha, _, _)| {
            let amax_curr = self.amax_unchecked(idx_s_start + i);
            if amax_curr.is_finite() {
                amax_stationary = amax_stationary.min(alpha * amax_curr);
            }
        });
        // 2nd-order constraints at the start
        vec_prepare
            .iter()
            .for_each(|(i, ds_to_start, alpha, _, _)| {
                let (acc_a, acc_b, acc_max) = self.acc_constraints_unchecked(idx_s_start + i);
                izip!(acc_a.iter(), acc_b.iter(), acc_max.iter()).for_each(
                    |(&a_coeff, &b_coeff, &amax_curr)| {
                        if amax_curr.is_finite() {
                            let coef = (a_coeff + b_coeff / (1.5 * ds_to_start)) * alpha;
                            if coef > EPSILON_NUMERIC {
                                amax_stationary = amax_stationary.min(amax_curr / coef);
                            } else if coef < -EPSILON_NUMERIC {
                                amin_stationary = amin_stationary.max(amax_curr / coef);
                            }
                        }
                    },
                )
            });
        // 3rd-order constraints at the start
        for &(i, ds_to_start, alpha, a_linear_half, a_linear_one_half) in vec_prepare.iter() {
            let (jerk_a, jerk_b, jerk_c, jerk_d, jerk_max) =
                self.jerk_constraints_unchecked(idx_s_start + i);
            for (&a_coeff, &b_coeff, &c_coeff, &d_coeff, &jmax_curr) in izip!(
                jerk_a.iter(),
                jerk_b.iter(),
                jerk_c.iter(),
                jerk_d.iter(),
                jerk_max.iter()
            ) {
                if jmax_curr.is_finite() {
                    let coeff: f64 = (a_coeff
                        + b_coeff / (1.5 * ds_to_start)
                        + c_coeff / (4.5 * ds_to_start * ds_to_start)
                        + 0.5 * jmax_curr * a_linear_one_half)
                        * alpha;
                    if coeff > EPSILON_NUMERIC {
                        amax_stationary = amax_stationary
                            .min((1.5 * jmax_curr * a_linear_half - d_coeff) / coeff);
                    } else if coeff < -EPSILON_NUMERIC {
                        amin_stationary = amin_stationary
                            .max((1.5 * jmax_curr * a_linear_half - d_coeff) / coeff);
                    }
                }
            }
        }

        (amax_stationary, amin_stationary)
    }

    /// Evaluate maximum violation magnitudes of first/second-order TOPP2 constraints.
    ///
    /// # Returns
    /// `(exceed_1order, exceed_2order)` where each value is `<= 0` when fully
    /// feasible and positive when violated.
    #[allow(dead_code)]
    pub(crate) fn exceed_topp2(&self, idx_s_start: usize, a_profile: &[f64]) -> (f64, f64) {
        let Ok(s) = self.s_vec(idx_s_start, idx_s_start + a_profile.len()) else {
            return (f64::NAN, f64::NAN);
        };
        let b_profile = a_to_b_topp2(&s, a_profile);
        let mut excced_1order = -a_profile
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
            .min(0.0);
        for (i, &a) in a_profile.iter().enumerate() {
            let idx_s = idx_s_start + i;
            let amax_curr = self.amax_unchecked(idx_s);
            if amax_curr.is_finite() {
                excced_1order = excced_1order.max(a - amax_curr);
            }
        }
        let mut excced_2order: f64 = 0.0;
        for (i, (a, &b)) in a_profile.windows(2).zip(b_profile.iter()).enumerate() {
            let idx_s = idx_s_start + i;
            let (acc_a_curr, acc_b_curr, acc_max_curr) = self.acc_constraints_unchecked(idx_s);
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_curr.iter(), acc_b_curr.iter(), acc_max_curr.iter())
            {
                if acc_max.is_finite() {
                    excced_2order = excced_2order.max(acc_a * a[0] + acc_b * b - acc_max);
                }
            }
            let (acc_a_next, acc_b_next, acc_max_next) = self.acc_constraints_unchecked(idx_s + 1);
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_next.iter(), acc_b_next.iter(), acc_max_next.iter())
            {
                if acc_max.is_finite() {
                    excced_2order = excced_2order.max(acc_a * a[1] + acc_b * b - acc_max);
                }
            }
        }

        (excced_1order, excced_2order)
    }

    /// Test-only projection of `a_ori` toward feasible profile `a_fea` for TOPP2.
    ///
    /// # Returns
    /// Interpolation factor applied to `a_ori` (in `[0, 1]` in typical cases).
    ///
    /// # Errors
    /// Returns `ConstraintError::InfeasibleReference` if provided `a_fea` itself
    /// violates active constraints.
    #[cfg(test)]
    pub(crate) fn project_to_feasible_topp2(
        &self,
        a_ori: &mut [f64],
        a_fea: &[f64],
        idx_s_start: usize,
    ) -> Result<f64, ConstraintError> {
        if a_ori.len() != a_fea.len() {
            return Err(ConstraintError::NoMatchDimensions);
        }
        if a_ori.is_empty() {
            return Ok(1.0);
        }
        if a_ori.len() == 1 {
            let a_ori = a_ori.first_mut().unwrap();
            let a_fea = a_fea.first().unwrap();
            return if (*a_ori - a_fea).abs() > EPSILON_NUMERIC {
                *a_ori = *a_fea;
                Ok(0.0)
            } else {
                return Ok(1.0);
            };
        }
        // Modify a_ori to meet the boundary conditions
        let s = self
            .s_vec(idx_s_start, idx_s_start + a_ori.len())
            .map_err(|_| ConstraintError::NoGivenQInfo)?;
        let delta_a0 = a_fea.first().unwrap() - a_ori.first().unwrap();
        let delta_af = a_fea.last().unwrap() - a_ori.last().unwrap();
        // linear interpolation
        let &s0 = s.first().unwrap();
        let ds_tol_down = 1.0 / (s.last().unwrap() - s0);
        for (a_ori, &s) in a_ori.iter_mut().zip(s.iter()) {
            let alpha = (s - s0) * ds_tol_down;
            *a_ori += alpha * delta_af + (1.0 - alpha) * delta_a0;
        }
        // Project: first-order
        let mut alpha: f64 = 1.0;
        for (i, (&a_o, &a_f)) in a_ori.iter().zip(a_fea.iter()).enumerate() {
            let idx_s = idx_s_start + i;
            let amax_curr = self.amax_unchecked(idx_s);
            if amax_curr.is_finite() {
                let exceed_ori = a_o - amax_curr;
                let exceed_fea = a_f - amax_curr;
                if exceed_fea > 0.0 {
                    crate::verbosity_log!(
                        crate::diag::Verbosity::Debug,
                        "a_ori[{i}] = {a_o}, a_fea[{i}] = {a_f}, amax_curr = {amax_curr}"
                    );
                    return Err(ConstraintError::InfeasibleReference);
                }
                if exceed_ori > 0.0 {
                    alpha = alpha.min(-exceed_fea / (exceed_ori - exceed_fea));
                }
            }
        }
        // Project: second-order
        let b_ori = a_to_b_topp2(&s, a_ori);
        let b_fea = a_to_b_topp2(&s, a_fea);
        for (i, (a_o, a_f, &b_o, &b_f)) in izip!(
            a_ori.windows(2),
            a_fea.windows(2),
            b_ori.iter(),
            b_fea.iter()
        )
        .enumerate()
        {
            let idx_s = idx_s_start + i;
            let (acc_a_curr, acc_b_curr, acc_max_curr) = self.acc_constraints_unchecked(idx_s);
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_curr.iter(), acc_b_curr.iter(), acc_max_curr.iter())
            {
                if acc_max.is_finite() {
                    let exceed_ori = acc_a * a_o[0] + acc_b * b_o - acc_max;
                    let exceed_fea = acc_a * a_f[0] + acc_b * b_f - acc_max;
                    if exceed_fea > 0.0 {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "a_ori[{}] = {}, b_ori[{}] = {}, exceed_ori={}, a_fea[{}] = {}, b_fea[{}] = {}, exceed_fea = {}",
                            i,
                            a_o[0],
                            i,
                            b_o,
                            exceed_ori,
                            i,
                            a_f[0],
                            i,
                            b_f,
                            exceed_fea
                        );
                        return Err(ConstraintError::InfeasibleReference);
                    }
                    if exceed_ori > 0.0 {
                        alpha = alpha.min(-exceed_fea / (exceed_ori - exceed_fea));
                    }
                }
            }
            let (acc_a_next, acc_b_next, acc_max_next) = self.acc_constraints_unchecked(idx_s + 1);
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_next.iter(), acc_b_next.iter(), acc_max_next.iter())
            {
                if acc_max.is_finite() {
                    let exceed_ori = acc_a * a_o[1] + acc_b * b_o - acc_max;
                    let exceed_fea = acc_a * a_f[1] + acc_b * b_f - acc_max;
                    if exceed_fea > 0.0 {
                        crate::verbosity_log!(
                            crate::diag::Verbosity::Debug,
                            "a_ori[{}] = {}, b_ori[{}] = {}, exceed_ori={}, a_fea[{}] = {}, b_fea[{}] = {}, exceed_fea = {}",
                            i + 1,
                            a_o[1],
                            i,
                            b_o,
                            exceed_ori,
                            i + 1,
                            a_f[1],
                            i,
                            b_f,
                            exceed_fea
                        );
                        return Err(ConstraintError::InfeasibleReference);
                    }
                    if exceed_ori > 0.0 {
                        alpha = alpha.min(-exceed_fea / (exceed_ori - exceed_fea));
                    }
                }
            }
        }
        alpha = 1.0 - alpha;
        for (a_o, a_f) in a_ori.iter_mut().zip(a_fea.iter()) {
            *a_o += alpha * (a_f - *a_o);
        }

        Ok(alpha)
    }

    /// Expand second-order TOPP2 constraints at edge `idx_s -> idx_s+1` into linear form.
    ///
    /// # Output format
    /// Push tuples `(coef_left, coef_right, rhs)` into `a_b`.
    /// - `REV = false`: `coef_left * a[idx_s] + coef_right * a[idx_s+1] <= rhs`
    /// - `REV = true`:  `coef_left * a[idx_s+1] + coef_right * a[idx_s] <= rhs`
    ///
    /// The function appends rows and does not clear `a_b`.
    pub(crate) fn fill_acc_topp2<const REV: bool>(
        &self,
        a_b: &mut Vec<(f64, f64, f64)>,
        idx_s: usize,
    ) {
        let ds_double_down = 0.5 / (self.s_unchecked(idx_s + 1) - self.s_unchecked(idx_s));
        let (acc_a_curr, acc_b_curr, acc_max_curr) = self.acc_constraints_unchecked(idx_s);
        let (acc_a_next, acc_b_next, acc_max_next) = self.acc_constraints_unchecked(idx_s + 1);
        if REV {
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_curr.iter(), acc_b_curr.iter(), acc_max_curr.iter())
            {
                // acc_a * a_curr + acc_b * b_curr <= acc_max
                // acc_a * a_curr + acc_b * (a_next - a_curr) * ds_double_down <= acc_max
                // acc_b * ds_double_down * a_next + (acc_a - acc_b * ds_double_down) * a_curr <= acc_max
                if acc_max.is_finite() {
                    let acc_b_scaled = acc_b * ds_double_down;
                    a_b.push((acc_b_scaled, acc_a - acc_b_scaled, acc_max));
                }
            }
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_next.iter(), acc_b_next.iter(), acc_max_next.iter())
            {
                // acc_a * a_next + acc_b * b_curr <= acc_max
                // acc_a * a_next + acc_b * (a_next - a_curr) * ds_double_down <= acc_max
                // (acc_a +acc_b * ds_double_down) * a_next - acc_b * ds_double_down *a_curr <= acc_max
                if acc_max.is_finite() {
                    let acc_b_scaled = acc_b * ds_double_down;
                    a_b.push((acc_a + acc_b_scaled, -acc_b_scaled, acc_max));
                }
            }
        } else {
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_curr.iter(), acc_b_curr.iter(), acc_max_curr.iter())
            {
                // acc_a * a_curr + acc_b * b_curr <= acc_max
                // acc_a * a_curr + acc_b * (a_next - a_curr) * ds_double_down <= acc_max
                // acc_b * ds_double_down * a_next + (acc_a - acc_b * ds_double_down) * a_curr <= acc_max
                if acc_max.is_finite() {
                    let acc_b_scaled = acc_b * ds_double_down;
                    a_b.push((acc_a - acc_b_scaled, acc_b_scaled, acc_max));
                }
            }
            for (&acc_a, &acc_b, &acc_max) in
                izip!(acc_a_next.iter(), acc_b_next.iter(), acc_max_next.iter())
            {
                // acc_a * a_next + acc_b * b_curr <= acc_max
                // acc_a * a_next + acc_b * (a_next - a_curr) * ds_double_down <= acc_max
                // (acc_a +acc_b * ds_double_down) * a_next - acc_b * ds_double_down *a_curr <= acc_max
                if acc_max.is_finite() {
                    let acc_b_scaled = acc_b * ds_double_down;
                    a_b.push((-acc_b_scaled, acc_a + acc_b_scaled, acc_max));
                }
            }
        }
    }
}

/// The mode for updating valid rows.
enum ModeUpdateValidRows {
    SetValues,
    AddValues,
}

/// The mode for popping constraints.
pub enum ModePopConstraints {
    /// `ModePopConstraints::CutAtIdxS(cut)`: keep stations with `id >= cut` or `id < cut`.
    CutAtIdxS(usize),
    /// `ModePopConstraints::PopNCols(n)`: remove first or last `n` logical stations.
    PopNCols(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::Robot;

    #[test]
    fn test_constraints() -> Result<(), ConstraintError> {
        let n: usize = 100;
        let qs = [|s: f64| s.sin(), |s: f64| s.cos()];
        let dqs = [|s: f64| s.cos(), |s: f64| -s.sin()];
        let ddqs = [|s: f64| -s.sin(), |s: f64| -s.cos()];
        let dddqs = [|s: f64| -s.cos(), |s: f64| s.sin()];
        let mut robot = Robot::with_capacity(2, 10);

        let s = DMatrix::<f64>::from_fn(1, n, |_r, c| (c as f64) * 0.1);
        let q = DMatrix::<f64>::from_fn(2, n, |r, c| qs[r]((c as f64) * 0.1));
        let dq = DMatrix::<f64>::from_fn(2, n, |r, c| dqs[r]((c as f64) * 0.1));
        let ddq = DMatrix::<f64>::from_fn(2, n, |r, c| ddqs[r]((c as f64) * 0.1));
        let dddq = DMatrix::<f64>::from_fn(2, n, |r, c| dddqs[r]((c as f64) * 0.1));
        // Test idx_s after adding s
        robot.constraints.with_s(&s.columns(0, 5))?;
        assert_eq!(robot.constraints.idx_s, 0);
        // Test adding q
        assert!(
            robot
                .constraints
                .with_q(
                    &q.as_view(),
                    &dq.as_view(),
                    &ddq.as_view(),
                    Some(&dddq.as_view()),
                    0,
                )
                .is_err()
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_q before adding q:"
        );
        println_validrows(&robot.constraints.valid_rows_q);
        println_constraints(&robot.constraints);
        robot.constraints.with_q(
            &q.columns(0, 4),
            &dq.columns(0, 4),
            &ddq.columns(0, 4),
            Some(&dddq.columns(0, 4)),
            0,
        )?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_q after adding q:"
        );
        println_validrows(&robot.constraints.valid_rows_q);
        println_constraints(&robot.constraints);
        assert_eq!(robot.constraints.valid_rows_q.len(), 2);
        // Test adding non-increasing s
        assert!(robot.constraints.with_s(&s.columns(3, 5)).is_err());
        // Test adding more s
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "capacity before adding more s: {}",
            robot.constraints.capacity_col
        );
        robot.constraints.with_s(&s.columns(5, 23))?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "capacity after adding more s: {}",
            robot.constraints.capacity_col
        );
        // Test with axial velocity constraints
        let axial_velocity_max = DMatrix::<f64>::from_element(2, 21, 1.0);
        let axial_velocity_min = DMatrix::<f64>::from_element(2, 21, -2.0);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_q after adding new s:"
        );
        println_validrows(&robot.constraints.valid_rows_q);
        assert!(
            robot
                .with_axial_velocity(
                    &axial_velocity_max.as_view(),
                    &axial_velocity_min.as_view(),
                    1,
                )
                .is_err()
        );
        robot.constraints.with_q(
            &q.columns(4, 24),
            &dq.columns(4, 24),
            &ddq.columns(4, 24),
            Some(&dddq.columns(4, 24)),
            4,
        )?;
        robot.with_axial_velocity(
            &axial_velocity_max.as_view(),
            &axial_velocity_min.as_view(),
            1,
        )?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_q after adding new q:"
        );
        println_validrows(&robot.constraints.valid_rows_q);
        println_constraints(&robot.constraints);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "amax={:?}",
            robot.constraints.amax
        );
        assert!(robot.constraints.amax[(0, 9)].is_finite());
        // Test with_axial_acceleration constraints
        let axial_acceleration_max = DMatrix::<f64>::from_element(2, 15, 0.5);
        let axial_acceleration_min = DMatrix::<f64>::from_element(2, 15, -0.5);
        robot.with_axial_acceleration(
            &axial_acceleration_max.as_view(),
            &axial_acceleration_min.as_view(),
            5,
        )?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_acc after adding axial acceleration:"
        );
        println_validrows(&robot.constraints.valid_rows_acc);
        // Test popping front constraints
        robot
            .constraints
            .pop_front(ModePopConstraints::PopNCols(10));
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "After popping front 10 cols:"
        );
        println_constraints(&robot.constraints);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "amax={:?}",
            robot.constraints.amax
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_q after popping front:"
        );
        println_validrows(&robot.constraints.valid_rows_q);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_acc after popping front:"
        );
        println_validrows(&robot.constraints.valid_rows_acc);
        assert!(!robot.constraints.amax[(0, 9)].is_finite());
        assert_eq!(robot.constraints.idx_s, 10);
        // Test CircularMatrix indexing and multi-acc-constraints
        let axial_acceleration_max = DMatrix::<f64>::from_element(2, 28, 0.5);
        let axial_acceleration_min = DMatrix::<f64>::from_element(2, 28, -0.5);
        robot.constraints.with_s(&s.columns(28, 26))?;
        robot.constraints.with_q(
            &q.columns(28, 26),
            &dq.columns(28, 26),
            &ddq.columns(28, 26),
            Some(&dddq.columns(28, 26)),
            28,
        )?;
        robot.with_axial_acceleration(
            &axial_acceleration_max.as_view(),
            &axial_acceleration_min.as_view(),
            19,
        )?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_acc after adding more axial acceleration:"
        );
        println_validrows(&robot.constraints.valid_rows_acc);
        println_constraints(&robot.constraints);
        // Test with axial velocity again
        robot.with_axial_velocity(
            &axial_velocity_max.as_view(),
            &axial_velocity_min.as_view(),
            15,
        )?;
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "amax={:?}",
            robot.constraints.amax
        );
        // Test pop back
        robot.constraints.pop_back(ModePopConstraints::PopNCols(25));
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "After popping back 15 cols:"
        );
        println_constraints(&robot.constraints);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "amax={:?}",
            robot.constraints.amax
        );
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_q after popping back:"
        );
        println_validrows(&robot.constraints.valid_rows_q);
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "valid_rows_acc after popping back:"
        );
        println_validrows(&robot.constraints.valid_rows_acc);

        Ok(())
    }

    fn println_validrows(valid_rows: &ValidRows) {
        for (k, v) in valid_rows.iter() {
            print!("  [{},{}): {}, ", v.0, k, v.1);
        }
        crate::verbosity_log!(crate::diag::Verbosity::Summary, "");
    }

    fn println_constraints(constraints: &Constraints) {
        crate::verbosity_log!(
            crate::diag::Verbosity::Summary,
            "Constraints: idx_s: {}, len: {}, capacity_col: {}, head_col: {}",
            constraints.idx_s,
            constraints.len,
            constraints.capacity_col,
            constraints.head_col
        );
    }
}
