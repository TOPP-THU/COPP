use clarabel::solver::{SolverError as ClarabelSolverError, SolverStatus as ClarabelSolverStatus};
use thiserror::Error;

/// The error type for COPP.
#[derive(Error)]
pub enum CoppError {
    /// Some error occurred in filesystem I/O operations, such as opening/creating log files.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    /// Some error occurred in the Constraint struct, such as invalid input or computation failure.
    #[error("Some error occurred in the Constraint struct: {0}")]
    ConstraintError(#[from] ConstraintError),
    /// Some error occurred in the Path struct, such as invalid input or computation failure.
    #[error("Some error occurred in the Path struct: {0}")]
    PathError(#[from] PathError),
    /// The specified optimization problem is reported infeasible by the backend solver.
    #[error("{0} reported an infeasibility: {1}")]
    Infeasible(String, String),
    /// The specified optimization problem is reported unbounded by the backend solver.
    #[error("{0} reported an unboundedness: {1}")]
    Unbounded(String, String),
    /// The solver/backend rejects the given model or data as invalid input.
    #[error("{0} reported an invalid input: {1}")]
    InvalidInput(String, String),
    /// The solver/backend rejects the provided configuration options.
    #[error("{0} reported an invalid options: {1}")]
    InvalidOptions(String, String),
    /// The Clarabel solver returned a concrete internal solver error.
    #[error("{0} reported an error in Clarabel solver: {1}")]
    ClarabelSolverError(String, #[source] ClarabelSolverError),
    /// The Clarabel solver terminated with a non-success status.
    #[error("{0} reported a failure in Clarabel solver with status {1}")]
    ClarabelSolverStatus(String, ClarabelSolverStatus),
    /// A backend-specific or uncategorized runtime error is reported.
    #[error("{0} reported an error: {1}")]
    Other(String, String),
}

/// Error type for constraint storage/query operations.
///
/// # Usage recommendation
/// For public-facing application code, prefer using [`CoppError`](crate::diag::CoppError)
/// as the unified error type.
///
/// [`ConstraintError`] is automatically converted into [`CoppError`] via
/// `From<ConstraintError> for CoppError`, so `?` can be used directly when
/// your function returns `Result<_, CoppError>`.
#[derive(Error, Debug)]
pub enum ConstraintError {
    /// Input station sequence is not strictly increasing.
    #[error("`s` must be strictly increasing; first violation at local index {index}.")]
    NonIncreasingS { index: usize },

    /// Input matrix/vector dimensions are incompatible with expected shape.
    #[error("Input dimensions do not match the expected shape.")]
    NoMatchDimensions,

    /// Input ordering contract is violated.
    #[error("Input order does not satisfy the expected contract.")]
    NoMatchOrder,

    /// Signed upper/lower bounds violate strict feasibility contract.
    #[error(
        "`{bound_name}` requires strict signed limits at every station: upper bound > 0 and lower bound < 0."
    )]
    InvalidSignedBounds { bound_name: &'static str },

    /// Requested station interval is outside currently stored constraints range.
    #[error("Requested station interval is out of bounds: idx_s={idx_s}, len={len}.")]
    OutOfSBounds { idx_s: usize, len: usize },

    /// `a` violates positivity / non-negativity preconditions.
    #[error(
        "Input `a` violates positivity requirements (must be nonnegative, and strictly positive where required)."
    )]
    NonPositiveA,

    /// Linearization floor must be strictly positive.
    #[error("Linearization floor must be strictly positive.")]
    NonPositiveLinearizationFloor,

    /// Required path-derivative data is missing in the requested interval.
    #[error(
        "Required derivative data (`q`, `dq`, `ddq`, `dddq`) is not fully available in the requested interval."
    )]
    NoGivenQInfo,

    /// Linearized jerk constraints are unavailable at the requested station.
    #[error(
        "Linearized jerk constraints are unavailable at idx_s={idx_s}; valid range is [{}, {}).",
        valid_range.0,
        valid_range.1
    )]
    LinearJerkNotAvailable {
        idx_s: usize,
        valid_range: (usize, usize),
    },

    /// Dynamic-model data has not been provided.
    #[error("Required dynamic-model information is not available.")]
    NoDynamic,

    /// Reference profile is infeasible under current constraints.
    #[error("Reference profile is infeasible under current constraints.")]
    InfeasibleReference,

    /// Requested interval is empty.
    #[error("Requested interval is empty: {start} <= idx_s < {end}.")]
    EmptyInterval { start: usize, end: usize },
}

/// Error type for path construction and path evaluation APIs.
///
/// This error is returned by path-related modules such as [`Path`](`crate::path::Path`)
/// and spline utilities when input data, parameter ranges, or numerical systems
/// are invalid.
#[derive(Error, Debug)]
pub enum PathError {
    /// Path dimension is invalid (typically zero).
    #[error("invalid dimension: {dim}")]
    InvalidDimension { dim: usize },
    /// Path parameter range is invalid (must satisfy finite `s_min < s_max`).
    #[error("invalid s range: [{s_min}, {s_max}]")]
    InvalidRange { s_min: f64, s_max: f64 },
    /// Spline order is invalid (must satisfy required minimum/order constraints).
    #[error("invalid spline order: {order}, expected >= 3")]
    InvalidOrder { order: usize },
    /// Matrix/tensor shapes are incompatible for the requested operation.
    #[error("dimension mismatch")]
    DimensionMismatch,
    /// Input `s` has invalid matrix shape (must be `1xN` or `Nx1`).
    #[error("invalid shape for parameter s: ({rows}, {cols}), expected 1xN or Nx1")]
    InvalidSShape { rows: usize, cols: usize },
    /// Waypoint sequence is too short to build a valid path.
    #[error("not enough waypoints: {n}, expected >= 2")]
    NotEnoughWaypoints { n: usize },
    /// Query parameter `s` is outside the configured valid interval.
    #[error("s out of range [{s_min}, {s_max}] at index {index}: {value}")]
    OutOfRangeS {
        s_min: f64,
        s_max: f64,
        index: usize,
        value: f64,
    },
    /// Boundary conditions are not supported for the requested spline order.
    #[error("unsupported boundary for order={order}")]
    UnsupportedBoundary { order: usize },
    /// Internal linear system is singular and cannot be solved robustly.
    #[error("singular linear system")]
    SingularSystem,
}

/// Force the debug format of ToppError to be the same as the display format, which is more concise and user-friendly.
impl std::fmt::Debug for CoppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Check whether a given s_interval is valid.
#[inline(always)]
pub(crate) fn check_s_interval_valid(
    function_name: &str,
    idx_s_start: usize,
    idx_s_final: usize,
) -> Result<(), CoppError> {
    if idx_s_final < idx_s_start + 2 {
        Err(CoppError::InvalidInput(
            function_name.into(),
            format!(
                "The final index {idx_s_final} must be at least two positions after the start index {idx_s_start} in Topp2Problem."
            ),
        ))
    } else {
        Ok(())
    }
}

/// Check the positivity of a given tolerance.
#[inline(always)]
pub(crate) fn check_abs_rel_tol(
    function_name: &str,
    abs_tol_name: &str,
    abs_tol: f64,
    rel_tol_name: &str,
    rel_tol: f64,
) -> Result<(), CoppError> {
    check_not_nan_infinite(function_name, abs_tol_name, abs_tol)?;
    check_not_nan_infinite(function_name, rel_tol_name, rel_tol)?;
    if abs_tol >= 0.0 && rel_tol >= 0.0 && (abs_tol > f64::EPSILON || rel_tol > f64::EPSILON) {
        Ok(())
    } else {
        Err(CoppError::InvalidOptions(
            function_name.into(),
            format!(
                "At least one of {abs_tol_name} = {abs_tol} and {rel_tol_name} = {rel_tol} must be strictly positive."
            ),
        ))
    }
}

#[inline(always)]
pub(crate) fn check_not_nan_infinite(
    function_name: &str,
    var_name: &str,
    var_value: f64,
) -> Result<(), CoppError> {
    if var_value.is_nan() {
        Err(CoppError::InvalidOptions(
            function_name.into(),
            format!("{var_name} = {var_value} must not be NaN",),
        ))
    } else if var_value.is_infinite() {
        Err(CoppError::InvalidOptions(
            function_name.into(),
            format!("{var_name} = {var_value} must not be infinite",),
        ))
    } else {
        Ok(())
    }
}

/// Check the non-negativity.
#[inline(always)]
pub(crate) fn check_non_negative(
    function_name: &str,
    var_name: &str,
    var_value: f64,
) -> Result<(), CoppError> {
    check_not_nan_infinite(function_name, var_name, var_value)?;
    if var_value < 0.0 {
        Err(CoppError::InvalidOptions(
            function_name.into(),
            format!("{var_name} = {var_value} must be strictly non-negative"),
        ))
    } else {
        Ok(())
    }
}

/// Check the positivity of a given tolerance.
#[inline(always)]
pub(crate) fn check_strictly_positive(
    function_name: &str,
    var_name: &str,
    var_value: f64,
) -> Result<(), CoppError> {
    check_not_nan_infinite(function_name, var_name, var_value)?;
    if var_value < f64::EPSILON {
        Err(CoppError::InvalidOptions(
            function_name.into(),
            format!("{var_name} = {var_value} must be strictly positive"),
        ))
    } else {
        Ok(())
    }
}

#[inline(always)]
pub(crate) fn check_boundary_state_copp3_valid(
    a_boundary: (f64, f64),
    b_boundary: (f64, f64),
) -> Result<(), CoppError> {
    if a_boundary.0 < 0.0 {
        return Err(CoppError::InvalidInput(
            "copp3_socp".into(),
            format!("The initial a = {} must be non-negative.", a_boundary.0),
        ));
    }
    if a_boundary.1 < 0.0 {
        return Err(CoppError::InvalidInput(
            "copp3_socp".into(),
            format!("The terminal a = {} must be non-negative.", a_boundary.1),
        ));
    }
    if a_boundary.0.abs() < f64::EPSILON {
        // If a[0]==0 but b[0]!=0, then a<0 will occur near s_start
        if b_boundary.0.abs() >= f64::EPSILON {
            return Err(CoppError::InvalidInput(
                "copp3_socp".into(),
                format!(
                    "The initial a = {} is zero, so the initial b = {} must also be zero.",
                    a_boundary.0, b_boundary.0
                ),
            ));
        }
    }
    if a_boundary.1.abs() < f64::EPSILON {
        // If a[n]==0 but b[n]!=0, then a<0 will occur near s_final
        if b_boundary.1.abs() >= f64::EPSILON {
            return Err(CoppError::InvalidInput(
                "copp3_socp".into(),
                format!(
                    "The terminal a = {} is zero, so the terminal b = {} must also be zero.",
                    a_boundary.1, b_boundary.1
                ),
            ));
        }
    }
    Ok(())
}
