use crate::diag::PathError;
use crate::path::OutOfRangeMode;
use crate::path::autodiff::Jet3;
use crate::path::spline::{SplineConfig, SplinePath};
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::sync::Arc;

const EPS_RANGE: f64 = 1e-12;

pub type ParametricFn = Arc<dyn Fn(Jet3) -> Vec<Jet3> + Send + Sync>;

/// Output of path evaluation.
///
/// `dq`, `ddq`, `dddq` are `None` when the evaluation did not request them
/// (e.g. `evaluate_q` only fills `q`; `evaluate_up_to_2nd` fills `q/dq/ddq`).
#[derive(Debug)]
pub struct PathDerivatives {
    pub q: DMatrix<f64>,
    pub dq: Option<DMatrix<f64>>,
    pub ddq: Option<DMatrix<f64>>,
    pub dddq: Option<DMatrix<f64>>,
}

/// How many derivative orders to compute.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    Zero,  // q only
    Two,   // q, dq, ddq
    Three, // q, dq, ddq, dddq
}

/// Unified path abstraction over parametric and spline representations.
///
/// Construct via [`Path::from_parametric`] or [`Path::from_waypoints`], then
/// query a batch of parameter values with the `evaluate_*` family of methods.
///
/// The valid parameter domain is `[s_min, s_max]` (set at construction time).
/// Out-of-range behaviour is controlled by [`OutOfRangeMode`]: the default is to
/// return an error; it can be changed to silent clamping.
pub struct Path {
    dim: usize,
    s_min: f64,
    s_max: f64,
    out_of_range_mode: OutOfRangeMode,
    repr: PathRepr,
}

enum PathRepr {
    /// Closure-based path evaluated via third-order forward AD.
    Parametric(ParametricFn),
    /// Piecewise-polynomial path built from waypoints.
    Spline(SplinePath),
}

impl Path {
    /// Build a parametric path from an analytic closure.
    ///
    /// Derivatives up to third order are computed automatically via [`Jet3`]
    /// forward-mode AD.  The closure only needs to express `q(s)` symbolically;
    /// no manual differentiation is required.
    ///
    /// # Arguments
    /// - `q_fn`  : closure mapping scalar `s` to a `dim`-dimensional position vector
    /// - `s_min` : lower bound of the path parameter
    /// - `s_max` : upper bound of the path parameter (`s_max > s_min` required)
    ///
    /// # Errors
    /// - [`PathError::InvalidRange`]     : `s_min >= s_max` or either value is non-finite
    /// - [`PathError::InvalidDimension`] : closure returned an empty vector
    pub fn from_parametric<F>(q_fn: F, s_min: f64, s_max: f64) -> Result<Self, PathError>
    where
        F: Fn(Jet3) -> Vec<Jet3> + Send + Sync + 'static,
    {
        validate_range(s_min, s_max)?;

        let sample = q_fn(Jet3::constant((s_min + s_max) * 0.5));
        if sample.is_empty() {
            return Err(PathError::InvalidDimension { dim: 0 });
        }
        let dim = sample.len();

        Ok(Self {
            dim,
            s_min,
            s_max,
            out_of_range_mode: OutOfRangeMode::Error,
            repr: PathRepr::Parametric(Arc::new(q_fn)),
        })
    }

    /// Build a spline path by interpolating a waypoint matrix.
    ///
    /// Internally solves the Hermite spline system with an O(N) block-Thomas
    /// algorithm; all dimensions are solved in parallel.
    /// The default configuration ([`SplineConfig::default`]) uses a quintic
    /// (order-5) spline with `s ∈ [0, 1]`.
    ///
    /// # Arguments
    /// - `waypoints` : matrix of shape `(dim, n_points)`; each column is one waypoint
    /// - `cfg`       : spline configuration (order, parameter range, boundary derivatives, out-of-range mode)
    ///
    /// # Errors
    /// - [`PathError::InvalidDimension`]   : `waypoints` has zero rows
    /// - [`PathError::NotEnoughWaypoints`] : fewer than 2 columns
    /// - [`PathError::InvalidOrder`]       : `order < 3`
    /// - [`PathError::InvalidRange`]       : invalid parameter range
    /// - [`PathError::SingularSystem`]     : spline system is singular (extremely rare)
    pub fn from_waypoints(waypoints: &DMatrix<f64>, cfg: SplineConfig) -> Result<Self, PathError> {
        if waypoints.nrows() == 0 {
            return Err(PathError::InvalidDimension {
                dim: waypoints.nrows(),
            });
        }
        if waypoints.ncols() < 2 {
            return Err(PathError::NotEnoughWaypoints {
                n: waypoints.ncols(),
            });
        }
        if cfg.order < 3 {
            return Err(PathError::InvalidOrder { order: cfg.order });
        }
        validate_range(cfg.s_min, cfg.s_max)?;

        let spline = SplinePath::from_waypoints(waypoints, &cfg)?;

        Ok(Self {
            dim: waypoints.nrows(),
            s_min: spline.s_min,
            s_max: spline.s_max,
            out_of_range_mode: spline.out_of_range_mode,
            repr: PathRepr::Spline(spline),
        })
    }

    /// Returns the spatial dimension (number of joints) of the path.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the valid parameter range `(s_min, s_max)`.
    #[inline(always)]
    pub fn s_range(&self) -> (f64, f64) {
        (self.s_min, self.s_max)
    }

    /// Evaluate position `q` only at the query points (cheapest; no derivatives).
    ///
    /// # Arguments
    /// - `s` : one-dimensional parameter samples (length `N`)
    ///
    /// # Returns
    /// [`PathDerivatives`] with `dq / ddq / dddq` all `None`;
    /// `q` has shape `(dim, N)`.
    ///
    /// # Errors
    /// - [`PathError::OutOfRangeS`] : a query value is out of range (only in `Error` mode)
    pub fn evaluate_q(&self, s: &[f64]) -> Result<PathDerivatives, PathError> {
        self.evaluate_impl(s, Order::Zero)
    }

    /// Evaluate position, velocity, and acceleration (`q`, `dq`, `ddq`); jerk is not computed.
    ///
    /// # Arguments
    /// - `s` : one-dimensional parameter samples (length `N`)
    ///
    /// # Returns
    /// [`PathDerivatives`] with `dddq = None`;
    /// `q / dq / ddq` each have shape `(dim, N)`.
    pub fn evaluate_up_to_2nd(&self, s: &[f64]) -> Result<PathDerivatives, PathError> {
        self.evaluate_impl(s, Order::Two)
    }

    /// Evaluate position and all three derivative orders (`q`, `dq`, `ddq`, `dddq`).
    ///
    /// This is the most expensive evaluation method.  If jerk is not needed,
    /// prefer [`evaluate_up_to_2nd`].
    ///
    /// # Arguments
    /// - `s` : one-dimensional parameter samples (length `N`)
    ///
    /// # Returns
    /// [`PathDerivatives`] with all four fields populated; each matrix has shape `(dim, N)`.
    ///
    /// # Errors
    /// - [`PathError::OutOfRangeS`] : a query value is out of range
    ///
    /// # Example
    /// ```rust
    /// # use copp::path::{Path, sin, cos};
    /// # use copp::path::autodiff::Jet3;
    /// let path = Path::from_parametric(
    ///     |s: Jet3| vec![sin(s), cos(s)],
    ///     0.0, 1.0,
    /// ).unwrap();
    ///
    /// let s = [0.0, 0.25, 0.5, 0.75, 1.0];
    /// let out = path.evaluate_up_to_3rd(&s).unwrap();
    ///
    /// let dq   = out.dq.as_ref().unwrap();
    /// let dddq = out.dddq.as_ref().unwrap();
    /// // dim 0 is sin(s); its first derivative is cos(s), evaluated at s[0]=0.0
    /// assert!((dq[(0, 0)] - 0.0_f64.cos()).abs() < 1e-10);
    /// // dim 1 is cos(s); its third derivative is sin(s), evaluated at s[0]=0.0
    /// assert!((dddq[(1, 0)] - 0.0_f64.sin()).abs() < 1e-10);
    /// ```
    ///
    /// [`evaluate_up_to_2nd`]: Path::evaluate_up_to_2nd
    pub fn evaluate_up_to_3rd(&self, s: &[f64]) -> Result<PathDerivatives, PathError> {
        self.evaluate_impl(s, Order::Three)
    }

    // ── internal ─────────────────────────────────────────────────────────────

    fn evaluate_impl(&self, s: &[f64], order: Order) -> Result<PathDerivatives, PathError> {
        let n = s.len();
        let dim = self.dim;

        // Allocate output buffers; skip higher-order buffers when not needed.
        let mut q = vec![0.0f64; dim * n];
        let mut dq = (order != Order::Zero).then(|| vec![0.0f64; dim * n]);
        let mut ddq = (order != Order::Zero).then(|| vec![0.0f64; dim * n]);
        let mut dddq = (order == Order::Three).then(|| vec![0.0f64; dim * n]);

        match &self.repr {
            PathRepr::Parametric(eval_fn) => {
                eval_parametric(
                    eval_fn,
                    self,
                    dim,
                    (s, &mut q, &mut dq, &mut ddq, &mut dddq),
                )?;
            }
            PathRepr::Spline(spline) => {
                eval_spline(spline, self, dim, (s, &mut q, &mut dq, &mut ddq, &mut dddq))?;
            }
        }

        Ok(PathDerivatives {
            q: DMatrix::from_vec(dim, n, q),
            dq: dq.map(|v| DMatrix::from_vec(dim, n, v)),
            ddq: ddq.map(|v| DMatrix::from_vec(dim, n, v)),
            dddq: dddq.map(|v| DMatrix::from_vec(dim, n, v)),
        })
    }

    #[inline(always)]
    fn validate_s(&self, s: f64, index: usize) -> Result<f64, PathError> {
        match self.out_of_range_mode {
            OutOfRangeMode::Error => {
                if s < self.s_min - EPS_RANGE || s > self.s_max + EPS_RANGE {
                    return Err(PathError::OutOfRangeS {
                        s_min: self.s_min,
                        s_max: self.s_max,
                        index,
                        value: s,
                    });
                }
                Ok(s.clamp(self.s_min, self.s_max))
            }
            OutOfRangeMode::Clamp => Ok(s.clamp(self.s_min, self.s_max)),
        }
    }
}

// ── free evaluation functions ─────────────────────────────────────────────────

/// The input of evaluation functions.
type EvalInput<'a> = (
    &'a [f64],                // s
    &'a mut [f64],            // q
    &'a mut Option<Vec<f64>>, // dq
    &'a mut Option<Vec<f64>>, // ddq
    &'a mut Option<Vec<f64>>, // dddq
);

/// Evaluate a parametric path into pre-allocated column-major buffers.
///
/// Per-column parallelism via Rayon: validate + AD-evaluate + write in one pass.
/// No intermediate `Vec<Vec<Jet3>>` allocation; results go directly into output buffers.
fn eval_parametric(
    eval_fn: &ParametricFn,
    path: &Path,
    dim: usize,
    input_eval: EvalInput,
) -> Result<(), PathError> {
    let (s_values, q, dq, ddq, dddq) = input_eval;
    let n = s_values.len();

    // Chunk each output buffer by `dim` so column j maps to slice [j*dim .. (j+1)*dim].
    // When a derivative level is not requested (`None`), we still need an
    // `IndexedParallelIterator` of the same length for `zip`; a Vec<None> is the
    // simplest way to satisfy Rayon's type constraints here.
    let dq_chunks: Vec<Option<&mut [f64]>> = dq.as_deref_mut().map_or_else(
        || (0..n).map(|_| None).collect(),
        |v| v.chunks_mut(dim).map(Some).collect(),
    );
    let ddq_chunks: Vec<Option<&mut [f64]>> = ddq.as_deref_mut().map_or_else(
        || (0..n).map(|_| None).collect(),
        |v| v.chunks_mut(dim).map(Some).collect(),
    );
    let dddq_chunks: Vec<Option<&mut [f64]>> = dddq.as_deref_mut().map_or_else(
        || (0..n).map(|_| None).collect(),
        |v| v.chunks_mut(dim).map(Some).collect(),
    );

    s_values
        .par_iter()
        .enumerate()
        .zip(q.par_chunks_mut(dim))
        .zip(dq_chunks.into_par_iter())
        .zip(ddq_chunks.into_par_iter())
        .zip(dddq_chunks.into_par_iter())
        .map(
            |(((((j, &s_raw), q_col), mut dq_col), mut ddq_col), mut dddq_col)| {
                let s_curr = path.validate_s(s_raw, j)?;
                let vals = eval_fn(Jet3::seed(s_curr));
                if vals.len() != dim {
                    return Err(PathError::DimensionMismatch);
                }
                for (i, jet) in vals.iter().enumerate() {
                    q_col[i] = jet.v;
                    if let Some(ref mut b) = dq_col {
                        b[i] = jet.d1;
                    }
                    if let Some(ref mut b) = ddq_col {
                        b[i] = jet.d2;
                    }
                    if let Some(ref mut b) = dddq_col {
                        b[i] = jet.d3;
                    }
                }
                Ok(())
            },
        )
        .collect()
}

/// Evaluate spline representation into pre-allocated column-major buffers.
///
/// Dispatches to `SplinePath::eval_at::<ORDER>` with the minimum derivative
/// order that satisfies the request — zero run-time branching per sample:
///   - `Order::Zero`  → `eval_at::<0>` (q only)
///   - `Order::Two`   → `eval_at::<2>` (q, dq, ddq)
///   - `Order::Three` → `eval_at::<3>` (q, dq, ddq, dddq)
fn eval_spline(
    spline: &SplinePath,
    path: &Path,
    dim: usize,
    input_eval: EvalInput,
) -> Result<(), PathError> {
    let (s_values, q, dq, ddq, dddq) = input_eval;
    match (dq.as_deref_mut(), ddq.as_deref_mut(), dddq.as_deref_mut()) {
        // ── Order::Zero: q only ───────────────────────────────────────────
        (None, None, None) => s_values
            .par_iter()
            .enumerate()
            .zip(q.par_chunks_mut(dim))
            .map(|((j, &s_raw), q_col)| -> Result<(), PathError> {
                let s_curr = path.validate_s(s_raw, j)?;
                // eval_at::<0> only writes q_col; the derivative slices are never
                // accessed, so zero-length arrays satisfy the borrow checker.
                let (mut no_dq, mut no_ddq, mut no_dddq): ([f64; 0], [f64; 0], [f64; 0]) =
                    ([], [], []);
                spline.eval_at::<0>(s_curr, dim, q_col, &mut no_dq, &mut no_ddq, &mut no_dddq);
                Ok(())
            })
            .collect(),
        // ── Order::Two: q, dq, ddq ────────────────────────────────────────
        (Some(dq_buf), Some(ddq_buf), None) => {
            let dq_chunks: Vec<&mut [f64]> = dq_buf.chunks_mut(dim).collect();
            let ddq_chunks: Vec<&mut [f64]> = ddq_buf.chunks_mut(dim).collect();
            s_values
                .par_iter()
                .enumerate()
                .zip(q.par_chunks_mut(dim))
                .zip(dq_chunks.into_par_iter())
                .zip(ddq_chunks.into_par_iter())
                .map(
                    |((((j, &s_raw), q_col), dq_col), ddq_col)| -> Result<(), PathError> {
                        let s_curr = path.validate_s(s_raw, j)?;
                        let mut s4 = [];
                        spline.eval_at::<2>(s_curr, dim, q_col, dq_col, ddq_col, &mut s4);
                        Ok(())
                    },
                )
                .collect()
        }
        // ── Order::Three: q, dq, ddq, dddq ───────────────────────────────
        (Some(dq_buf), Some(ddq_buf), Some(dddq_buf)) => {
            let dq_chunks: Vec<&mut [f64]> = dq_buf.chunks_mut(dim).collect();
            let ddq_chunks: Vec<&mut [f64]> = ddq_buf.chunks_mut(dim).collect();
            let dddq_chunks: Vec<&mut [f64]> = dddq_buf.chunks_mut(dim).collect();
            s_values
                .par_iter()
                .enumerate()
                .zip(q.par_chunks_mut(dim))
                .zip(dq_chunks.into_par_iter())
                .zip(ddq_chunks.into_par_iter())
                .zip(dddq_chunks.into_par_iter())
                .map(
                    |(((((j, &s_raw), q_col), dq_col), ddq_col), dddq_col)| -> Result<(), PathError> {
                        let s_curr = path.validate_s(s_raw, j)?;
                        spline.eval_at::<3>(s_curr, dim, q_col, dq_col, ddq_col, dddq_col);
                        Ok(())
                    },
                )
                .collect()
        }
        // Unreachable: evaluate_impl only produces the three patterns above.
        _ => unreachable!("unexpected dq/ddq/dddq combination"),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn validate_range(s_min: f64, s_max: f64) -> Result<(), PathError> {
    if !s_min.is_finite() || !s_max.is_finite() || s_max <= s_min {
        return Err(PathError::InvalidRange { s_min, s_max });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::PathDerivatives;
    use crate::path::{Jet3, Path as PathModel, PathError, SplineConfig, cos, exp, sin};
    use nalgebra::DMatrix;
    use plotters::prelude::*;
    use rand::RngExt;
    use std::error::Error;
    use std::fs::create_dir_all;
    use std::hint::black_box;
    use std::path::Path as StdPath;
    use std::time::Instant;

    const DIM: usize = 6;

    fn make_s(n: usize) -> DMatrix<f64> {
        DMatrix::<f64>::from_fn(1, n, |_, j| j as f64 / (n - 1) as f64)
    }

    fn make_parametric_path() -> Result<PathModel, PathError> {
        PathModel::from_parametric(
            |s: Jet3| {
                vec![
                    sin(s),
                    cos(s),
                    exp(0.3 * s) - 1.0,
                    s + 0.1 * s * s - 0.01 * s * s * s * s,
                    sin(2.0 * s) + 0.15 * cos(3.0 * s),
                    sin(s) * cos(s),
                ]
            },
            0.0,
            1.0,
        )
    }

    fn make_waypoints(n_pts: usize) -> DMatrix<f64> {
        let mut rng = rand::rng();
        // Random-walk waypoints: each row is one DOF, each column is a waypoint.
        let mut waypoints = DMatrix::<f64>::zeros(DIM, n_pts);
        for mut row in waypoints.row_iter_mut() {
            row[0] = rng.random_range(-1.0..1.0);
            for j in 1..n_pts {
                let step = rng.random_range(-0.35..0.35);
                row[j] = row[j - 1] + step;
            }
        }
        waypoints
    }

    #[test]
    fn test_parametric_autodiff_dim6() -> Result<(), PathError> {
        let path = make_parametric_path()?;

        let n = 300;
        let s = make_s(n);
        let out = path.evaluate_up_to_3rd(s.as_slice())?;
        let dq = out.dq.as_ref().unwrap();
        let ddq = out.ddq.as_ref().unwrap();
        let dddq = out.dddq.as_ref().unwrap();

        // Build expected values for all query points and check all 4 derivative orders.
        s.as_slice().iter().enumerate().for_each(|(j, &x)| {
            let e03x = (0.3 * x).exp();
            let expected_q = [
                x.sin(),
                x.cos(),
                e03x - 1.0,
                x + 0.1 * x * x - 0.01 * x * x * x * x,
                (2.0 * x).sin() + 0.15 * (3.0 * x).cos(),
                x.sin() * x.cos(),
            ];
            let expected_dq = [
                x.cos(),
                -x.sin(),
                0.3 * e03x,
                1.0 + 0.2 * x - 0.04 * x * x * x,
                2.0 * (2.0 * x).cos() - 0.45 * (3.0 * x).sin(),
                (2.0 * x).cos(),
            ];
            let expected_ddq = [
                -x.sin(),
                -x.cos(),
                0.09 * e03x,
                0.2 - 0.12 * x * x,
                -4.0 * (2.0 * x).sin() - 1.35 * (3.0 * x).cos(),
                -2.0 * (2.0 * x).sin(),
            ];
            let expected_dddq = [
                -x.cos(),
                x.sin(),
                0.027 * e03x,
                -0.24 * x,
                -8.0 * (2.0 * x).cos() + 4.05 * (3.0 * x).sin(),
                -4.0 * (2.0 * x).cos(),
            ];

            for i in 0..DIM {
                assert!(
                    (out.q[(i, j)] - expected_q[i]).abs() < 1e-10,
                    "q    dim={i} idx={j}"
                );
                assert!(
                    (dq[(i, j)] - expected_dq[i]).abs() < 1e-10,
                    "dq   dim={i} idx={j}"
                );
                assert!(
                    (ddq[(i, j)] - expected_ddq[i]).abs() < 1e-10,
                    "ddq  dim={i} idx={j}"
                );
                assert!(
                    (dddq[(i, j)] - expected_dddq[i]).abs() < 1e-10,
                    "dddq dim={i} idx={j}"
                );
            }
        });

        Ok(())
    }

    #[test]
    fn test_evaluate_q_only() -> Result<(), PathError> {
        let path = make_parametric_path()?;
        let n = 100;
        let s = make_s(n);
        let out = path.evaluate_q(s.as_slice())?;

        assert!(out.dq.is_none());
        assert!(out.ddq.is_none());
        assert!(out.dddq.is_none());

        for j in 0..n {
            let x = s[(0, j)];
            assert!((out.q[(0, j)] - x.sin()).abs() < 1e-10);
            assert!((out.q[(1, j)] - x.cos()).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_evaluate_up_to_2nd() -> Result<(), PathError> {
        let path = make_parametric_path()?;
        let n = 100;
        let s = make_s(n);
        let out = path.evaluate_up_to_2nd(s.as_slice())?;
        let dq = out.dq.as_ref().unwrap();
        let ddq = out.ddq.as_ref().unwrap();

        assert!(out.dddq.is_none());

        for j in 0..n {
            let x = s[(0, j)];
            assert!((out.q[(0, j)] - x.sin()).abs() < 1e-10);
            assert!((dq[(0, j)] - x.cos()).abs() < 1e-10);
            assert!((ddq[(0, j)] - (-x.sin())).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_quintic_spline_interpolates_waypoints_dim6() -> Result<(), PathError> {
        let n_pts = 25;
        let waypoints = make_waypoints(n_pts);

        let cfg = SplineConfig::default();
        let path = PathModel::from_waypoints(&waypoints, cfg)?;

        let s = DMatrix::<f64>::from_fn(1, n_pts, |_, j| j as f64 / (n_pts - 1) as f64);
        let out = path.evaluate_up_to_3rd(s.as_slice())?;
        let dq = out.dq.as_ref().unwrap();
        let ddq = out.ddq.as_ref().unwrap();
        let dddq = out.dddq.as_ref().unwrap();

        // The spline must interpolate every waypoint exactly (up to floating-point rounding)
        // and all derivatives must be finite (no blowup).
        for (i, j) in (0..DIM).flat_map(|i| (0..n_pts).map(move |j| (i, j))) {
            assert!((out.q[(i, j)] - waypoints[(i, j)]).abs() < 1e-8);
            assert!(dq[(i, j)].is_finite());
            assert!(ddq[(i, j)].is_finite());
            assert!(dddq[(i, j)].is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_s_out_of_range_error_dim6() -> Result<(), PathError> {
        let waypoints = make_waypoints(12);
        let path = PathModel::from_waypoints(&waypoints, SplineConfig::default())?;
        let s = DMatrix::<f64>::from_row_slice(1, 3, &[-0.1, 0.5, 1.1]);
        let err = path.evaluate_up_to_3rd(s.as_slice()).unwrap_err();
        match err {
            PathError::OutOfRangeS { .. } => {}
            _ => panic!("expected OutOfRangeS"),
        }
        Ok(())
    }

    #[test]
    fn test_benchmark_parametric_and_spline_dim6() -> Result<(), PathError> {
        let n_eval = 3000;
        let n_repeat = 8;
        let s = make_s(n_eval);

        let start = Instant::now();
        let param_path = make_parametric_path()?;
        let tc_build_param = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        for _ in 0..n_repeat {
            let out = param_path.evaluate_up_to_3rd(s.as_slice())?;
            black_box(out.q[(0, 0)]);
        }
        let tc_eval_param = start.elapsed().as_secs_f64() * 1e3 / n_repeat as f64;
        println!(
            "[bench][parametric][dim=6] build={tc_build_param:.3} ms eval={tc_eval_param:.3} ms (N={n_eval})"
        );

        let n_waypoints_list = [16usize, 32, 64, 128, 192, 256, 512, 1024];
        for &n_pts in &n_waypoints_list {
            let waypoints = make_waypoints(n_pts);
            let start = Instant::now();
            let spline_path = PathModel::from_waypoints(&waypoints, SplineConfig::default())?;
            let tc_build = start.elapsed().as_secs_f64() * 1e3;

            let start = Instant::now();
            for _ in 0..n_repeat {
                let out = spline_path.evaluate_up_to_3rd(s.as_slice())?;
                black_box(out.q[(0, 0)]);
            }
            let tc_eval = start.elapsed().as_secs_f64() * 1e3 / n_repeat as f64;

            println!(
                "[bench][spline][dim=6][n_pts={n_pts}] build={tc_build:.3} ms eval={tc_eval:.3} ms"
            );
        }

        Ok(())
    }

    #[test]
    #[ignore = "plotting"]
    fn test_plot_parametric_and_spline_derivatives() -> Result<(), Box<dyn Error>> {
        let dir = "data/path_plots";
        create_dir_all(dir)?;

        let n = 600;
        let s = make_s(n);
        let s_vec: Vec<f64> = (0..n).map(|j| s[(0, j)]).collect();

        let param_path = make_parametric_path()?;
        let param_out = param_path.evaluate_up_to_3rd(s.as_slice())?;
        plot_grid_4x6(
            &format!("{dir}/parametric_dim6_grid.png"),
            "parametric dim=6",
            &s_vec,
            &param_out,
            None,
        )?;

        let n_pts = 10;
        let waypoints = make_waypoints(n_pts);
        let spline_path = PathModel::from_waypoints(&waypoints, SplineConfig::default())?;
        let spline_out = spline_path.evaluate_up_to_3rd(s.as_slice())?;
        let wp_s: Vec<f64> = (0..n_pts).map(|j| j as f64 / (n_pts - 1) as f64).collect();
        plot_grid_4x6(
            &format!("{dir}/spline_order5_dim6_grid.png"),
            "spline order=5 dim=6",
            &s_vec,
            &spline_out,
            Some((&wp_s, &waypoints)),
        )?;

        Ok(())
    }

    fn plot_grid_4x6(
        file: &str,
        title: &str,
        s: &[f64],
        data: &PathDerivatives,
        waypoints: Option<(&[f64], &DMatrix<f64>)>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = StdPath::new(file).parent() {
            create_dir_all(parent)?;
        }

        let root = BitMapBackend::new(file, (2400, 1400)).into_drawing_area();
        root.fill(&WHITE)?;

        let empty = DMatrix::<f64>::zeros(0, 0);
        let dq = data.dq.as_ref().unwrap_or(&empty);
        let ddq = data.ddq.as_ref().unwrap_or(&empty);
        let dddq = data.dddq.as_ref().unwrap_or(&empty);

        let areas = root.split_evenly((4, DIM));
        let mats = [&data.q, dq, ddq, dddq];
        let row_names = ["q", "dq", "ddq", "dddq"];

        for row in 0..4 {
            for col in 0..DIM {
                let area = &areas[row * DIM + col];
                let series = mat_row(mats[row], col);
                let (mut y_min, mut y_max) = min_max_slice(&series);
                if (y_max - y_min).abs() < 1e-12 {
                    y_min -= 1.0;
                    y_max += 1.0;
                } else {
                    let pad = 0.08 * (y_max - y_min);
                    y_min -= pad;
                    y_max += pad;
                }

                let mut chart = ChartBuilder::on(area)
                    .margin(8)
                    .caption(
                        format!("{} j{}", row_names[row], col + 1),
                        ("sans-serif", 16),
                    )
                    .x_label_area_size(24)
                    .y_label_area_size(38)
                    .build_cartesian_2d(s[0]..s[s.len() - 1], y_min..y_max)?;

                chart
                    .configure_mesh()
                    .x_desc(if row == 3 { "s" } else { "" })
                    .y_desc("")
                    .draw()?;

                chart.draw_series(LineSeries::new(
                    (0..s.len()).map(|j| (s[j], series[j])),
                    &BLUE,
                ))?;

                if row == 0
                    && let Some((s_wp, q_wp)) = waypoints
                {
                    chart.draw_series(
                        s_wp.iter()
                            .zip(q_wp.row(col).iter())
                            .map(|(&xs, &ys)| Circle::new((xs, ys), 2, RED.filled())),
                    )?;
                }
            }
        }

        root.titled(title, ("sans-serif", 28))?;
        root.present()?;
        Ok(())
    }

    fn mat_row(mat: &DMatrix<f64>, row: usize) -> Vec<f64> {
        mat.row(row).iter().copied().collect()
    }

    fn min_max_slice(data: &[f64]) -> (f64, f64) {
        data.iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            })
    }
}
