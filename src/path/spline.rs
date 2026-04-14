//! Waypoint-based spline path construction and evaluation kernels.
//!
//! # Design
//!
//! All splines are **odd-order** (`p = 2m+1`, `m >= 1`):
//!   - order 3 (m=1): C2 cubic, boundary specifies v at both ends
//!   - order 5 (m=2): C4 quintic, boundary specifies v,a at both ends
//!   - order 7 (m=3): C6 septic, boundary specifies v,a,j at both ends
//!
//! The user supplies `start_state` and `end_state` matrices of shape `(dim, m)`,
//! where column `r-1` contains the r-th derivative value at the endpoint.
//! `None` means all-zero (the most common default).
//!
//! Internally, all orders share a single `solve_general_thomas` solver that
//! implements the O(N) block-Thomas algorithm on the `m×m` block-tridiagonal
//! system arising from Hermite parametrisation + C^{m+1}..C^{2m} continuity.

use crate::diag::PathError;
use crate::path::OutOfRangeMode;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

// ── Public configuration ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum Parametrization {
    Uniform,
}

pub struct SplineConfig {
    /// Spline order: must be an odd number >= 3.
    pub order: usize,
    pub parametrization: Parametrization,
    pub s_min: f64,
    pub s_max: f64,
    pub out_of_range_mode: OutOfRangeMode,
    /// Boundary derivatives at s_min: shape `(dim, m)` where `m = (order-1)/2`.
    /// Column r (0-indexed) = (r+1)-th derivative value.
    /// `None` = all-zero (default).
    pub start_state: Option<DMatrix<f64>>,
    /// Boundary derivatives at s_max: same shape as `start_state`.
    /// `None` = all-zero (default).
    pub end_state: Option<DMatrix<f64>>,
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            order: 5,
            parametrization: Parametrization::Uniform,
            s_min: 0.0,
            s_max: 1.0,
            out_of_range_mode: OutOfRangeMode::Error,
            start_state: None,
            end_state: None,
        }
    }
}

// ── SplinePath ────────────────────────────────────────────────────────────────

/// Piecewise-polynomial path in normalised segment coordinates.
///
/// Coefficients are stored in column-major blocks by dimension:
/// `[dim0_seg0.., dim0_seg1.., ..., dim1_seg0.., ...]`.
pub struct SplinePath {
    // ── Evaluation metadata (all cheap Copy types) ───────────────────────
    pub order: usize,
    pub s_min: f64,
    pub s_max: f64,
    pub out_of_range_mode: OutOfRangeMode,
    n_coef: usize,
    n_segments: usize,
    /// Scale factor: maps a change in `s` to normalised segment units.
    /// `inv_ds = n_segments / (s_max - s_min)`.
    /// Each derivative order gains one additional factor of `inv_ds`.
    inv_ds: f64,
    // ── Coefficient storage ───────────────────────────────────────────────
    /// Flat Horner coefficients, layout: `[dim][segment][coef]`.
    coeffs: Vec<f64>,
}

impl SplinePath {
    /// Build spline coefficients from a waypoint matrix of shape `(dim, n_points)`.
    pub fn from_waypoints(waypoints: &DMatrix<f64>, cfg: &SplineConfig) -> Result<Self, PathError> {
        let order = cfg.order;
        if order < 3 || order.is_multiple_of(2) {
            return Err(PathError::InvalidOrder { order });
        }
        let dim = waypoints.nrows();
        let n_points = waypoints.ncols();
        if n_points < 2 {
            return Err(PathError::NotEnoughWaypoints { n: n_points });
        }
        let n_segments = n_points - 1;
        let n_coef = order + 1;
        let m = (order - 1) / 2;

        // Validate / extract boundary state (dim × m)
        let start = extract_boundary(cfg.start_state.as_ref(), dim, m)?;
        let end = extract_boundary(cfg.end_state.as_ref(), dim, m)?;
        let coeffs = solve_general_thomas(waypoints, order, m, &start, &end)?;

        let range = cfg.s_max - cfg.s_min;
        let inv_ds = n_segments as f64 / range;

        Ok(Self {
            order,
            s_min: cfg.s_min,
            s_max: cfg.s_max,
            out_of_range_mode: cfg.out_of_range_mode,
            n_coef,
            n_segments,
            inv_ds,
            coeffs,
        })
    }

    #[inline(always)]
    fn segment_tau(&self, s: f64) -> (usize, f64) {
        let scaled = (s - self.s_min) * self.inv_ds;
        if scaled <= 0.0 {
            return (0, 0.0);
        }
        let max_scaled = self.n_segments as f64;
        if scaled >= max_scaled {
            return (self.n_segments - 1, 1.0);
        }
        let seg = scaled.floor() as usize;
        (seg, scaled - seg as f64)
    }

    /// Evaluate Horner polynomial and up to 3 derivatives at `t ∈ [0,1]`.
    ///
    /// Returns `(q, dq, ddq, dddq)`.  Specialised fast paths for degree 5 and 3
    /// (the two most common cases); falls back to the general Horner+derivative
    /// accumulation for other degrees.
    #[inline(always)]
    fn eval_poly<const ORDER: u8>(coeffs: &[f64], t: f64) -> (f64, f64, f64, f64) {
        if coeffs.len() == 6 {
            let [a0, a1, a2, a3, a4, a5] = [
                coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5],
            ];
            let q = a5
                .mul_add(t, a4)
                .mul_add(t, a3)
                .mul_add(t, a2)
                .mul_add(t, a1)
                .mul_add(t, a0);
            if ORDER == 0 {
                return (q, 0.0, 0.0, 0.0);
            }
            let dq = (5.0 * a5)
                .mul_add(t, 4.0 * a4)
                .mul_add(t, 3.0 * a3)
                .mul_add(t, 2.0 * a2)
                .mul_add(t, a1);
            let ddq = (20.0 * a5)
                .mul_add(t, 12.0 * a4)
                .mul_add(t, 6.0 * a3)
                .mul_add(t, 2.0 * a2);
            if ORDER == 2 {
                return (q, dq, ddq, 0.0);
            }
            let dddq = (60.0 * a5).mul_add(t, 24.0 * a4).mul_add(t, 6.0 * a3);
            return (q, dq, ddq, dddq);
        }
        if coeffs.len() == 4 {
            let [a0, a1, a2, a3] = [coeffs[0], coeffs[1], coeffs[2], coeffs[3]];
            let q = a3.mul_add(t, a2).mul_add(t, a1).mul_add(t, a0);
            if ORDER == 0 {
                return (q, 0.0, 0.0, 0.0);
            }
            let dq = (3.0 * a3).mul_add(t, 2.0 * a2).mul_add(t, a1);
            let ddq = (6.0 * a3).mul_add(t, 2.0 * a2);
            if ORDER == 2 {
                return (q, dq, ddq, 0.0);
            }
            return (q, dq, ddq, 6.0 * a3);
        }
        // General Horner + simultaneous derivative accumulation (arbitrary degree).
        let n = coeffs.len() - 1;
        let mut b0 = coeffs[n];
        let mut b1 = 0.0_f64;
        let mut b2 = 0.0_f64;
        let mut b3 = 0.0_f64;
        for k in (0..n).rev() {
            if ORDER >= 3 {
                b3 = b3 * t + 3.0 * b2;
            }
            if ORDER >= 2 {
                b2 = b2 * t + 2.0 * b1;
            }
            if ORDER >= 1 {
                b1 = b1 * t + b0;
            }
            b0 = b0 * t + coeffs[k];
        }
        (b0, b1, b2, b3)
    }

    /// Evaluate position and up to `ORDER` derivatives at path parameter `s`
    /// for all `dim` joints, writing results into the supplied column slices.
    ///
    /// `ORDER` controls which slices must be valid:
    ///   - `0` : only `q_col` is written; `dq_col / ddq_col / dddq_col` are ignored
    ///   - `2` : `q_col`, `dq_col`, `ddq_col` are written; `dddq_col` is ignored
    ///   - `3` : all four slices are written
    ///
    /// Callers in `path_core.rs` specialise this with `eval_at::<0>`, `eval_at::<2>`,
    /// and `eval_at::<3>`, replacing the former `eval_at_q_only`, `eval_at_up_to_2nd`,
    /// and `eval_at_full` methods.
    #[inline(always)]
    pub fn eval_at<const ORDER: u8>(
        &self,
        s: f64,
        _dim: usize,
        q_col: &mut [f64],
        dq_col: &mut [f64],
        ddq_col: &mut [f64],
        dddq_col: &mut [f64],
    ) {
        let (seg, tau) = self.segment_tau(s);
        let seg_offset = seg * self.n_coef;
        let inv_ds2 = self.inv_ds * self.inv_ds;
        let inv_ds3 = inv_ds2 * self.inv_ds;
        let stride = self.n_segments * self.n_coef;
        for (i, q_v) in q_col.iter_mut().enumerate() {
            let start = i * stride + seg_offset;
            let (q_val, dq_val, ddq_val, dddq_val) =
                Self::eval_poly::<ORDER>(&self.coeffs[start..start + self.n_coef], tau);
            *q_v = q_val;
            if ORDER >= 2 {
                dq_col[i] = dq_val * self.inv_ds;
                ddq_col[i] = ddq_val * inv_ds2;
            }
            if ORDER >= 3 {
                dddq_col[i] = dddq_val * inv_ds3;
            }
        }
    }
}

// ── Boundary helpers ──────────────────────────────────────────────────────────

/// Extract boundary state as a flat `dim × m` array (row-major: `[d][r]`).
/// Returns `vec![0; dim*m]` when `state` is `None`.
fn extract_boundary(
    state: Option<&DMatrix<f64>>,
    dim: usize,
    m: usize,
) -> Result<Vec<f64>, PathError> {
    match state {
        None => Ok(vec![0.0; dim * m]),
        Some(mat) => {
            if mat.nrows() != dim || mat.ncols() != m {
                return Err(PathError::DimensionMismatch);
            }
            // Row-major: out[d * m + r] = mat[(d, r)]
            let mut out = vec![0.0; dim * m];
            for (d, row) in out.chunks_mut(m).enumerate() {
                for (r, val) in row.iter_mut().enumerate() {
                    *val = mat[(d, r)];
                }
            }
            Ok(out)
        }
    }
}

// ── Unified O(N) block-Thomas solver ─────────────────────────────────────────
//
// For any odd order p=2m+1 with uniform parametrisation (h=1 per segment):
//
// Each segment i has 2m+2 Horner coefficients.  The lower m+1 are:
//   a[0] = y_i,  a[r] = u_r(i)  for r=1..m
// where u_r(i) are the m derivative unknowns at node i.
//
// The upper m+1 (a[m+1]..a[2m+1]) are determined by the endpoint conditions
// and expressed as a linear combination of (dy_i, u(i), u(i+1)).
//
// Enforcing C^{m+1}..C^{2m} continuity at each interior node gives:
//
//   A·u_{i-1}  +  B·u_i  +  C·u_{i+1}  =  R·[dy_{i-1}, dy_i]^T
//
// where A, B, C, R are constant m×m (resp. m×2) matrices depending only on m.
// These are stored as compile-time constants for m=1,2,3 and computed
// at run-time for larger m via the general formula.
//
// The block Thomas algorithm solves this in O(N) time per dimension.
// The solve for each dimension is independent and runs in parallel via Rayon.
//
// Boundary conditions at the two endpoint nodes:
//   u(0)   = start_state[d, :]   (the m boundary derivative values)
//   u(N-1) = end_state[d, :]
//
// These are absorbed as known vectors; the interior system has size (N-2) × m.

// ── Block matrices ────────────────────────────────────────────────────────────

/// Precomputed block-tridiagonal matrices for a given `m`.
///
/// All m×m matrices use `DMatrix<f64>` for nalgebra operations.
/// `h_coeff` has shape `(m+1) × (1+2m)`.
struct BlockMatrices {
    m: usize,
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    /// R: m×2
    r: DMatrix<f64>,
    /// shape (m+1) × (1+2m); row i ↔ a[m+1+i]
    h_coeff: DMatrix<f64>,
}

impl BlockMatrices {
    fn for_order(m: usize) -> Self {
        // Use precomputed exact-rational values for m=1,2,3; fall back to
        // the general derivation (floating point) for larger m.
        match m {
            1 => Self::m1(),
            2 => Self::m2(),
            3 => Self::m3(),
            _ => Self::general(m),
        }
    }

    // p=3 (m=1): A=[1], B=[4], C=[1], R=[[3,3]]
    // Hermite: a[2]=3*dy-2*u0-u1, a[3]=-2*dy+u0+u1
    fn m1() -> Self {
        BlockMatrices {
            m: 1,
            a: DMatrix::from_row_slice(1, 1, &[1.0]),
            b: DMatrix::from_row_slice(1, 1, &[4.0]),
            c: DMatrix::from_row_slice(1, 1, &[1.0]),
            r: DMatrix::from_row_slice(1, 2, &[3.0, 3.0]),
            h_coeff: DMatrix::from_row_slice(2, 3, &[3.0, -2.0, -1.0, -2.0, 1.0, 1.0]),
        }
    }

    // p=5 (m=2): known exact matrices (verified by Python derivation)
    // Hermite: a[3]=10dy-6u0_1-3u0_2-4u1_1+u1_2
    //          a[4]=-15dy+8u0_1+3u0_2+7u1_1-2u1_2
    //          a[5]=6dy-3u0_1-u0_2-3u1_1+u1_2
    // A=[[-4,-1],[-7,-2]], B=[[0,6],[-16,0]], C=[[4,-1],[-7,2]]
    // R=[[-10,10],[-15,-15]]
    fn m2() -> Self {
        BlockMatrices {
            m: 2,
            a: DMatrix::from_row_slice(2, 2, &[-4.0, -1.0, -7.0, -2.0]),
            b: DMatrix::from_row_slice(2, 2, &[0.0, 6.0, -16.0, 0.0]),
            c: DMatrix::from_row_slice(2, 2, &[4.0, -1.0, -7.0, 2.0]),
            r: DMatrix::from_row_slice(2, 2, &[-10.0, 10.0, -15.0, -15.0]),
            h_coeff: DMatrix::from_row_slice(
                3,
                5,
                &[
                    10.0, -6.0, -3.0, -4.0, 1.0, -15.0, 8.0, 3.0, 7.0, -2.0, 6.0, -3.0, -1.0, -3.0,
                    1.0,
                ],
            ),
        }
    }

    // p=7 (m=3): derived by Python symbolic computation
    // A=[[15,5,1],[39,14,3],[34,13,3]]
    // B=[[40,0,8],[0,-40,0],[72,0,8]]
    // C=[[15,-5,1],[-39,14,-3],[34,-13,3]]
    // R=[[35,35],[84,-84],[70,70]]
    fn m3() -> Self {
        BlockMatrices {
            m: 3,
            a: DMatrix::from_row_slice(3, 3, &[15.0, 5.0, 1.0, 39.0, 14.0, 3.0, 34.0, 13.0, 3.0]),
            b: DMatrix::from_row_slice(3, 3, &[40.0, 0.0, 8.0, 0.0, -40.0, 0.0, 72.0, 0.0, 8.0]),
            c: DMatrix::from_row_slice(
                3,
                3,
                &[15.0, -5.0, 1.0, -39.0, 14.0, -3.0, 34.0, -13.0, 3.0],
            ),
            r: DMatrix::from_row_slice(3, 2, &[35.0, 35.0, 84.0, -84.0, 70.0, 70.0]),
            h_coeff: DMatrix::from_row_slice(
                4,
                7,
                &[
                    35.0, -20.0, -10.0, -4.0, -15.0, 5.0, -1.0, -84.0, 45.0, 20.0, 6.0, 39.0,
                    -14.0, 3.0, 70.0, -36.0, -15.0, -4.0, -34.0, 13.0, -3.0, -20.0, 10.0, 4.0, 1.0,
                    10.0, -4.0, 1.0,
                ],
            ),
        }
    }

    /// General derivation for m >= 4 via Gauss-Jordan inversion (f64 arithmetic).
    fn general(m: usize) -> Self {
        let p = 2 * m + 1;

        // Build M (m+1 × m+1): endpoint conditions at t=1
        // M[r][k-(m+1)] = C(k, r)  for r=0..m, k=m+1..2m+1
        let n = m + 1;

        let binom = |nn: usize, k: usize| -> f64 {
            if k > nn {
                return 0.0;
            }
            (0..k).fold(1.0f64, |acc, i| acc * (nn - i) as f64 / (i + 1) as f64)
        };

        // Augmented matrix [M | I] for Gauss-Jordan inversion
        let mut aug = DMatrix::<f64>::zeros(n, 2 * n);
        for r in 0..n {
            for (col_idx, k) in (m + 1..=p).enumerate() {
                aug[(r, col_idx)] = binom(k, r);
            }
            aug[(r, n + r)] = 1.0;
        }
        for col in 0..n {
            let pivot_row = (col..n)
                .max_by(|&a, &b| {
                    aug[(a, col)]
                        .abs()
                        .partial_cmp(&aug[(b, col)].abs())
                        .unwrap()
                })
                .unwrap();
            if pivot_row != col {
                aug.swap_rows(col, pivot_row);
            }
            let piv_inv = 1.0 / aug[(col, col)];
            for j in 0..2 * n {
                aug[(col, j)] *= piv_inv;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug[(r, col)];
                if factor == 0.0 {
                    continue;
                }
                for j in 0..2 * n {
                    let sub = factor * aug[(col, j)];
                    aug[(r, j)] -= sub;
                }
            }
        }
        let minv = aug.columns(n, n).into_owned();

        // RHS basis vectors: shape (n) × (1+2m)
        let basis = 1 + 2 * m;
        let mut rhs_mat = DMatrix::<f64>::zeros(n, basis);
        rhs_mat[(0, 0)] = 1.0;
        for b in 1..=m {
            rhs_mat[(0, b)] = -1.0;
        }
        for r in 1..n {
            rhs_mat[(r, m + r)] = 1.0;
            for k in r..n {
                rhs_mat[(r, k)] -= binom(k, r);
            }
        }

        // h_coeff = M^{-1} * rhs_mat
        let h_coeff = &minv * &rhs_mat;

        // Build A, B, C, R from continuity equations
        let bs = 2 + 3 * m;
        let get_coeff_vec = |k: usize, is_left: bool| -> DVector<f64> {
            let mut v = DVector::<f64>::zeros(bs);
            if k == 0 {
                return v;
            }
            if k <= m {
                let offset = if is_left { 2 } else { 2 + m };
                v[offset + k - 1] = 1.0;
                return v;
            }
            let ic = k - (m + 1);
            for (b, &cv) in h_coeff.row(ic).iter().enumerate() {
                if cv == 0.0 {
                    continue;
                }
                if b == 0 {
                    v[if is_left { 0 } else { 1 }] += cv;
                } else if b <= m {
                    let offset = if is_left { 2 } else { 2 + m };
                    v[offset + b - 1] += cv;
                } else {
                    let offset = if is_left { 2 + m } else { 2 + 2 * m };
                    v[offset + b - m - 1] += cv;
                }
            }
            v
        };

        let mut a_mat = DMatrix::<f64>::zeros(m, m);
        let mut b_mat = DMatrix::<f64>::zeros(m, m);
        let mut c_mat = DMatrix::<f64>::zeros(m, m);
        let mut r_mat = DMatrix::<f64>::zeros(m, 2);

        for (eq_idx, r) in (m + 1..=2 * m).enumerate() {
            let lhs = (r..=p).fold(DVector::<f64>::zeros(bs), |acc, k| {
                acc + get_coeff_vec(k, true) * binom(k, r)
            });
            let rhs_v = get_coeff_vec(r, false);
            let eq = lhs - rhs_v;
            for j in 0..m {
                a_mat[(eq_idx, j)] = eq[2 + j];
                b_mat[(eq_idx, j)] = eq[2 + m + j];
                c_mat[(eq_idx, j)] = eq[2 + 2 * m + j];
            }
            r_mat[(eq_idx, 0)] = -eq[0];
            r_mat[(eq_idx, 1)] = -eq[1];
        }

        BlockMatrices {
            m,
            a: a_mat,
            b: b_mat,
            c: c_mat,
            r: r_mat,
            h_coeff,
        }
    }

    /// Compute RHS: `out = R * [dy_prev, dy_next]^T`
    #[inline(always)]
    fn compute_rhs(&self, dy_prev: f64, dy_next: f64, out: &mut [f64]) {
        let r0 = self.r.column(0);
        let r1 = self.r.column(1);
        for (o, (&c0, &c1)) in out.iter_mut().zip(r0.iter().zip(r1.iter())) {
            *o = c0 * dy_prev + c1 * dy_next;
        }
    }

    /// Reconstruct upper Horner coefficients `a[m+1]..a[2m+1]` into `out`.
    ///
    /// `h_coeff` row layout: `[dy_coeff, u0_1..m, u1_1..m]` (1 + 2m columns).
    #[inline(always)]
    fn upper_coeffs(&self, dy: f64, u0: &[f64], u1: &[f64], out: &mut [f64]) {
        let m = self.m;
        for (o, row) in out.iter_mut().zip(self.h_coeff.row_iter()) {
            // dot(row[1..=m], u0) + dot(row[m+1..=2m], u1)
            let dot_u0: f64 = row
                .iter()
                .skip(1)
                .take(m)
                .zip(u0.iter())
                .map(|(&c, &v)| c * v)
                .sum();
            let dot_u1: f64 = row
                .iter()
                .skip(1 + m)
                .zip(u1.iter())
                .map(|(&c, &v)| c * v)
                .sum();
            *o = row[0] * dy + dot_u0 + dot_u1;
        }
    }
}

// ── Main solver ───────────────────────────────────────────────────────────────

/// Unified O(N) block-Thomas spline solver for any odd order `p = 2m+1`.
///
/// `start_bd` and `end_bd` are flat `dim × m` arrays (row-major) containing
/// the m boundary derivative values at the first and last waypoint respectively.
fn solve_general_thomas(
    waypoints: &DMatrix<f64>,
    order: usize,
    m: usize,
    start_bd: &[f64],
    end_bd: &[f64],
) -> Result<Vec<f64>, PathError> {
    let dim = waypoints.nrows();
    let n = waypoints.ncols(); // number of nodes
    let ns = n - 1; // number of segments
    let n_coef = order + 1; // = 2m+2

    let bm = BlockMatrices::for_order(m);

    let chunk_len = ns * n_coef;
    let mut coeffs = vec![0.0f64; dim * chunk_len];

    coeffs.par_chunks_mut(chunk_len).enumerate().try_for_each(
        |(d, chunk)| -> Result<(), PathError> {
            // Boundary derivative slices for this dimension — no allocation needed.
            let u_start = &start_bd[d * m..(d + 1) * m];
            let u_end = &end_bd[d * m..(d + 1) * m];
            thomas_row(
                waypoints,
                (d, n, ns, m, n_coef),
                &bm,
                (u_start, u_end),
                chunk,
            )
        },
    )?;

    Ok(coeffs)
}

/// Solve one dimension with the block-Thomas algorithm.
///
/// # Block-Thomas algorithm
///
/// The block-tridiagonal system is:
///   A·u_{i-1} + B·u_i + C·u_{i+1} = rhs_i,  i=1..N-2
/// with u_0 = u_start, u_{N-1} = u_end (known boundary values).
///
/// Forward sweep: B'_0 = B - A*(0) = B (u_{-1}=u_start contributes only to rhs).
///   B'_i = B - A * (B'_{i-1}^{-1} * C)
///   r'_i = rhs_i - A * (B'_{i-1}^{-1} * r'_{i-1})
///
/// Back substitution: u_{N-2} = B'_{N-2}^{-1} * r'_{N-2}
///   u_i = B'_i^{-1} * (r'_i - C * u_{i+1})
fn thomas_row(
    waypoints: &DMatrix<f64>,
    (d, n, ns, m, n_coef): (usize, usize, usize, usize, usize),
    bm: &BlockMatrices,
    (u_start, u_end): (&[f64], &[f64]),
    chunk: &mut [f64],
) -> Result<(), PathError> {
    // Segment differences: dy[i] = waypoints[d, i+1] - waypoints[d, i]
    let row_d = waypoints.row(d);
    let dy: Vec<f64> = row_d
        .iter()
        .zip(row_d.iter().skip(1))
        .map(|(&a, &b)| b - a)
        .collect();

    // Scratch buffers
    let mut rhs = vec![0.0f64; m];
    let mut upper_buf = vec![0.0f64; m + 1];

    // Special case: single segment
    if n == 2 {
        bm.upper_coeffs(dy[0], u_start, u_end, &mut upper_buf);
        write_segment_coeffs(chunk, 0, n_coef, m, waypoints[(d, 0)], u_start, &upper_buf);
        return Ok(());
    }

    let ni = n - 2; // number of interior nodes (indices 1..=n-2)

    // ── Forward sweep ─────────────────────────────────────────────────────────
    // b_inv_c_list[i] : B'_i^{-1} * C   (m×m DMatrix)
    // b_inv_r_list[i] : B'_i^{-1} * r'_i (m-DVector)
    // Forward sweep: accumulate B'^{-1}*C and B'^{-1}*rhs using DMatrix
    let mut b_inv_c_list: Vec<DMatrix<f64>> = Vec::with_capacity(ni);
    let mut b_inv_r_list: Vec<DVector<f64>> = Vec::with_capacity(ni);

    // Precompute boundary correction vectors (allocated once, reused across iterations)
    let u_start_vec = DVector::from_column_slice(u_start);
    let u_end_vec = DVector::from_column_slice(u_end);

    for i in 0..ni {
        let node = i + 1;
        bm.compute_rhs(dy[node - 1], dy[node], &mut rhs);
        let mut rhs_curr = DVector::from_column_slice(&rhs);

        // Absorb boundary nodes into the RHS at the two ends of the interior system
        if i == 0 {
            rhs_curr -= &bm.a * &u_start_vec;
        }
        if i == ni - 1 {
            rhs_curr -= &bm.c * &u_end_vec;
        }

        // B'_i = B  (i=0) or  B - A * (B'_{i-1}^{-1} * C)
        let b_cur = if i == 0 {
            bm.b.clone()
        } else {
            &bm.b - &bm.a * &b_inv_c_list[i - 1]
        };

        // r'_i -= A * (B'_{i-1}^{-1} * r'_{i-1})  for i > 0
        if i > 0 {
            rhs_curr -= &bm.a * &b_inv_r_list[i - 1];
        }

        let lu = b_cur.lu();
        b_inv_c_list.push(lu.solve(&bm.c).ok_or(PathError::SingularSystem)?);
        b_inv_r_list.push(lu.solve(&rhs_curr).ok_or(PathError::SingularSystem)?);
    }

    // ── Back substitution ──────────────────────────────────────────────────────
    // u_list[i] stores the recovered derivative DVector at interior node i+1.
    let mut u_list: Vec<DVector<f64>> = vec![DVector::zeros(m); ni];
    u_list[ni - 1] = b_inv_r_list[ni - 1].clone();
    for i in (0..ni - 1).rev() {
        u_list[i] = &b_inv_r_list[i] - &b_inv_c_list[i] * &u_list[i + 1];
    }

    // ── Reconstruct Horner coefficients ───────────────────────────────────────
    let u_at = |k: usize| -> &[f64] {
        if k == 0 {
            u_start
        } else if k >= n - 1 {
            u_end
        } else {
            u_list[k - 1].as_slice()
        }
    };

    for seg in 0..ns {
        let ul = u_at(seg);
        let ur = u_at(seg + 1);
        bm.upper_coeffs(dy[seg], ul, ur, &mut upper_buf);
        write_segment_coeffs(chunk, seg, n_coef, m, waypoints[(d, seg)], ul, &upper_buf);
    }

    Ok(())
}

/// Write the full Horner coefficient block for one segment into `chunk`.
/// Layout: `chunk[seg * n_coef .. (seg+1) * n_coef]`
///   - a[0]       = y_left
///   - a[1..m]    = u_left[0..m-1]  (derivatives 1..m)
///   - a[m+1..2m+1] = upper[0..m]
#[inline(always)]
fn write_segment_coeffs(
    chunk: &mut [f64],
    seg: usize,
    n_coef: usize,
    m: usize,
    y_left: f64,
    u_left: &[f64],
    upper: &[f64],
) {
    let base = seg * n_coef;
    chunk[base] = y_left;
    chunk[base + 1..base + 1 + m].copy_from_slice(u_left);
    chunk[base + m + 1..base + m + 1 + (m + 1)].copy_from_slice(upper);
}
