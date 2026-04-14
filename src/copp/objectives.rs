//! Objective definitions and shared validators for COPP2/COPP3.  
//! Basic symbols (consistent with constraints):  
//! - $a(s)=\dot s^2$, sampled as `a[k]=a(s_k)`.  
//! - $b(s)=\ddot s$.  
//! - Path grid is `s[0], s[1], ..., s[n-1]` with `n = s.len()`.
//!
//! Discretization difference:  
//! - COPP2:  
//!   - `a.len() == s.len() == n`  
//!   - `b.len() == n-1`, where `b[k]` is on interval `[s_k, s_{k+1}]`  
//!   - typically `b[k] = (a[k+1]-a[k]) / (2*(s[k+1]-s[k]))`  
//! - COPP3:  
//!   - `a.len() == b.len() == s.len() == n`  
//!   - `b[k] = b(s_k)` is node-based.

use crate::diag::CoppError;

/// Objective terms for COPP optimization.  
/// Continuous formulation is shared by COPP2/COPP3; discrete form depends on how `b` and torque are sampled.  
/// Torque notation:  
/// - continuous: $\boldsymbol{\tau}(s)$.  
/// - discrete: `tau[i][k]` for joint `i` at `s[k]`.  
/// - COPP2: `tau[i][k]` is the right-limit value $\boldsymbol{\tau}(s_k^+)$, i.e. computed on interval $[s_k, s_{k+1}]$ from `(a[k], a[k+1], b[k])`.  
/// - COPP3: `tau[i][k]` is node value $\boldsymbol{\tau}(s_k)$, computed from `(a[k], b[k])`.
pub enum CoppObjective<'a> {
    /// Time objective.  
    /// + Continuous: $J_{\mathrm{time}} = w_t\int_{0}^{t_f} 1 \mathrm{d}t = w_t\int_{s_0}^{s_f} \frac{1}{\sqrt{a(s)}} \mathrm{d}s$.  
    /// + Discrete (COPP2): `J_time = 2*w_t*sum_{k=0}^{n-2} (s[k+1]-s[k])/(sqrt(a[k])+sqrt(a[k+1]))`.  
    /// + Discrete (COPP3): `J_time = w_t*sum_k weight_a_time[k]/sqrt(a[k])`
    Time(f64),
    /// Thermal-energy objective.  
    /// + Continuous: $J_{\mathrm{th}} = w_e\int_{0}^{t_f} \sum_i(\tau_i(s)\nu_i)^2\mathrm{d}t = w_e\int_{s_0}^{s_f} \sum_i(\tau_i(s)\nu_i)^2\frac{1}{\sqrt{a(s)}}\mathrm{d}s$ where `nu_i = normalize[i]`.  
    /// + Discrete (COPP2): `J_th = 2*w_e*sum_{k=0}^{n-2} (s[k+1]-s[k])/(sqrt(a[k])+sqrt(a[k+1])) * sum_i (tau[i][k]*normalize[i])^2`.  
    /// + Discrete (COPP3): `J_th = w_e*sum_k weight_a_torque[k] * sum_i (tau[i][k]*normalize[i])^2`.
    ThermalEnergy(f64, &'a [f64]),
    /// Total-variation of torque objective.  
    /// + Continuous: $J_{\mathrm{tv}} = w_v\sum_i \int_{0}^{t_f} \left|\frac{d\tau_i}{\mathrm{d}s}(s)\right|\nu_i\mathrm{d}t = w_v\sum_i \int_{s_0}^{s_f} \left|\frac{d\tau_i}{\mathrm{d}s}(s)\right|\nu_i\mathrm{d}s$.
    /// + Discrete:  `J_tv = w_v*sum_i sum_k |tau[i][k+1]-tau[i][k]|*normalize[i]`
    TotalVariationTorque(f64, &'a [f64]),
    /// Linear objective over `a` and `b`.  
    /// + Continuous: $J_{\mathrm{lin}} = w_l\int_{s_0}^{s_f}(\alpha(s)a(s)+\beta(s)b(s))\mathrm{d}s$.
    /// + Discrete (COPP2):
    ///   - `alpha.len()==n`, `beta.len()==n-1`
    ///   - `J_lin = w_l*( sum_{k=0}^{n-1} alpha[k]*a[k] + sum_{k=0}^{n-2} beta[k]*b[k] )`
    /// + Discrete (COPP3):
    ///   - `alpha.len()==beta.len()==n`
    ///   - `J_lin = w_l*sum_{k=0}^{n-1} (alpha[k]*a[k] + beta[k]*b[k])`
    Linear(f64, &'a [f64], &'a [f64]),
}

/// Validate objective terms for COPP2.  
/// `s_len` must be the number of grid points in the closed interval `[idx_s_start, idx_s_final]`, i.e. `idx_s_final - idx_s_start + 1`.  
/// Rules:  
/// - `objectives` must be non-empty.  
/// - all weights must be non-negative and finite.  
/// - for `ThermalEnergy` / `TotalVariationTorque`, `normalize.len() == dim`.  
/// - all normalize entries must be non-negative and finite.  
/// - for `Linear`, `alpha.len() == s_len` and `beta.len() + 1 == s_len`.
#[inline]
pub(crate) fn validate_copp2_objectives(
    function_name: &str,
    objectives: &[CoppObjective<'_>],
    dim: usize,
    s_len: usize,
) -> Result<(), CoppError> {
    if objectives.is_empty() {
        return Err(CoppError::InvalidInput(
            function_name.into(),
            "objectives must not be empty".into(),
        ));
    }

    for (i, objective) in objectives.iter().enumerate() {
        match objective {
            CoppObjective::Time(weight) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Time(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
            }
            CoppObjective::ThermalEnergy(weight, normalize)
            | CoppObjective::TotalVariationTorque(weight, normalize) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!("objectives[{i}] weight={weight} must be finite and non-negative"),
                    ));
                }
                if normalize.len() != dim {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}] normalize.len()={} must equal robot dim={dim}",
                            normalize.len()
                        ),
                    ));
                }
                for (j, v) in normalize.iter().enumerate() {
                    if v.is_nan() || v.is_infinite() || *v < 0.0 {
                        return Err(CoppError::InvalidInput(
                            function_name.into(),
                            format!(
                                "objectives[{i}] normalize[{j}]={v} must be finite and non-negative"
                            ),
                        ));
                    }
                }
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
                if alpha.len() != s_len || beta.len() + 1 != s_len {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear length mismatch: alpha.len()={}, beta.len()={}, expected alpha.len()={s_len}, beta.len()={}",
                            alpha.len(),
                            beta.len(),
                            s_len - 1
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Validate objective terms for COPP3.  
/// `s_len` must be the number of grid points used by TOPP3/COPP3 (typically `a_linear.len()`).  
/// Rules:  
/// - `objectives` must be non-empty.  
/// - all weights must be non-negative and finite.  
/// - for `ThermalEnergy` / `TotalVariationTorque`, `normalize.len() == dim`.  
/// - all normalize entries must be non-negative and finite.  
/// - for `Linear`, `alpha.len() == s_len` and `beta.len() == s_len`.
#[inline]
pub(crate) fn validate_copp3_objectives(
    function_name: &str,
    objectives: &[CoppObjective<'_>],
    dim: usize,
    s_len: usize,
) -> Result<(), CoppError> {
    if objectives.is_empty() {
        return Err(CoppError::InvalidInput(
            function_name.into(),
            "objectives must not be empty".into(),
        ));
    }

    for (i, objective) in objectives.iter().enumerate() {
        match objective {
            CoppObjective::Time(weight) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Time(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
            }
            CoppObjective::ThermalEnergy(weight, normalize)
            | CoppObjective::TotalVariationTorque(weight, normalize) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!("objectives[{i}] weight={weight} must be finite and non-negative"),
                    ));
                }
                if normalize.len() != dim {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}] normalize.len()={} must equal robot dim={dim}",
                            normalize.len()
                        ),
                    ));
                }
                for (j, v) in normalize.iter().enumerate() {
                    if v.is_nan() || v.is_infinite() || *v < 0.0 {
                        return Err(CoppError::InvalidInput(
                            function_name.into(),
                            format!(
                                "objectives[{i}] normalize[{j}]={v} must be finite and non-negative"
                            ),
                        ));
                    }
                }
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
                if alpha.len() != s_len || beta.len() != s_len {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear length mismatch: alpha.len()={}, beta.len()={}, expected alpha.len()={s_len}, beta.len()={s_len}",
                            alpha.len(),
                            beta.len(),
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}
