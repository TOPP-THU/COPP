//! Test-only planar 2-DoF robot with closed-form dynamics.
//!
//! # Method identity
//! This module provides a compact analytic robot model used to validate
//! `inverse_dynamics()` consistency against explicit
//! `M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)` assembly.
//!
//! # Scope
//! The module is compiled only for tests (`#[cfg(test)]` in parent module).

use crate::robot::robot_core::{RobotBasic, RobotTorque};
use nalgebra::DMatrix;

/// Planar 2-link revolute robot with endpoint lumped masses.
///
/// # Assumptions
/// - Link inertias are neglected.
/// - Masses `m1`, `m2` are concentrated at link endpoints.
/// - Joint state is `q = [theta1, theta2]`,
///   velocity is `dq = [dtheta1, dtheta2]`.
pub(crate) struct Plannar2LinkEnd {
    /// Mass associated with first link endpoint.
    pub m1: f64,
    /// Mass associated with second link endpoint.
    pub m2: f64,
    /// Length of first link.
    pub l1: f64,
    /// Length of second link.
    pub l2: f64,
    /// Gravity acceleration magnitude.
    pub g: f64,
}

impl Plannar2LinkEnd {
    /// Construct the 2-link model with default gravity `g = 9.81`.
    ///
    /// # Parameters
    /// - `m1`, `m2`: endpoint masses.
    /// - `l1`, `l2`: link lengths.
    pub fn new(m1: f64, m2: f64, l1: f64, l2: f64) -> Self {
        Self {
            m1,
            m2,
            l1,
            l2,
            g: 9.81,
        }
    }

    /// Fill the joint-space mass matrix `M(q)`.
    ///
    /// # Parameters
    /// - `q`: joint positions (`[theta1, theta2]`).
    /// - `mass`: writable `2x2` matrix buffer.
    fn fill_mass_matrix(&self, q: &[f64], mass: &mut DMatrix<f64>) {
        let q2 = q[1];
        let (_s2, c2) = q2.sin_cos();

        // M11 = m1*l1^2 + m2*(l1^2 + l2^2 + 2*l1*l2*cos(q2))
        // M12 = M21 = m2*(l2^2 + l1*l2*cos(q2))
        // M22 = m2*l2^2
        let m11 = (self.m1 + self.m2) * self.l1.powi(2)
            + self.m2 * self.l2.powi(2)
            + 2.0 * self.m2 * self.l1 * self.l2 * c2;
        let m12 = self.m2 * self.l2.powi(2) + self.m2 * self.l1 * self.l2 * c2;
        let m22 = self.m2 * self.l2.powi(2);

        mass[(0, 0)] = m11;
        mass[(0, 1)] = m12;
        mass[(1, 0)] = m12;
        mass[(1, 1)] = m22;
    }

    /// Fill Coriolis matrix `C(q, dq)`.
    ///
    /// # Parameters
    /// - `q`: joint positions.
    /// - `dq`: joint velocities.
    /// - `coriolis`: writable `2x2` matrix buffer.
    fn fill_coriolis_matrix(&self, q: &[f64], dq: &[f64], coriolis: &mut DMatrix<f64>) {
        let q2 = q[1];
        let dq1 = dq[0];
        let dq2 = dq[1];
        let s2 = q2.sin();

        // C11 = -m2*l1*l2*sin(q2)*dq2
        // C12 = -m2*l1*l2*sin(q2)*(dq1 + dq2)
        // C21 = m2*l1*l2*sin(q2)*dq1
        // C22 = 0
        let h = self.m2 * self.l1 * self.l2 * s2;

        coriolis[(0, 0)] = -h * dq2;
        coriolis[(0, 1)] = -h * (dq1 + dq2);
        coriolis[(1, 0)] = h * dq1;
        coriolis[(1, 1)] = 0.0;
    }

    /// Fill gravity vector `g(q)`.
    ///
    /// # Parameters
    /// - `q`: joint positions.
    /// - `gravity`: writable vector buffer of length `2`.
    fn fill_gravity_vector(&self, q: &[f64], gravity: &mut [f64]) {
        let q1 = q[0];
        let q2 = q[1];
        let (_s1, c1) = q1.sin_cos();
        let (_s12, c12) = (q1 + q2).sin_cos();
        // y-axis is up, gravity is along -y
        let g = [
            (self.m1 + self.m2) * self.g * self.l1 * c1 + self.m2 * self.g * self.l2 * c12,
            self.m2 * self.g * self.l2 * c12,
        ];
        gravity.copy_from_slice(&g);
    }
}

impl RobotBasic for Plannar2LinkEnd {
    /// Return fixed DoF = 2.
    #[inline(always)]
    fn dim(&self) -> usize {
        2
    }
}

impl RobotTorque for Plannar2LinkEnd {
    /// Evaluate inverse dynamics in closed form.
    ///
    /// Implements
    /// `tau = M(q) * ddq + C(q, dq) * dq + g(q)`
    /// with explicit scalar expressions for this 2-link model.
    fn inverse_dynamics(&self, q: &[f64], dq: &[f64], ddq: &[f64], tau: &mut [f64]) {
        let q2 = q[1];
        let q1 = q[0];
        let dq2 = dq[1];
        let dq1 = dq[0];
        let ddq2 = ddq[1];
        let ddq1 = ddq[0];

        let c1 = q1.cos();
        let (s2, c2) = q2.sin_cos();
        let c12 = (q1 + q2).cos();

        // Common factors
        let l1_sq = self.l1 * self.l1;
        let l2_sq = self.l2 * self.l2;
        let m2_l2_sq = self.m2 * l2_sq;
        let m2_l1_l2 = self.m2 * self.l1 * self.l2;
        let h_cos = m2_l1_l2 * c2;
        let h_sin = m2_l1_l2 * s2;

        // Inertia part (only terms needed by tau)
        let i11 = ((self.m1 + self.m2) * l1_sq + m2_l2_sq + 2.0 * h_cos) * ddq1;
        let i12 = (m2_l2_sq + h_cos) * ddq2;
        let i21 = (m2_l2_sq + h_cos) * ddq1;
        let i22 = m2_l2_sq * ddq2;

        // Coriolis/centrifugal part: C(q, dq) * dq
        let c_tau1 = -h_sin * (2.0 * dq1 * dq2 + dq2 * dq2);
        let c_tau2 = h_sin * dq1 * dq1;

        // Gravity part
        let g2 = self.m2 * self.g * self.l2 * c12;
        let g1 = (self.m1 + self.m2) * self.g * self.l1 * c1 + g2;

        // tau = M(q) * ddq + C(q, dq) * dq + g(q)
        tau[0] = i11 + i12 + c_tau1 + g1;
        tau[1] = i21 + i22 + c_tau2 + g2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::robot_core::RobotTorque;
    use rand::RngExt;

    /// Monte-Carlo consistency test:
    /// compares closed-form `inverse_dynamics()` against explicit
    /// `M*ddq + C*dq + g` assembly over random states.
    #[test]
    #[ignore = "slow"]
    fn test_inverse_dynamics_consistency_random() {
        let n_exp = 1000000;
        let tol = 1e-10;

        let robot = Plannar2LinkEnd::new(2.0, 1.5, 0.8, 0.6);
        let mut rng = rand::rng();

        let mut m = DMatrix::<f64>::zeros(2, 2);
        let mut c = DMatrix::<f64>::zeros(2, 2);
        let mut g = [0.0_f64; 2];
        let mut tau_id = [0.0_f64; 2];

        for i_exp in 0..n_exp {
            let q = [
                rng.random_range(-std::f64::consts::PI..std::f64::consts::PI),
                rng.random_range(-std::f64::consts::PI..std::f64::consts::PI),
            ];
            let dq = [rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)];
            let ddq = [rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0)];

            <Plannar2LinkEnd as RobotTorque>::inverse_dynamics(&robot, &q, &dq, &ddq, &mut tau_id);

            Plannar2LinkEnd::fill_mass_matrix(&robot, &q, &mut m);
            Plannar2LinkEnd::fill_coriolis_matrix(&robot, &q, &dq, &mut c);
            Plannar2LinkEnd::fill_gravity_vector(&robot, &q, &mut g);

            let tau_ref0 = m[(0, 0)] * ddq[0]
                + m[(0, 1)] * ddq[1]
                + c[(0, 0)] * dq[0]
                + c[(0, 1)] * dq[1]
                + g[0];
            let tau_ref1 = m[(1, 0)] * ddq[0]
                + m[(1, 1)] * ddq[1]
                + c[(1, 0)] * dq[0]
                + c[(1, 1)] * dq[1]
                + g[1];
            let tau_ref = [tau_ref0, tau_ref1];

            let e0 = (tau_id[0] - tau_ref0).abs();
            let e1 = (tau_id[1] - tau_ref1).abs();
            assert!(
                e0 <= tol && e1 <= tol,
                "Mismatch at sample {i_exp}: e0={e0:.3e}, e1={e1:.3e}, q={q:?}, dq={dq:?}, ddq={ddq:?}, tau_id={tau_id:?}, tau_ref={tau_ref:?}"
            );

            if (i_exp + 1) % 10000 == 0 {
                crate::verbosity_log!(
                    crate::diag::Verbosity::Summary,
                    "Exp {}: e0={e0:.6e}, e1={e1:.6e}",
                    i_exp + 1
                );
            }
        }
    }
}
