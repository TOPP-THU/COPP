//! Robot namespace for model abstractions and demo models.
//!
//! # Method identity
//! This module groups:
//! - core robot traits/wrappers in [`Robot`], [`RobotBasic`], [`RobotTorque`],
//! - lightweight demo/reference robot models in `demo`.
//!
//! # Usage guidance
//! - For most users, prefer [`Robot`] as the primary entry instead of operating
//!   on [`Constraints`](crate::constraints::Constraints) directly. `Robot`
//!   enables physically meaningful high-level constraint APIs.
//! - `Topp*Problem` workflows can use any model implementing [`RobotBasic`].
//! - `Copp*Problem` workflows require models implementing [`RobotTorque`].
//! - If your workflow does not need real dynamics but still expects a
//!   [`RobotTorque`] implementation, use `usize` as a trivial placeholder
//!   model (`tau = ddq`).
//! - For real systems, implement [`RobotTorque`]
//!   yourself with your inverse-dynamics model.
//! - Advanced users who need full manual control can still operate directly on
//!   [`Constraints`](crate::constraints::Constraints).
//!
//! # Export policy
//! - Re-export all public symbols from `robot_core` as primary robot API surface.

pub(crate) mod demo;
pub(crate) mod robot_core;

pub use robot_core::*;
