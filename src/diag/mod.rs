//! Diagnostics, error types, and verbosity utilities.
//!
//! Input: none (foundation module used by other public APIs).
//! Output: common errors, verbosity/logging abstractions, and validation helpers.
//! Scenario: import from here when you need unified error handling ([`crate::diag::CoppError`]) or
//! solver verbosity control across modules.

pub(crate) mod diagnostics;
pub(crate) mod error;

pub use diagnostics::*;
pub use error::*;
