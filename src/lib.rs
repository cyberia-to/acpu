//! acpu — pure Rust driver for Apple Silicon CPU compute
//!
//! direct access to every useful compute unit in M1–M4:
//! matrix coprocessor (AMX), vector engine (NEON), numeric
//! extensions, sync primitives, and performance counters.
//!
//! zero external dependencies. only inline assembly and system calls.

pub mod convert;
pub mod gemm;
pub mod matrix;
pub mod numeric;
pub mod probe;
pub mod pulse;
pub mod sync;
pub mod vector;

pub use convert::{cvt_bf16_f32, cvt_f16_f32, cvt_f32_bf16, cvt_f32_f16, cvt_f32_i8, cvt_i8_f32};
pub use gemm::{bgemm, hgemm, qgemm, sgemm};
pub use matrix::AmxCtx;
pub use numeric::{bf16, complex, fp16, quant};
pub use probe::{detect, Caps, Chip, Feature};
pub use pulse::PulseCtx;
pub use sync::{affinity, prefetch};

use std::fmt;

/// all acpu errors
#[derive(Debug)]
pub enum RamxError {
    /// running on non-Apple-Silicon hardware
    ChipNotSupported,
    /// AMX_SET instruction failed
    AmxSetFailed,
    /// AMX operation error
    AmxOpFailed(String),
    /// libkperf.dylib not found or kpc access denied
    PmuNotAvailable,
    /// PMU counter configuration rejected
    PmuConfigFailed(String),
    /// required CPU extension absent on this chip
    FeatureNotAvailable(Feature),
    /// QoS class change failed
    AffinityFailed(String),
    /// sysctl query failed
    SysctlFailed(String),
}

impl fmt::Display for RamxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChipNotSupported => write!(f, "not Apple Silicon"),
            Self::AmxSetFailed => write!(f, "AMX_SET failed"),
            Self::AmxOpFailed(s) => write!(f, "AMX op failed: {s}"),
            Self::PmuNotAvailable => write!(f, "PMU not available (libkperf.dylib)"),
            Self::PmuConfigFailed(s) => write!(f, "PMU config failed: {s}"),
            Self::FeatureNotAvailable(feat) => write!(f, "feature not available: {feat:?}"),
            Self::AffinityFailed(s) => write!(f, "affinity failed: {s}"),
            Self::SysctlFailed(s) => write!(f, "sysctl failed: {s}"),
        }
    }
}

impl std::error::Error for RamxError {}

pub type Result<T> = std::result::Result<T, RamxError>;
