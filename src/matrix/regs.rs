//! Typed AMX register row indices.
//!
//! The AMX coprocessor exposes three register files (X, Y, Z), each
//! containing 8 rows of 64 bytes. These newtypes enforce the valid
//! range (0..=7) at construction time so downstream code can rely on
//! the index being in-bounds.

use crate::RamxError;
use core::fmt;

// ---------------------------------------------------------------------------
// XRow
// ---------------------------------------------------------------------------

/// Index of an X-register row (0..=7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct XRow(u8);

impl XRow {
    /// Create a new X-row index.
    ///
    /// # Errors
    ///
    /// Returns `RamxError::AmxOpFailed` if `index > 7`.
    #[inline]
    pub fn new(index: u8) -> crate::Result<Self> {
        if index > 7 {
            Err(RamxError::AmxOpFailed(format!(
                "XRow index {index} out of range 0..=7"
            )))
        } else {
            Ok(Self(index))
        }
    }

    /// Create a new X-row index without bounds checking.
    ///
    /// # Safety
    ///
    /// `index` must be in `0..=7`.
    #[inline]
    pub const unsafe fn new_unchecked(index: u8) -> Self {
        Self(index)
    }

    /// Return the raw row index.
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl fmt::Display for XRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "X{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// YRow
// ---------------------------------------------------------------------------

/// Index of a Y-register row (0..=7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct YRow(u8);

impl YRow {
    /// Create a new Y-row index.
    ///
    /// # Errors
    ///
    /// Returns `RamxError::AmxOpFailed` if `index > 7`.
    #[inline]
    pub fn new(index: u8) -> crate::Result<Self> {
        if index > 7 {
            Err(RamxError::AmxOpFailed(format!(
                "YRow index {index} out of range 0..=7"
            )))
        } else {
            Ok(Self(index))
        }
    }

    /// Create a new Y-row index without bounds checking.
    ///
    /// # Safety
    ///
    /// `index` must be in `0..=7`.
    #[inline]
    pub const unsafe fn new_unchecked(index: u8) -> Self {
        Self(index)
    }

    /// Return the raw row index.
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl fmt::Display for YRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Y{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ZRow
// ---------------------------------------------------------------------------

/// Index of a Z-register row (0..=7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ZRow(u8);

impl ZRow {
    /// Create a new Z-row index.
    ///
    /// # Errors
    ///
    /// Returns `RamxError::AmxOpFailed` if `index > 7`.
    #[inline]
    pub fn new(index: u8) -> crate::Result<Self> {
        if index > 7 {
            Err(RamxError::AmxOpFailed(format!(
                "ZRow index {index} out of range 0..=7"
            )))
        } else {
            Ok(Self(index))
        }
    }

    /// Create a new Z-row index without bounds checking.
    ///
    /// # Safety
    ///
    /// `index` must be in `0..=7`.
    #[inline]
    pub const unsafe fn new_unchecked(index: u8) -> Self {
        Self(index)
    }

    /// Return the raw row index.
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl fmt::Display for ZRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Z{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Convenience const arrays for iteration
// ---------------------------------------------------------------------------

/// All 8 X-register rows, in order.
pub const ALL_X: [XRow; 8] = [
    XRow(0),
    XRow(1),
    XRow(2),
    XRow(3),
    XRow(4),
    XRow(5),
    XRow(6),
    XRow(7),
];

/// All 8 Y-register rows, in order.
pub const ALL_Y: [YRow; 8] = [
    YRow(0),
    YRow(1),
    YRow(2),
    YRow(3),
    YRow(4),
    YRow(5),
    YRow(6),
    YRow(7),
];

/// All 8 Z-register rows, in order.
pub const ALL_Z: [ZRow; 8] = [
    ZRow(0),
    ZRow(1),
    ZRow(2),
    ZRow(3),
    ZRow(4),
    ZRow(5),
    ZRow(6),
    ZRow(7),
];
