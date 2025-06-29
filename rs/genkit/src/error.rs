// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Genkit Error and Result types
//!
//! This module defines the standard error and result types used throughout the
//! `genkit` crate.

use std::fmt;

/// A specialized `Result` type for Genkit operations.
///
/// This type alias is used across the crate for functions that can return
/// a Genkit-specific error.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error type for the Genkit crate.
///
/// This enum consolidates various potential failure modes from dependencies
/// (like `reqwest` for HTTP and `serde_json` for serialization) and internal
/// application logic into a single, consistent error type.
#[derive(Debug)]
pub enum Error {
    /// An internal error, often with a descriptive message.
    /// This is used for application-specific failures that don't fit into
    /// other categories.
    Internal(String),
    /// An error originating from the underlying HTTP client (`reqwest`).
    Reqwest(reqwest::Error),
    /// An error during JSON serialization or deserialization.
    Json(serde_json::Error),
    /// An error during UTF-8 string conversion from a byte sequence.
    Utf8(std::string::FromUtf8Error),
    /// An error for an invalid HTTP header value.
    InvalidHeaderValue(reqwest::header::InvalidHeaderValue),
    /// An error for an invalid HTTP header name.
    InvalidHeaderName(reqwest::header::InvalidHeaderName),
    /// An error from a `tokio` task, e.g., when a spawned task panics or fails.
    JoinError(tokio::task::JoinError),
    /// An error when a required feature is not supported.
    NotSupported(String),
}

impl Error {
    /// Creates a new `Error::Internal` with a message.
    pub fn new_internal<S: Into<String>>(message: S) -> Self {
        Error::Internal(message.into())
    }

    /// Creates a new `Error::NotSupported` with a message.
    pub fn new_not_supported<S: Into<String>>(feature: S) -> Self {
        Error::NotSupported(format!("{} is not supported", feature.into()))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Internal(msg) => write!(f, "Internal Genkit error: {}", msg),
            Error::Reqwest(err) => write!(f, "HTTP request error: {}", err),
            Error::Json(err) => write!(f, "JSON serialization/deserialization error: {}", err),
            Error::Utf8(err) => write!(f, "UTF-8 conversion error: {}", err),
            Error::InvalidHeaderValue(err) => write!(f, "Invalid HTTP header value: {}", err),
            Error::InvalidHeaderName(err) => write!(f, "Invalid HTTP header name: {}", err),
            Error::JoinError(err) => write!(f, "Tokio task error: {}", err),
            Error::NotSupported(msg) => write!(f, "Feature not supported: {}", msg),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Reqwest(err) => Some(err),
            Error::Json(err) => Some(err),
            Error::Utf8(err) => Some(err),
            Error::InvalidHeaderValue(err) => Some(err),
            Error::InvalidHeaderName(err) => Some(err),
            Error::JoinError(err) => Some(err),
            Error::Internal(_) | Error::NotSupported(_) => None,
        }
    }
}

// From trait implementations to allow for easy error conversion with `?`.

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Reqwest(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err)
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(err: std::string::FromUtf8Error) -> Self {
        Error::Utf8(err)
    }
}

impl From<reqwest::header::InvalidHeaderValue> for Error {
    fn from(err: reqwest::header::InvalidHeaderValue) -> Self {
        Error::InvalidHeaderValue(err)
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(err: tokio::task::JoinError) -> Self {
        Error::JoinError(err)
    }
}

impl From<reqwest::header::InvalidHeaderName> for Error {
    fn from(err: reqwest::header::InvalidHeaderName) -> Self {
        Error::InvalidHeaderName(err)
    }
}
