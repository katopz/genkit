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

//! # Error and Result types
//!
//! This module defines the primary error and result types for the Genkit framework.
//! It is a Rust-idiomatic translation of `error.ts`.

use crate::schema::ValidationError;
use crate::status::{Status, StatusCode};
use thiserror::Error;

/// A convenient Result type for genkit operations, defaulting to the crate's `Error` type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The primary error type for the Genkit framework.
///
/// This enum uses variants to represent different error conditions, which is a
/// common and idiomatic pattern in Rust, analogous to the different `Error`
// classes in the original TypeScript implementation.
#[derive(Debug, Error)]
pub enum Error {
    /// An error that is safe to return to the end-user in an API response.
    /// This directly maps to `UserFacingError` from the TypeScript implementation.
    #[error("{0}")]
    UserFacing(#[from] Status),

    /// An internal framework error. Details of this error should generally not
    /// be exposed to external users to avoid leaking implementation details.
    /// This is the equivalent of the base `GenkitError` in TypeScript.
    #[error("Internal error: {message}")]
    Internal {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// An error indicating that a feature is being used without the required
    /// stability level enabled. This maps to `UnstableApiError`.
    #[error("Unstable API Error: {0}")]
    UnstableApi(String),

    /// An error originating from schema validation failure.
    #[error(transparent)]
    Validation(#[from] ValidationError),
}

impl Error {
    /// Creates a new `UserFacing` error, which is safe to expose in APIs.
    pub fn new_user_facing(
        status_code: StatusCode,
        message: impl Into<String>,
        details: Option<serde_json::Value>,
    ) -> Self {
        Error::UserFacing(Status {
            code: status_code,
            message: message.into(),
            details,
        })
    }

    /// Creates a new `Internal` error.
    pub fn new_internal(message: impl Into<String>) -> Self {
        Error::Internal {
            message: message.into(),
            source: None,
        }
    }

    /// Creates a new `UnstableApi` error.
    pub fn new_unstable_api(level: &str, feature_message: Option<&str>) -> Self {
        let message = format!(
            "{}This API requires the '{}' stability level.\n\nTo use this feature, please enable it during Genkit initialization.",
            feature_message.map(|m| format!("{} ", m)).unwrap_or_default(),
            level,
        );
        Error::UnstableApi(message)
    }

    /// Returns the corresponding HTTP status code for the error.
    /// This is the equivalent of `getHttpStatus` in TypeScript.
    pub fn http_status(&self) -> u16 {
        match self {
            Error::UserFacing(status) => status.code.to_http_status(),
            Error::Validation(_) => StatusCode::InvalidArgument.to_http_status(),
            _ => StatusCode::Internal.to_http_status(),
        }
    }

    /// Returns a `Status` struct representation of the error, suitable for
    /// serialization in API responses.
    ///
    /// For internal errors, it returns a generic message to avoid leaking
    /// sensitive information. This is equivalent to `getCallableJSON` in TypeScript.
    pub fn as_status(&self) -> Status {
        match self {
            Error::UserFacing(status) => status.clone(),
            Error::Validation(ve) => Status {
                code: StatusCode::InvalidArgument,
                message: "Schema validation failed".to_string(),
                details: Some(serde_json::json!({ "errors": ve.errors() })),
            },
            Error::UnstableApi(msg) => Status {
                code: StatusCode::FailedPrecondition,
                message: msg.clone(),
                details: None,
            },
            _ => Status {
                code: StatusCode::Internal,
                message: "Internal Server Error".to_string(),
                details: None,
            },
        }
    }
}

/// Asserts that the required API stability level is met.
///
/// In Rust, this is often better handled with feature flags at compile time.
/// This runtime check is provided for scenarios requiring dynamic validation.
///
/// # Arguments
///
/// * `registry_stability` - The current stability level of the application (e.g., "stable").
/// * `required_level` - The stability level required by the feature (e.g., "beta").
/// * `feature_message` - An optional message describing the feature.
pub fn assert_unstable(
    registry_stability: &str,
    required_level: &str,
    feature_message: Option<&str>,
) -> Result<()> {
    // A simple model where "beta" is a higher requirement than "stable".
    if registry_stability == "stable" && required_level == "beta" {
        return Err(Error::new_unstable_api(required_level, feature_message));
    }
    Ok(())
}

/// Extracts a user-friendly error message from any error type that implements
/// the `std::error::Error` trait.
///
/// This is equivalent to `getErrorMessage` in the TypeScript implementation.
pub fn get_error_message(e: &dyn std::error::Error) -> String {
    e.to_string()
}
