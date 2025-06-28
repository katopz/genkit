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

//! # Status Codes and Types
//!
//! This module defines the status codes and related data structures used for
//! representing the outcome of operations within the Genkit framework. This is
// a port of `statusTypes.ts`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::str::FromStr;
use thiserror::Error;

/// Enumeration of response status codes.
/// This is a direct port of the `StatusCodes` enum in TypeScript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum StatusCode {
    /// Not an error; returned on success.
    ///
    /// HTTP Mapping: 200 OK
    Ok = 0,

    /// The operation was cancelled, typically by the caller.
    ///
    /// HTTP Mapping: 499 Client Closed Request
    Cancelled = 1,

    /// Unknown error. For example, this error may be returned when
    /// a `Status` value received from another address space belongs to
    /// an error space that is not known in this address space. Also
    /// errors raised by APIs that do not return enough error information
    /// may be converted to this error.
    ///
    /// HTTP Mapping: 500 Internal Server Error
    Unknown = 2,

    /// The client specified an invalid argument. Note that this differs
    /// from `FailedPrecondition`. `InvalidArgument` indicates arguments
    /// that are problematic regardless of the state of the system
    /// (e.g., a malformed file name).
    ///
    /// HTTP Mapping: 400 Bad Request
    InvalidArgument = 3,

    /// The deadline expired before the operation could complete. For operations
    /// that change the state of the system, this error may be returned
    // even if the operation has completed successfully. For example, a
    // successful response from a server could have been delayed long
    // enough for the deadline to expire.
    ///
    /// HTTP Mapping: 504 Gateway Timeout
    DeadlineExceeded = 4,

    /// Some requested entity (e.g., file or directory) was not found.
    ///
    /// Note to server developers: if a request is denied for an entire class
    /// of users, such as gradual feature rollout or undocumented allowlist,
    /// `NotFound` may be used. If a request is denied for some users within
    /// a class of users, such as user-based access control, `PermissionDenied`
    /// must be used.
    ///
    /// HTTP Mapping: 404 Not Found
    NotFound = 5,

    /// The entity that a client attempted to create (e.g., file or directory)
    /// already exists.
    ///
    /// HTTP Mapping: 409 Conflict
    AlreadyExists = 6,

    /// The caller does not have permission to execute the specified
    /// operation. `PermissionDenied` must not be used for rejections
    /// caused by exhausting some resource (use `ResourceExhausted`
    /// instead for those errors). `PermissionDenied` must not be
    /// used if the caller can not be identified (use `Unauthenticated`
    /// instead for those errors). This error code does not imply the
    /// request is valid or the requested entity exists or satisfies
    /// other pre-conditions.
    ///
    /// HTTP Mapping: 403 Forbidden
    PermissionDenied = 7,

    /// The request does not have valid authentication credentials for the
    /// operation.
    ///
    /// HTTP Mapping: 401 Unauthorized
    Unauthenticated = 16,

    /// Some resource has been exhausted, perhaps a per-user quota, or
    /// perhaps the entire file system is out of space.
    ///
    /// HTTP Mapping: 429 Too Many Requests
    ResourceExhausted = 8,

    /// The operation was rejected because the system is not in a state
    /// required for the operation's execution. For example, the directory
    /// to be deleted is non-empty, an rmdir operation is applied to
    /// a non-directory, etc.
    ///
    /// Service implementors can use the following guidelines to decide
    /// between `FailedPrecondition`, `Aborted`, and `Unavailable`:
    ///  (a) Use `Unavailable` if the client can retry just the failing call.
    ///  (b) Use `Aborted` if the client should retry at a higher level. For
    ///      example, when a client-specified test-and-set fails, indicating the
    ///      client should restart a read-modify-write sequence.
    ///  (c) Use `FailedPrecondition` if the client should not retry until
    ///      the system state has been explicitly fixed. For example, if an "rmdir"
    ///      fails because the directory is non-empty, `FailedPrecondition`
    ///      should be returned since the client should not retry unless
    ///      the files are deleted from the directory.
    ///
    /// HTTP Mapping: 400 Bad Request
    FailedPrecondition = 9,

    /// The operation was aborted, typically due to a concurrency issue such as
    /// a sequencer check failure or transaction abort.
    ///
    /// See the guidelines above for deciding between `FailedPrecondition`,
    /// `Aborted`, and `Unavailable`.
    ///
    /// HTTP Mapping: 409 Conflict
    Aborted = 10,

    /// The operation was attempted past the valid range. E.g., seeking or
    /// reading past end-of-file.
    ///
    /// Unlike `InvalidArgument`, this error indicates a problem that may
    /// be fixed if the system state changes. For example, a 32-bit file
    /// system will generate `InvalidArgument` if asked to read at an
    /// offset that is not in the range [0,2^32-1], but it will generate
    /// `OutOfRange` if asked to read from an offset past the current
    /// file size.
    ///
    /// There is a fair bit of overlap between `FailedPrecondition` and
    /// `OutOfRange`. We recommend using `OutOfRange` (the more specific
    /// error) when it applies so that callers who are iterating through
    /// a space can easily look for an `OutOfRange` error to detect when
    /// they are done.
    ///
    /// HTTP Mapping: 400 Bad Request
    OutOfRange = 11,

    /// The operation is not implemented or is not supported/enabled in this
    /// service.
    ///
    /// HTTP Mapping: 501 Not Implemented
    Unimplemented = 12,

    /// Internal errors. This means that some invariants expected by the
    /// underlying system have been broken. This error code is reserved
    /// for serious errors.
    ///
    /// HTTP Mapping: 500 Internal Server Error
    Internal = 13,

    /// The service is currently unavailable. This is most likely a
    /// transient condition, which can be corrected by retrying with
    /// a backoff. Note that it is not always safe to retry
    /// non-idempotent operations.
    ///
    /// See the guidelines above for deciding between `FailedPrecondition`,
    /// `Aborted`, and `Unavailable`.
    ///
    /// HTTP Mapping: 503 Service Unavailable
    Unavailable = 14,

    /// Unrecoverable data loss or corruption.
    ///
    /// HTTP Mapping: 500 Internal Server Error
    DataLoss = 15,
}

impl Serialize for StatusCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u16(*self as u16)
    }
}

impl<'de> Deserialize<'de> for StatusCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u16::deserialize(deserializer)?;
        match value {
            0 => Ok(StatusCode::Ok),
            1 => Ok(StatusCode::Cancelled),
            2 => Ok(StatusCode::Unknown),
            3 => Ok(StatusCode::InvalidArgument),
            4 => Ok(StatusCode::DeadlineExceeded),
            5 => Ok(StatusCode::NotFound),
            6 => Ok(StatusCode::AlreadyExists),
            7 => Ok(StatusCode::PermissionDenied),
            8 => Ok(StatusCode::ResourceExhausted),
            9 => Ok(StatusCode::FailedPrecondition),
            10 => Ok(StatusCode::Aborted),
            11 => Ok(StatusCode::OutOfRange),
            12 => Ok(StatusCode::Unimplemented),
            13 => Ok(StatusCode::Internal),
            14 => Ok(StatusCode::Unavailable),
            15 => Ok(StatusCode::DataLoss),
            16 => Ok(StatusCode::Unauthenticated),
            _ => Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Unsigned(value as u64),
                &"a valid StatusCode integer",
            )),
        }
    }
}

/// Represents the status of an operation, often used in API responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    /// The status code for this operation.
    pub code: StatusCode,
    /// A developer-facing error message.
    pub message: String,
    /// Indicates that the request was blocked.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_blocked: bool,
    /// Additional details about the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Status {}

impl Display for StatusCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let s = match self {
            StatusCode::Ok => "OK",
            StatusCode::Cancelled => "CANCELLED",
            StatusCode::Unknown => "UNKNOWN",
            StatusCode::InvalidArgument => "INVALID_ARGUMENT",
            StatusCode::DeadlineExceeded => "DEADLINE_EXCEEDED",
            StatusCode::NotFound => "NOT_FOUND",
            StatusCode::AlreadyExists => "ALREADY_EXISTS",
            StatusCode::PermissionDenied => "PERMISSION_DENIED",
            StatusCode::Unauthenticated => "UNAUTHENTICATED",
            StatusCode::ResourceExhausted => "RESOURCE_EXHAUSTED",
            StatusCode::FailedPrecondition => "FAILED_PRECONDITION",
            StatusCode::Aborted => "ABORTED",
            StatusCode::OutOfRange => "OUT_OF_RANGE",
            StatusCode::Unimplemented => "UNIMPLEMENTED",
            StatusCode::Internal => "INTERNAL",
            StatusCode::Unavailable => "UNAVAILABLE",
            StatusCode::DataLoss => "DATA_LOSS",
        };
        write!(f, "{}", s)
    }
}

/// An error returned when parsing a `StatusCode` from a string fails.
#[derive(Debug, Error)]
#[error("invalid status name: {0}")]
pub struct ParseStatusError(String);

impl FromStr for StatusCode {
    type Err = ParseStatusError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "OK" => Ok(StatusCode::Ok),
            "CANCELLED" => Ok(StatusCode::Cancelled),
            "UNKNOWN" => Ok(StatusCode::Unknown),
            "INVALID_ARGUMENT" => Ok(StatusCode::InvalidArgument),
            "DEADLINE_EXCEEDED" => Ok(StatusCode::DeadlineExceeded),
            "NOT_FOUND" => Ok(StatusCode::NotFound),
            "ALREADY_EXISTS" => Ok(StatusCode::AlreadyExists),
            "PERMISSION_DENIED" => Ok(StatusCode::PermissionDenied),
            "UNAUTHENTICATED" => Ok(StatusCode::Unauthenticated),
            "RESOURCE_EXHAUSTED" => Ok(StatusCode::ResourceExhausted),
            "FAILED_PRECONDITION" => Ok(StatusCode::FailedPrecondition),
            "ABORTED" => Ok(StatusCode::Aborted),
            "OUT_OF_RANGE" => Ok(StatusCode::OutOfRange),
            "UNIMPLEMENTED" => Ok(StatusCode::Unimplemented),
            "INTERNAL" => Ok(StatusCode::Internal),
            "UNAVAILABLE" => Ok(StatusCode::Unavailable),
            "DATA_LOSS" => Ok(StatusCode::DataLoss),
            _ => Err(ParseStatusError(s.to_string())),
        }
    }
}

impl StatusCode {
    /// Converts a `StatusCode` to its corresponding HTTP status code.
    pub fn to_http_status(self) -> u16 {
        match self {
            StatusCode::Ok => 200,
            StatusCode::Cancelled => 499,
            StatusCode::Unknown => 500,
            StatusCode::InvalidArgument => 400,
            StatusCode::DeadlineExceeded => 504,
            StatusCode::NotFound => 404,
            StatusCode::AlreadyExists => 409,
            StatusCode::PermissionDenied => 403,
            StatusCode::Unauthenticated => 401,
            StatusCode::ResourceExhausted => 429,
            StatusCode::FailedPrecondition => 400,
            StatusCode::Aborted => 409,
            StatusCode::OutOfRange => 400,
            StatusCode::Unimplemented => 501,
            StatusCode::Internal => 500,
            StatusCode::Unavailable => 503,
            StatusCode::DataLoss => 500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_status_code_serialization() {
        assert_eq!(serde_json::to_string(&StatusCode::Ok).unwrap(), "0");
        assert_eq!(
            serde_json::to_string(&StatusCode::Unauthenticated).unwrap(),
            "16"
        );
    }

    #[test]
    fn test_status_code_deserialization() {
        assert_eq!(
            serde_json::from_str::<StatusCode>("0").unwrap(),
            StatusCode::Ok
        );
        assert_eq!(
            serde_json::from_str::<StatusCode>("16").unwrap(),
            StatusCode::Unauthenticated
        );
    }

    #[test]
    fn test_status_serialization() {
        let status = Status {
            code: StatusCode::NotFound,
            message: "Entity not found".to_string(),
            details: Some(json!({"resource": "item/123"})),
            is_blocked: false,
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(
            json,
            json!({
                "code": 5,
                "message": "Entity not found",
                "details": {
                    "resource": "item/123"
                }
            })
        );

        let blocked_status = Status {
            code: StatusCode::ResourceExhausted,
            message: "Blocked".to_string(),
            is_blocked: true,
            details: None,
        };
        let blocked_json = serde_json::to_value(&blocked_status).unwrap();
        assert_eq!(
            blocked_json,
            json!({
                "code": 8,
                "message": "Blocked",
                "is_blocked": true,
            })
        );
    }

    #[test]
    fn test_status_deserialization() {
        let data = json!({
            "code": 7,
            "message": "Forbidden"
        });
        let status: Status = serde_json::from_value(data).unwrap();
        assert_eq!(
            status,
            Status {
                code: StatusCode::PermissionDenied,
                message: "Forbidden".to_string(),
                is_blocked: false,
                details: None,
            }
        );
    }

    #[test]
    fn test_to_http_status() {
        assert_eq!(StatusCode::Ok.to_http_status(), 200);
        assert_eq!(StatusCode::NotFound.to_http_status(), 404);
        assert_eq!(StatusCode::Internal.to_http_status(), 500);
    }

    #[test]
    fn test_display_and_from_str() {
        let status_str = "PERMISSION_DENIED";
        let status_code = StatusCode::PermissionDenied;

        assert_eq!(status_code.to_string(), status_str);
        assert_eq!(StatusCode::from_str(status_str).unwrap(), status_code);

        let invalid_str = "INVALID_STATUS";
        assert!(StatusCode::from_str(invalid_str).is_err());
    }
}
