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

//! # Internal Utilities
//!
//! This module provides internal macros and utilities used throughout the Genkit
//! framework. These are not intended for public use.

/// A macro to implement the `From` trait for a given wrapper type,
/// allowing it to be converted into a `serde_json::Value`. This is useful for
/// passing complex types as `Value` where needed.
#[macro_export]
macro_rules! impl_trait_value {
    // Handles non-generic structs. The `$pt` and `$msg` arguments are currently
    // unused but are kept for compatibility with the original design.
    ($t:ty, $pt:ty, $msg:tt) => {
        impl From<$t> for serde_json::Value {
            fn from(value: $t) -> Self {
                // This will panic if serialization fails. In Genkit, this is
                // generally considered a programmer error, as all action
                // inputs/outputs should be serializable.
                serde_json::to_value(value).unwrap()
            }
        }
    };
    // Handles generic structs like `MyStruct<T>`.
    ($t:ident < $($p:ident),+ >, $pt:ty, $msg:tt) => {
        impl<$($p: serde::Serialize),+> From<$t<$($p),+>> for serde_json::Value {
            fn from(value: $t<$($p),+>) -> Self {
                serde_json::to_value(value).unwrap()
            }
        }
    };
}
