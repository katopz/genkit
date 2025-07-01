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

//! # Types for Context Caching
//!
//! This module defines the data structures for configuring context caching.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Represents the configuration for context caching.
///
/// It can be a simple boolean to enable/disable caching, or an object
/// specifying more details like `ttl_seconds`.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, PartialEq)]
#[serde(untagged)]
pub enum CacheConfig {
    /// A simple boolean flag to enable or disable caching.
    Boolean(bool),
    /// An object with configuration details, such as TTL.
    Object {
        #[serde(rename = "ttlSeconds", skip_serializing_if = "Option::is_none")]
        ttl_seconds: Option<u32>,
    },
}

/// Contains the details needed to process a context cache request.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CacheConfigDetails {
    /// The cache configuration.
    pub cache_config: CacheConfig,
    /// The index of the last message in the request that should be included
    /// in the cached content.
    pub end_of_cached_contents: usize,
}
