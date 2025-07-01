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

//! # Context Caching Constants
//!
//! This module contains constants for the context caching feature.

/// Models that support context caching.
pub const CONTEXT_CACHE_SUPPORTED_MODELS: &[&str] = &["gemini-1.5-flash-001", "gemini-1.5-pro-001"];

/// Error messages for invalid arguments.
pub mod invalid_argument_messages {
    pub const MODEL_VERSION: &str = "Model version is required for context caching, supported only in gemini-1.5-flash-001, gemini-1.5-pro-001 models.";
    pub const TOOLS: &str = "Context caching cannot be used simultaneously with tools.";
    pub const CODE_EXECUTION: &str =
        "Context caching cannot be used simultaneously with code execution.";
}

/// Default time-to-live for cached content in seconds.
pub const DEFAULT_TTL: u32 = 300;
