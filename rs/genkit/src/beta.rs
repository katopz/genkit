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

// This file serves as the root of the `beta` module.
//
// Declare the submodules. Rust will look for `src/beta/client.rs` and `src/beta/genkit_beta.rs`.
pub mod client;
pub mod genkit_beta;

// Re-export key items for easier access from `genkit::beta::*`.
pub use self::genkit_beta::{genkit, GenkitBeta, GenkitBetaOptions};
pub use crate::common::*;
