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

//! # Genkit Public API Facade
//!
//! This module re-exports the primary public-facing types, functions, and
//! structs from the underlying `genkit-ai` and `genkit-core` crates, providing
//! a single, convenient entry point for developers using the Genkit framework.

//
// Re-exports from `genkit-ai`
//

pub use genkit_ai::chat::*;
pub use genkit_ai::session::*;
pub use genkit_ai::tool::*;
pub use genkit_ai::*;

//
// Re-exports from `genkit-core`
//
pub use genkit_core::*;
