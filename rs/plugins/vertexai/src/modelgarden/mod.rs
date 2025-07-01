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

//! # Vertex AI Model Garden
//!
//! This module provides support for various models available in the Vertex AI Model Garden.

// Declare the modules within the `modelgarden` directory.
mod anthropic;
mod index;
mod mistral;
mod model_garden;
mod openai_compatibility;
pub mod types;

// Expose the public items from the sub-modules.
pub use self::index::vertex_ai_model_garden;
pub use self::types::ModelGardenPluginOptions;

// Re-export model references for easy access.
