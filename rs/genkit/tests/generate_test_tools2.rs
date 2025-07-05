//! Copyright 2024 Google LLC
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use genkit::{
    model::{FinishReason, Part, Role},
    Genkit, Model, ToolArgument, ToolConfig,
};
use genkit_ai::{
    model::{CandidateData, GenerateResponseData},
    GenerateOptions, MessageData, ToolRequest,
};

use genkit::ActionContext;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, Mutex};

//
// Tools Tests
//

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
struct TestToolInput {}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

