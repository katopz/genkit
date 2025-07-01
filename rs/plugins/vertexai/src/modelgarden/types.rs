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

//! # Types for Vertex AI Model Garden
//!
//! This module defines the data structures for configuring the Model Garden plugin.

use crate::common::VertexAIPluginOptions;
use genkit_ai::ModelRef;
use serde::Deserialize;

/// Options specific to Model Garden configuration.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelGardenOptions {
    /// A list of Model Garden models to register.
    #[serde(default)]
    pub models: Vec<ModelRef>,
    /// A template for the OpenAI-compatible base URL.
    ///
    /// This is used for models that expose an OpenAI-compatible endpoint.
    /// The template can contain `{location}` and `{projectId}` placeholders.
    pub open_ai_base_url_template: Option<String>,
}

/// Plugin options for the Vertex AI Model Garden.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelGardenPluginOptions {
    #[serde(flatten)]
    pub common: VertexAIPluginOptions,
    #[serde(flatten)]
    pub model_garden: ModelGardenOptions,
}
