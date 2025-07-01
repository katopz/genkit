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

//! # Vertex AI Evaluation
//!
//! This module provides the Vertex AI evaluation plugin and related types.

mod evaluation;
mod evaluator_factory;
mod index;
pub mod types;

pub use self::index::vertex_ai_evaluation;
pub use self::types::{
    EvaluationPluginOptions, VertexAIEvaluationMetric, VertexAIEvaluationMetricConfig,
    VertexAIEvaluationMetricType,
};
