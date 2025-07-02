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

//! # Types for Vertex AI Evaluation
//!
//! This module defines the data structures used for configuring and running
//! evaluations with the Vertex AI evaluation service.

use crate::common::VertexAIPluginOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use strum::Display;

/// The type of metric to be used for evaluation.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Display)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum VertexAIEvaluationMetricType {
    Bleu,
    Rouge,
    Fluency,
    Safety,
    Groundedness,
    SummarizationQuality,
    SummarizationHelpfulness,
    SummarizationVerbosity,
}

/// Detailed configuration for an evaluation metric.
#[derive(Debug, Deserialize, Clone)]
pub struct VertexAIEvaluationMetricConfig {
    #[serde(rename = "type")]
    pub metric_type: VertexAIEvaluationMetricType,
    /// The `metricSpec` defines the behavior of the metric. The value will be
    /// included in the request to the API. See the API documentation for details
    /// on the possible values for each metric:
    /// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation#parameter-list
    #[serde(default)]
    pub metric_spec: Value,
}

/// Represents an evaluation metric, which can be specified as a simple type
/// or with a detailed configuration.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum VertexAIEvaluationMetric {
    /// A simple metric type.
    Type(VertexAIEvaluationMetricType),
    /// A detailed metric configuration.
    Config(VertexAIEvaluationMetricConfig),
}

/// Options specific to evaluation configuration.
#[derive(Debug, Deserialize, Default)]
pub struct EvaluationOptions {
    /// The list of metrics to be used for evaluation.
    pub metrics: Vec<VertexAIEvaluationMetric>,
}

/// Plugin options for the Vertex AI evaluation service.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationPluginOptions {
    #[serde(flatten)]
    pub common: VertexAIPluginOptions,
    #[serde(flatten)]
    pub evaluation: EvaluationOptions,
}
