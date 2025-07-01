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

//! # Vertex AI Evaluation Plugin
//!
//! This module provides the main plugin for Vertex AI evaluation.

use super::{
    evaluation::vertex_evaluators, evaluator_factory::EvaluatorFactory,
    types::EvaluationPluginOptions,
};
use crate::common::get_derived_params;
use async_trait::async_trait;
use genkit_core::{plugin::Plugin, registry::Registry, Result};
use std::sync::Arc;

/// The Vertex AI evaluation plugin.
#[derive(Debug)]
pub struct VertexAIEvaluationPlugin {
    options: EvaluationPluginOptions,
}

impl VertexAIEvaluationPlugin {
    /// Creates a new `VertexAIEvaluationPlugin`.
    pub fn new(options: EvaluationPluginOptions) -> Self {
        Self { options }
    }
}

#[async_trait]
impl Plugin for VertexAIEvaluationPlugin {
    fn name(&self) -> &'static str {
        "vertexai_evaluation"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let params = Arc::new(get_derived_params(&self.options.common).await?);
        let factory = EvaluatorFactory::new(params);
        let evaluators = vertex_evaluators(&factory, &self.options.evaluation.metrics);
        for evaluator in evaluators {
            registry.register_action(self.name().to_owned(), evaluator)?;
        }
        Ok(())
    }
}

/// Creates an instance of the Vertex AI evaluation plugin.
pub fn vertex_ai_evaluation(options: EvaluationPluginOptions) -> Arc<dyn Plugin> {
    Arc::new(VertexAIEvaluationPlugin::new(options))
}
