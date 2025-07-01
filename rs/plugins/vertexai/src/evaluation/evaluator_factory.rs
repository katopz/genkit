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

//! # Evaluator Factory
//!
//! This module provides a factory for creating evaluators for the Vertex AI evaluation service.

use super::types::VertexAIEvaluationMetricType;
use crate::common::DerivedParams;
use crate::{Error, Result};
use genkit_ai::evaluator::{
    define_evaluator, BaseEvalDataPoint, EvalResponse, EvaluatorAction, Score,
};
use genkit_core::Registry;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;

/// A factory for creating Vertex AI evaluator actions.
#[derive(Clone)]
pub struct EvaluatorFactory {
    params: Arc<DerivedParams>,
}

impl EvaluatorFactory {
    /// Creates a new `EvaluatorFactory`.
    pub fn new(params: Arc<DerivedParams>) -> Self {
        Self { params }
    }

    /// Creates a new `EvaluatorAction` for a given metric.
    pub fn create<Req, Resp, ToRequest, ResponseHandler>(
        &self,
        metric: VertexAIEvaluationMetricType,
        _display_name: &'static str,
        _definition: &'static str,
        to_request: ToRequest,
        response_handler: ResponseHandler,
    ) -> EvaluatorAction
    where
        Req: Serialize + Send + Sync + 'static,
        Resp: DeserializeOwned + Send + Sync + 'static,
        ToRequest: Fn(BaseEvalDataPoint) -> Req + Send + Sync + 'static,
        ResponseHandler: Fn(Resp) -> Score + Send + Sync + 'static,
    {
        let factory = self.clone();
        let to_request = Arc::new(to_request);
        let response_handler = Arc::new(response_handler);
        let metric_name = metric.to_string();
        let registry = Registry::new();

        (*define_evaluator(
            &mut registry,
            &format!("vertexai/{}", metric_name.to_lowercase()),
            move |datapoint: BaseEvalDataPoint| {
                let factory_clone = factory.clone();
                let to_request_clone = to_request.clone();
                let response_handler_clone = response_handler.clone();

                async move {
                    let test_case_id = datapoint.test_case_id.clone();
                    let request_data = to_request_clone(datapoint);
                    let response = factory_clone
                        .evaluate_instances::<Req, Resp>(request_data)
                        .await?;
                    let score = response_handler_clone(response);
                    Ok(EvalResponse {
                        evaluation: score,
                        test_case_id,
                        trace_id: None,
                    })
                }
            },
        ))
        .clone()
    }

    /// Calls the `evaluateInstances` endpoint of the Vertex AI API.
    async fn evaluate_instances<Req, Resp>(&self, partial_request: Req) -> Result<Resp>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let location_name = format!(
            "projects/{}/locations/{}",
            self.params.project_id, self.params.location
        );
        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1beta1/{}:evaluateInstances",
            self.params.location, location_name
        );

        let token = self
            .params
            .token_provider
            .token(&["https://www.googleapis.com/auth/cloud-platform"])
            .await
            .map_err(|e| Error::GcpAuth(e.to_string()))?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", token.as_str()).parse().unwrap(),
        );

        let client = reqwest::Client::new();
        let request_body = serde_json::to_value(&partial_request)
            .map_err(|e| Error::VertexAI(format!("Failed to serialize request body: {}", e)))?;

        let response = client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(Error::VertexAI(format!(
                "Evaluation API request failed: {}",
                error_text
            )));
        }

        response.json::<Resp>().await.map_err(Error::from)
    }
}

impl ToString for VertexAIEvaluationMetricType {
    fn to_string(&self) -> String {
        serde_json::to_string(self)
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }
}
