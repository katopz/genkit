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

//! # Model and Flow Evaluators
//!
//! This module provides the core structures and functions for evaluating
//! generative models and AI flows. It is the Rust equivalent of `evaluator.ts`.

use async_trait::async_trait;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

//
// Core Data Structures
//

/// Represents a single data point for an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BaseDataPoint {
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_case_id: Option<String>,
}

/// A data point guaranteed to have a `testCaseId`, used within evaluator actions.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BaseEvalDataPoint {
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<Value>,
    pub test_case_id: String,
}

/// A type alias for a dataset, which is a collection of data points.
pub type Dataset = Vec<BaseDataPoint>;

/// The pass/fail status of an evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum EvalStatusEnum {
    #[serde(rename = "UNKNOWN")]
    Unknown,
    #[serde(rename = "PASS")]
    Pass,
    #[serde(rename = "FAIL")]
    Fail,
}

/// The result of a single evaluation metric.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Score {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<EvalStatusEnum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// The evaluation response for a single test case.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EvalResponse {
    pub test_case_id: String,
    pub evaluation: Score,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    // Other fields like sampleIndex, spanId would be added in a full trace-aware implementation.
}

/// A collection of evaluation responses for a dataset.
pub type EvalResponses = Vec<EvalResponse>;

//
// Action & Function Types
//

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EvalRequest<O = Value> {
    pub dataset: Dataset,
    pub eval_run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

/// A function that implements the logic for an evaluator.
pub type EvaluatorFn = dyn Fn(BaseEvalDataPoint) -> Box<dyn Future<Output = Result<EvalResponse>> + Send>
    + Send
    + Sync;

/// A wrapper for an evaluator `Action`.
#[derive(Clone)]
pub struct EvaluatorAction<I = Value>(pub Action<EvalRequest<I>, EvalResponses, ()>);

impl<I: 'static> Deref for EvaluatorAction<I> {
    type Target = Action<EvalRequest<I>, EvalResponses, ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl<I> ErasedAction for EvaluatorAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
{
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<Value> {
        self.0.run_http_json(input, context).await
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<genkit_core::action::StreamingResponse<Value, Value>> {
        self.0.stream_http_json(input, context)
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn metadata(&self) -> &genkit_core::action::ActionMetadata {
        self.0.metadata()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Descriptive information about an evaluator.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EvaluatorInfo {
    pub label: String,
    pub metrics: Vec<String>,
}

//
// Define Evaluator
//

/// Defines a new evaluator and registers it.
pub fn define_evaluator<I, F, Fut>(
    registry: &mut Registry,
    name: &str,
    runner: F,
) -> EvaluatorAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
    F: FnMut(BaseEvalDataPoint) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<EvalResponse>> + Send + 'static,
{
    let runner = Arc::new(Mutex::new(runner));
    let action = ActionBuilder::new(
        ActionType::Evaluator,
        name.to_string(),
        move |req: EvalRequest<I>, _context| {
            let runner_clone = runner.clone();
            let eval_futures = req
                .dataset
                .into_iter()
                .map(|dp| {
                    let runner_clone_inner = runner_clone.clone();
                    let test_case_id = dp
                        .test_case_id
                        .unwrap_or_else(|| Uuid::new_v4().to_string());
                    let eval_dp = BaseEvalDataPoint {
                        test_case_id,
                        input: dp.input,
                        output: dp.output,
                        context: dp.context,
                        reference: dp.reference,
                    };
                    // Synchronously acquire lock, call runner, and drop lock.
                    let inner_fut = {
                        let mut runner_guard = runner_clone_inner.lock().unwrap();
                        (runner_guard)(eval_dp)
                    }; // `runner_guard` is dropped here.

                    // Explicitly Box::pin the async move block
                    Box::pin(inner_fut)
                        as Pin<Box<dyn Future<Output = Result<EvalResponse>> + Send>>
                })
                .collect::<Vec<_>>();

            async move {
                let results = futures_util::future::join_all(eval_futures).await;
                // Note: This basic version doesn't handle individual errors well.
                let successful_results: Vec<EvalResponse> =
                    results.into_iter().filter_map(Result::ok).collect();
                Ok(successful_results)
            }
        },
    )
    .build();
    let evaluator_action = EvaluatorAction(action);
    registry
        .register_action(name.to_string(), evaluator_action.clone())
        .expect("Failed to register evaluator");
    evaluator_action
}

//
// High-Level API (`evaluate`)
//

/// A reference to an evaluator.
#[derive(Clone)]
pub enum EvaluatorArgument {
    Name(String),
    Action(Arc<EvaluatorAction>),
}

/// Parameters for the `evaluate` function.
pub struct EvaluatorParams {
    pub evaluator: EvaluatorArgument,
    pub dataset: Dataset,
    pub eval_run_id: Option<String>,
    pub options: Option<Value>,
}

/// Runs an evaluator on a dataset.
pub async fn evaluate(registry: &Registry, params: EvaluatorParams) -> Result<EvalResponses> {
    let evaluator_action = match params.evaluator {
        EvaluatorArgument::Name(name) => registry
            .lookup_action(&format!("/evaluator/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Evaluator '{}' not found", name)))?,
        EvaluatorArgument::Action(action) => action,
    };

    let request = EvalRequest {
        dataset: params.dataset,
        eval_run_id: params
            .eval_run_id
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize eval request: {}", e)))?;
    let response_value = evaluator_action.run_http_json(request_value, None).await?;
    let response: EvalResponses = serde_json::from_value(response_value)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize eval response: {}", e)))?;
    Ok(response)
}

//
// Reference Helpers
//

/// A serializable reference to an evaluator.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EvaluatorRef {
    pub name: String,
    // config and info would be here in a full port
}

/// Helper to create an `EvaluatorRef`.
pub fn evaluator_ref(name: &str) -> EvaluatorRef {
    EvaluatorRef {
        name: name.to_string(),
    }
}
