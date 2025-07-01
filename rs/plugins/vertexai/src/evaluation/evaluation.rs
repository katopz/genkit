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

//! # Vertex AI Evaluators
//!
//! This module defines the evaluators for various metrics supported by the
//! Vertex AI evaluation service.

use super::evaluator_factory::EvaluatorFactory;
use super::types::{VertexAIEvaluationMetric, VertexAIEvaluationMetricType};
use genkit_ai::evaluator::{BaseEvalDataPoint, EvaluatorAction, Score};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Helper to convert a serde_json::Value to a String.
fn stringify(input: &Value) -> String {
    if let Some(s) = input.as_str() {
        s.to_string()
    } else {
        input.to_string()
    }
}

/// Creates a vector of evaluator actions based on the provided metrics.
pub fn vertex_evaluators(
    factory: &EvaluatorFactory,
    metrics: &[VertexAIEvaluationMetric],
) -> Vec<EvaluatorAction<BaseEvalDataPoint>> {
    metrics
        .iter()
        .map(|metric| {
            let (metric_type, metric_spec) = match metric {
                VertexAIEvaluationMetric::Type(t) => (t, serde_json::json!({})),
                VertexAIEvaluationMetric::Config(c) => (&c.metric_type, c.metric_spec.clone()),
            };

            match metric_type {
                VertexAIEvaluationMetricType::Bleu => create_bleu_evaluator(factory, metric_spec),
                VertexAIEvaluationMetricType::Rouge => create_rouge_evaluator(factory, metric_spec),
                VertexAIEvaluationMetricType::Fluency => {
                    create_fluency_evaluator(factory, metric_spec)
                }
                VertexAIEvaluationMetricType::Safety => {
                    create_safety_evaluator(factory, metric_spec)
                }
                VertexAIEvaluationMetricType::Groundedness => {
                    create_groundedness_evaluator(factory, metric_spec)
                }
                VertexAIEvaluationMetricType::SummarizationQuality => {
                    create_summarization_quality_evaluator(factory, metric_spec)
                }
                VertexAIEvaluationMetricType::SummarizationHelpfulness => {
                    create_summarization_helpfulness_evaluator(factory, metric_spec)
                }
                VertexAIEvaluationMetricType::SummarizationVerbosity => {
                    create_summarization_verbosity_evaluator(factory, metric_spec)
                }
            }
        })
        .collect()
}

// Common structs for pointwise evaluators
#[derive(Serialize)]
struct PointwiseInstance {
    prediction: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instruction: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PointwiseResult {
    score: f32,
    explanation: String,
    confidence: f32,
}

// BLEU Evaluator
#[derive(Serialize)]
struct BleuInstance {
    prediction: String,
    reference: String,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BleuInput {
    metric_spec: Value,
    instances: Vec<BleuInstance>,
}
#[derive(Serialize)]
#[serde(rename = "bleuInput")]
struct BleuRequest {
    bleu_input: BleuInput,
}
#[derive(Deserialize)]
struct BleuMetricValue {
    score: f32,
}
#[derive(Deserialize)]
struct BleuResults {
    bleu_metric_values: Vec<BleuMetricValue>,
}
#[derive(Deserialize)]
struct BleuResponse {
    bleu_results: BleuResults,
}

fn create_bleu_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::Bleu,
        "BLEU",
        "Computes the BLEU score by comparing the output against the ground truth",
        move |datapoint: BaseEvalDataPoint| BleuRequest {
            bleu_input: BleuInput {
                metric_spec: metric_spec.clone(),
                instances: vec![BleuInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    reference: stringify(&datapoint.reference.unwrap_or_default()),
                }],
            },
        },
        |response: BleuResponse| Score {
            score: Some(serde_json::json!(
                response.bleu_results.bleu_metric_values[0].score
            )),
            details: None,
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// ROUGE Evaluator
#[derive(Serialize)]
struct RougeInstance {
    prediction: String,
    reference: String,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RougeInput {
    metric_spec: Value,
    instances: RougeInstance,
}
#[derive(Serialize)]
#[serde(rename = "rougeInput")]
struct RougeRequest {
    rouge_input: RougeInput,
}
#[derive(Deserialize)]
struct RougeMetricValue {
    score: f32,
}
#[derive(Deserialize)]
struct RougeResults {
    rouge_metric_values: Vec<RougeMetricValue>,
}
#[derive(Deserialize)]
struct RougeResponse {
    rouge_results: RougeResults,
}

fn create_rouge_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::Rouge,
        "ROUGE",
        "Computes the ROUGE score by comparing the output against the ground truth",
        move |datapoint: BaseEvalDataPoint| RougeRequest {
            rouge_input: RougeInput {
                metric_spec: metric_spec.clone(),
                instances: RougeInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    reference: stringify(&datapoint.reference.unwrap_or_default()),
                },
            },
        },
        |response: RougeResponse| Score {
            score: Some(serde_json::json!(
                response.rouge_results.rouge_metric_values[0].score
            )),
            details: None,
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// Fluency Evaluator
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FluencyInput {
    metric_spec: Value,
    instance: PointwiseInstance,
}
#[derive(Serialize)]
#[serde(rename = "fluencyInput")]
struct FluencyRequest {
    fluency_input: FluencyInput,
}
#[derive(Deserialize)]
struct FluencyResponse {
    fluency_result: PointwiseResult,
}

fn create_fluency_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::Fluency,
        "Fluency",
        "Assesses the language mastery of an output",
        move |datapoint: BaseEvalDataPoint| FluencyRequest {
            fluency_input: FluencyInput {
                metric_spec: metric_spec.clone(),
                instance: PointwiseInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    context: None,
                    instruction: None,
                },
            },
        },
        |response: FluencyResponse| Score {
            score: Some(serde_json::json!(response.fluency_result.score)),
            details: Some(serde_json::json!({ "reasoning": response.fluency_result.explanation })),
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// Safety Evaluator
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SafetyInput {
    metric_spec: Value,
    instance: PointwiseInstance,
}
#[derive(Serialize)]
#[serde(rename = "safetyInput")]
struct SafetyRequest {
    safety_input: SafetyInput,
}
#[derive(Deserialize)]
struct SafetyResponse {
    safety_result: PointwiseResult,
}

fn create_safety_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::Safety,
        "Safety",
        "Assesses the level of safety of an output",
        move |datapoint: BaseEvalDataPoint| SafetyRequest {
            safety_input: SafetyInput {
                metric_spec: metric_spec.clone(),
                instance: PointwiseInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    context: None,
                    instruction: None,
                },
            },
        },
        |response: SafetyResponse| Score {
            score: Some(serde_json::json!(response.safety_result.score)),
            details: Some(serde_json::json!({ "reasoning": response.safety_result.explanation })),
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// Groundedness Evaluator
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GroundednessInput {
    metric_spec: Value,
    instance: PointwiseInstance,
}
#[derive(Serialize)]
#[serde(rename = "groundednessInput")]
struct GroundednessRequest {
    groundedness_input: GroundednessInput,
}
#[derive(Deserialize)]
struct GroundednessResponse {
    groundedness_result: PointwiseResult,
}

fn create_groundedness_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::Groundedness,
        "Groundedness",
        "Assesses the ability to provide or reference information included only in the context",
        move |datapoint: BaseEvalDataPoint| GroundednessRequest {
            groundedness_input: GroundednessInput {
                metric_spec: metric_spec.clone(),
                instance: PointwiseInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    context: datapoint.context.map(|c| {
                        c.iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(". ")
                    }),
                    instruction: None,
                },
            },
        },
        |response: GroundednessResponse| Score {
            score: Some(serde_json::json!(response.groundedness_result.score)),
            details: Some(
                serde_json::json!({ "reasoning": response.groundedness_result.explanation }),
            ),
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// Summarization Quality Evaluator
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SummarizationQualityInput {
    metric_spec: Value,
    instance: PointwiseInstance,
}
#[derive(Serialize)]
#[serde(rename = "summarizationQualityInput")]
struct SummarizationQualityRequest {
    summarization_quality_input: SummarizationQualityInput,
}
#[derive(Deserialize)]
struct SummarizationQualityResponse {
    summarization_quality_result: PointwiseResult,
}

fn create_summarization_quality_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::SummarizationQuality,
        "Summarization Quality",
        "Assesses the overall ability to summarize text",
        move |datapoint: BaseEvalDataPoint| SummarizationQualityRequest {
            summarization_quality_input: SummarizationQualityInput {
                metric_spec: metric_spec.clone(),
                instance: PointwiseInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    context: datapoint
                        .context
                        .map(|c| c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(". ")),
                    instruction: Some(stringify(&datapoint.input)),
                },
            },
        },
        |response: SummarizationQualityResponse| Score {
            score: Some(serde_json::json!(
                response.summarization_quality_result.score
            )),
            details: Some(serde_json::json!({ "reasoning": response.summarization_quality_result.explanation })),
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// Summarization Helpfulness Evaluator
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SummarizationHelpfulnessInput {
    metric_spec: Value,
    instance: PointwiseInstance,
}
#[derive(Serialize)]
#[serde(rename = "summarizationHelpfulnessInput")]
struct SummarizationHelpfulnessRequest {
    summarization_helpfulness_input: SummarizationHelpfulnessInput,
}
#[derive(Deserialize)]
struct SummarizationHelpfulnessResponse {
    summarization_helpfulness_result: PointwiseResult,
}

fn create_summarization_helpfulness_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::SummarizationHelpfulness,
        "Summarization Helpfulness",
        "Assesses the ability to provide a summarization, which contains the details necessary to substitute the original text",
        move |datapoint: BaseEvalDataPoint| SummarizationHelpfulnessRequest {
            summarization_helpfulness_input: SummarizationHelpfulnessInput {
                metric_spec: metric_spec.clone(),
                instance: PointwiseInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    context: datapoint
                        .context
                        .map(|c| c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(". ")),
                    instruction: Some(stringify(&datapoint.input)),
                },
            },
        },
        |response: SummarizationHelpfulnessResponse| Score {
            score: Some(serde_json::json!(
                response.summarization_helpfulness_result.score
            )),
            details: Some(serde_json::json!({ "reasoning": response.summarization_helpfulness_result.explanation })),
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}

// Summarization Verbosity Evaluator
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SummarizationVerbosityInput {
    metric_spec: Value,
    instance: PointwiseInstance,
}
#[derive(Serialize)]
#[serde(rename = "summarizationVerbosityInput")]
struct SummarizationVerbosityRequest {
    summarization_verbosity_input: SummarizationVerbosityInput,
}
#[derive(Deserialize)]
struct SummarizationVerbosityResponse {
    summarization_verbosity_result: PointwiseResult,
}

fn create_summarization_verbosity_evaluator(
    factory: &EvaluatorFactory,
    metric_spec: Value,
) -> EvaluatorAction<BaseEvalDataPoint> {
    factory.create(
        VertexAIEvaluationMetricType::SummarizationVerbosity,
        "Summarization Verbosity",
        "Assess the ability to provide a succinct summarization",
        move |datapoint: BaseEvalDataPoint| SummarizationVerbosityRequest {
            summarization_verbosity_input: SummarizationVerbosityInput {
                metric_spec: metric_spec.clone(),
                instance: PointwiseInstance {
                    prediction: datapoint.output.as_ref().map(stringify).unwrap_or_default(),
                    context: datapoint
                        .context
                        .map(|c| c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(". ")),
                    instruction: Some(stringify(&datapoint.input)),
                },
            },
        },
        |response: SummarizationVerbosityResponse| Score {
            score: Some(serde_json::json!(
                response.summarization_verbosity_result.score
            )),
            details: Some(serde_json::json!({ "reasoning": response.summarization_verbosity_result.explanation })),
            status: Some(genkit_ai::evaluator::EvalStatusEnum::Pass),
            error: None,
        },
    )
}
