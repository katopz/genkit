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

//! # Model Tester
//!
//! This module provides a suite of tests to run against generative models to
//! verify their capabilities and conformance. It is the Rust equivalent of
// `testing/model-tester.ts`.

use crate::generate::{generate, GenerateOptions};
use crate::model;

use genkit_core::error::Result;
use genkit_core::registry::Registry;

use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

//
// Test Reporting Structs
//

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TestError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelReport {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<TestError>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TestReport {
    pub description: String,
    pub models: Vec<ModelReport>,
}

//
// Test Case Definition
//

/// The signature for a single test case function.
type TestCase = fn(&Registry, String) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

//
// Test Suite
//

async fn test_basic_hi(registry: &Registry, model: String) -> Result<()> {
    let response = generate::<Value>(
        registry,
        GenerateOptions {
            model: Some(model::Model::Name(model)),
            prompt: Some(vec![crate::document::Part {
                text: Some("just say \"Hi\", literally".to_string()),
                ..Default::default()
            }]),
            ..Default::default()
        },
    )
    .await?;

    let got = response.text()?.trim().to_lowercase();
    assert!(got.contains("hi"), "Expected 'Hi', got '{}'", got);
    Ok(())
}

// Additional test cases (multimodal, history, etc.) would be defined here.
// They are omitted for brevity in this initial port but would follow the same
// async fn(&Registry, String) -> Result<()> pattern.

/// Runs a suite of conformance tests against a list of models.
pub async fn test_models(registry: &Registry, models: &[String]) -> Result<Vec<TestReport>> {
    // In a full implementation, a tool would be defined here for the tool calling test.
    // tool::define_tool(registry, "gablorkenTool", ...);

    let mut tests: HashMap<&str, TestCase> = HashMap::new();
    tests.insert("basic hi", |r, m| Box::pin(test_basic_hi(r, m)));
    // Additional tests would be inserted here.

    let mut reports = Vec::new();

    for (description, test_fn) in tests {
        let mut model_reports = Vec::new();
        for model_name in models {
            let mut report = ModelReport {
                name: model_name.clone(),
                passed: false,
                skipped: None,
                error: None,
            };

            let result = test_fn(registry, model_name.clone()).await;

            match result {
                Ok(_) => report.passed = true,
                Err(e) => {
                    if e.to_string() == "SKIP_TEST" {
                        report.skipped = Some(true);
                    } else {
                        report.error = Some(TestError {
                            message: e.to_string(),
                            stack: None, // Stack traces are not standard in Rust errors.
                        });
                    }
                }
            }
            model_reports.push(report);
        }
        reports.push(TestReport {
            description: description.to_string(),
            models: model_reports,
        });
    }

    Ok(reports)
}
