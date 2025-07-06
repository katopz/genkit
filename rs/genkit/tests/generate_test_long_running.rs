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
    model::{Candidate, FinishReason, Part, Role},
    registry::ActionType,
    Genkit, Model,
};
use genkit_ai::{self as genkit_ai, model::GenerateResponseData, GenerateOptions};

use genkit_core::{
    action::{define_action, ActionFnArg},
    background_action::Operation,
};
use rstest::{fixture, rstest};
use serde_json::json;
use std::sync::Arc;

use genkit_ai::message::MessageData;
use helpers::ProgrammableModel;

use genkit::ToolConfig;
use schemars::JsonSchema;

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
struct TestToolInput {}

// The tests in this file are based on the concept of "background models"
// that can handle long-running generation tasks. This functionality is
// not fully implemented in the Rust version yet, so these tests are
// currently ignored and act as placeholders for future development.
// They require:
// 1. A way to define models with `start` and `check` logic for operations.
// 2. A `genkit.check_operation()` function.
// 3. The `GenerateResponse` struct to include an `operation` field.

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

#[rstest]
#[tokio::test]
async fn test_starts_the_operation(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, ProgrammableModel),
) {
    let (genkit, pm) = genkit_with_programmable_model.await;

    // Define a tool that stringifies its context.
    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(TestToolInput {}),
            output_schema: Some(json!("")), // Output is a string
            metadata: None,
        },
        // The tool handler now uses the context.
        |_, ctx| async move {
            // Due to `serde(flatten)`, serializing the whole context object
            // will produce the desired JSON string.
            Ok(json!(serde_json::to_string(&ctx.context).unwrap()))
        },
    );

    // Configure the programmable model to simulate the `start` of a long-running operation.
    // It returns a `GenerateResponseData` containing an `Operation`'s ID.
    {
        let mut handler = pm.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_req, _cb| {
            Box::pin(async {
                Ok(GenerateResponseData {
                    operation: Some("123".to_string()),
                    ..Default::default()
                })
            })
        }));
    }

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec!["testTool".into()]),
            ..Default::default()
        })
        .await
        .unwrap();

    let op = response.operation.unwrap();
    assert_eq!(op.id, "123");
    assert!(!op.done);
}

#[rstest]
#[tokio::test]
async fn test_checks_operation_status(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, ProgrammableModel),
) {
    let (genkit, _pm) = genkit_with_programmable_model.await;
    let mut registry = genkit.registry().clone();
    // This test requires a `genkit.check_operation()` function and a background
    // model implementation that can be polled.

    // 1. Define a background model's check action that returns a completed operation.
    let result_data = GenerateResponseData {
        candidates: vec![Candidate {
            index: 0,
            finish_reason: Some(FinishReason::Stop),
            message: MessageData {
                role: Role::Model,
                content: vec![Part::text("done".to_string())],
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    let completed_op_result = result_data;
    define_action(
        &mut registry,
        ActionType::CheckOperation,
        "/checkoperation/bkg-model/check",
        move |_op: genkit_core::background_action::Operation<serde_json::Value>,
              _ctx: ActionFnArg<()>| {
            let completed_op_result = completed_op_result.clone();
            async move {
                let mut op = Operation {
                    done: true,
                    ..Default::default()
                };
                op.output = Some(serde_json::to_value(completed_op_result).unwrap());
                Ok(op)
            }
        },
    );

    // 3. Call `check_operation`.
    let op_to_check = Operation::<serde_json::Value> {
        action: Some("/background-model/bkg-model".to_string()),
        id: "123".to_string(),
        ..Default::default()
    };
    let checked_op = genkit.check_operation(&op_to_check).await.unwrap();

    // 4. Assert the result.
    assert!(checked_op.done);
    let result: GenerateResponseData = serde_json::from_value(checked_op.output.unwrap()).unwrap();
    let text = result.candidates.first().unwrap().message.text();
    assert_eq!(text, "done");
}
