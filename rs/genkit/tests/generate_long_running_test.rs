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

mod helpers;

use genkit::{
    model::{Candidate, FinishReason, Part, Role},
    Genkit, Model,
};
use genkit_ai::{
    self as genkit_ai,
    model::{DefineBackgroundModelOptions, GenerateResponseData},
    GenerateOptions,
};

use genkit_core::background_action::Operation;
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
                    candidates: vec![Candidate {
                        ..Default::default()
                    }],
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

use genkit_core::action::ActionFnArg;
use std::future::Future;
use std::pin::Pin;

#[rstest]
#[tokio::test]
/// 'checks operation status'
async fn test_checks_operation_status(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, ProgrammableModel),
) {
    let (genkit, _pm) = genkit_with_programmable_model.await;

    // 1. Define the expected final result from the background model.
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

    // 2. This is the completed operation that the `check` function will return.
    let final_op = Operation {
        id: "123".to_string(),
        done: true,
        output: Some(result_data.clone()),
        ..Default::default()
    };

    // 3. Define the type for the cancel function pointer to resolve the ambiguity.
    type CancelFut =
        Pin<Box<dyn Future<Output = genkit::Result<Operation<GenerateResponseData>>> + Send>>;
    type CancelFn = fn(Operation<GenerateResponseData>, ActionFnArg<()>) -> CancelFut;

    // 4. Define the background model with its start and check logic.
    genkit.define_background_model(DefineBackgroundModelOptions {
        name: "bkg-model".to_string(),
        versions: None,
        label: None,
        supports: None,
        config_schema: None,
        // The `start` handler returns a pending operation.
        start: |_req, _args| async {
            Ok(Operation {
                id: "123".to_string(),
                done: false,
                ..Default::default()
            })
        },
        // The `check` handler returns the final, completed operation.
        check: move |_op, _args| {
            let final_op = final_op.clone();
            async move { Ok(final_op) }
        },
        // Explicitly type `None` to resolve the ambiguity.
        cancel: None::<CancelFn>,
    });

    // 5. Create an operation object representing the one we want to check.
    //    The `action` key is crucial for looking up the correct check handler.
    let op_to_check = Operation::<serde_json::Value> {
        action: Some("/background-model/bkg-model".to_string()),
        id: "123".to_string(),
        ..Default::default()
    };

    // 6. Call `check_operation`. This will invoke the `check` handler defined above.
    let checked_op = genkit.check_operation(&op_to_check).await.unwrap();

    // 7. Assert that the operation is now done and contains the correct output.
    assert!(checked_op.done);
    let output_data: GenerateResponseData =
        serde_json::from_value(checked_op.output.unwrap()).unwrap();
    assert_eq!(output_data, result_data);
    let text = output_data.candidates.first().unwrap().message.text();
    assert_eq!(text, "done");
}
