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

use async_trait::async_trait;
use futures_util::StreamExt;
use genkit::{
    define_flow,
    error::{Error, Result},
    model::Part,
    plugin::Plugin,
    registry::Registry,
    FinishReason, GenerateRequest, Genkit, GenkitOptions, Model, Role,
};
use genkit_ai::{
    self as genkit_ai, define_model, model::DefineModelOptions, CandidateData, GenerateOptions,
    GenerateResponseChunk, GenerateResponseChunkData, GenerateResponseData, MessageData,
};

use genkit_core::{action::ActionRunOptions, ActionFnArg};
use rstest::{fixture, rstest};
use serde_json::Value;
use std::sync::{Arc, Mutex};

use helpers::StreamingCallback;

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

//
// Streaming Tests
//

struct ErrorModelPlugin;
#[async_trait]
impl Plugin for ErrorModelPlugin {
    fn name(&self) -> &'static str {
        "errorModelPlugin"
    }
    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        define_model(
            registry,
            DefineModelOptions {
                name: "errorModel".to_string(),
                ..Default::default()
            },
            |_, _| async { Err(Error::new_internal("foo")) },
        );
        Ok(())
    }
}

#[rstest]
#[tokio::test]
async fn test_streams_default_model() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(|_, on_chunk| {
            if let Some(cb) = on_chunk.as_ref() {
                for i in 0..3 {
                    cb(GenerateResponseChunkData {
                        index: i,
                        content: vec![Part::text(format!("chunk{}", i + 1))],
                        ..Default::default()
                    });
                }
            }
            Box::pin(async {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        finish_reason: Some(FinishReason::Stop),
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text("chunk1chunk2chunk3")],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));
    }

    let mut response_container: genkit::GenerateStreamResponse =
        genkit.generate_stream(GenerateOptions {
            prompt: Some(vec![Part::text("unused".to_string())]),
            ..Default::default()
        });

    let mut chunks = Vec::new();
    while let Some(chunk_result) = response_container.stream.next().await {
        chunks.push(chunk_result.unwrap());
    }

    assert_eq!(chunks.len(), 3, "Should have received 3 chunks");
    assert_eq!(chunks[0].text(), "3");
    assert_eq!(chunks[1].text(), "2");
    assert_eq!(chunks[2].text(), "1");

    let final_response = response_container
        .response
        .await
        .expect("Tokio task should not panic")
        .expect("Generation process should complete successfully");

    let output_text = final_response
        .text()
        .expect("Final response should contain text");

    assert!(
        output_text.starts_with("Echo: unused; config: null"),
        "Final response text did not match expected output"
    );
}

#[rstest]
#[tokio::test]
async fn test_streaming_rethrows_response_errors() {
    let error_plugin = Arc::new(ErrorModelPlugin);
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![error_plugin as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();

    let model = Some(Model::Name("blockingModel".to_string()));
    let stream_response: genkit::GenerateStreamResponse = genkit.generate_stream(GenerateOptions {
        model,
        prompt: Some(vec![Part::text("short and sweet".to_string())]),
        ..Default::default()
    });

    assert!(stream_response.response.await.unwrap().is_err());
}

#[rstest]
#[tokio::test]
async fn test_generate_stream_rethrows_initialization_errors() {
    let (genkit, _) = helpers::genkit_instance_for_test().await;

    // Call generate_stream with a model that doesn't exist.
    let mut resp: genkit::GenerateStreamResponse = genkit.generate_stream(GenerateOptions {
        model: Some(Model::Name("modelNotFound".to_string())),
        prompt: Some(vec![Part::text("hi")]),
        ..Default::default()
    });

    // Try to get the first item from the stream.
    // This is where the initialization error should now appear.
    let first_item = resp.stream.next().await;

    assert!(
        first_item.is_some(),
        "Stream should have emitted an error item"
    );

    let result = first_item.unwrap();
    assert!(result.is_err(), "Stream item should be an Err");

    // Check that the error is the one we expect.
    let err = result.unwrap_err();
    let err_string = err.to_string();
    assert!(
        err_string.contains("not found"),
        "Error message should indicate that the model was not found"
    );
}

#[rstest]
#[tokio::test]
async fn test_flow_passes_streaming_callback_to_generate() {
    // 1. Setup the environment with a programmable model
    let (genkit, pm_handle) = helpers::genkit_with_programmable_model().await;
    let mut registry = genkit.registry().clone();

    // 2. Create a flag to track if the model received the streaming signal
    let streaming_was_received = Arc::new(Mutex::new(false));
    let streaming_was_received_clone = streaming_was_received.clone();

    // 3. Configure the mock model to set the flag if it receives a callback
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(
            move |_, streaming_callback: Option<helpers::StreamingCallback>| {
                if streaming_callback.is_some() {
                    *streaming_was_received_clone.lock().unwrap() = true;
                }
                Box::pin(async { Ok(Default::default()) })
            },
        ));
    }

    // 4. Define the flow that calls `generate`
    let wrapper_flow = define_flow(
        &mut registry,
        "wrapper",
        // The flow's chunk type is `()`, defined by `ActionFnArg<()>`
        move |_: (), _args: ActionFnArg<()>| {
            let genkit_clone = genkit.clone();
            async move {
                // Inside the flow, call `generate` with its own `on_chunk` callback.
                let response = genkit_clone
                    .generate_with_options(GenerateOptions {
                        model: Some(Model::Name("programmableModel".to_string())),
                        prompt: Some(vec![Part::text("hi")]),
                        on_chunk: Some(Arc::new(|_chunk: GenerateResponseChunk<Value>| Ok(()))),
                        ..Default::default()
                    })
                    .await?;

                Ok(response.text().unwrap_or_default())
            }
        },
    );

    // 5. Run the flow itself in streaming mode.
    let run_options = ActionRunOptions {
        // This `on_chunk` must match the flow's chunk type, which is `()`.
        on_chunk: Some(Arc::new(|_chunk: Result<(), Error>| {})),
        ..Default::default()
    };

    // We don't need the flow's final output, just its side effect on the model.
    let _ = wrapper_flow.run((), Some(run_options)).await;

    // 6. Assert that the model handler received the streaming callback.
    assert!(
        *streaming_was_received.lock().unwrap(),
        "The model action was not notified of the streaming request."
    );
}

#[tokio::test]
async fn test_flow_propagates_streaming_to_generate() {
    // 1. Setup the test environment with the correct fixture
    let (genkit, pm_handle) = helpers::genkit_with_programmable_model().await;
    let mut registry = genkit.registry().clone();

    // 2. Create a flag to track if the model received the streaming signal
    let streaming_was_requested = Arc::new(Mutex::new(false));
    let streaming_was_requested_clone = streaming_was_requested.clone();

    // 3. Configure the programmable model's handler
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        // The handler's signature now uses the correct type alias from your helper file
        *handler = Arc::new(Box::new(
            move |_, streaming_callback: Option<helpers::StreamingCallback>| {
                let was_called_clone_2 = streaming_was_requested_clone.clone();
                Box::pin(async move {
                    if streaming_callback.is_some() {
                        *was_called_clone_2.lock().unwrap() = true;
                    }
                    Ok(Default::default())
                })
            },
        ));
    }

    // 4. Define the flow
    let wrapper_flow = define_flow(
        &mut registry,
        "wrapper",
        move |_: (), _args: ActionFnArg<()>| {
            let genkit_clone = genkit.clone();
            async move {
                let response = genkit_clone
                    .generate_with_options(GenerateOptions {
                        model: Some(Model::Name("programmableModel".to_string())),
                        prompt: Some(vec![Part::text("hi")]),
                        on_chunk: Some(Arc::new(|_chunk: GenerateResponseChunk<Value>| Ok(()))),
                        ..Default::default()
                    })
                    .await?;

                Ok(response.text().unwrap_or_default())
            }
        },
    );

    // 5. Run the flow in a streaming context
    let run_options = ActionRunOptions {
        on_chunk: Some(Arc::new(|_chunk: Result<(), Error>| {})),
        ..Default::default()
    };

    let _ = wrapper_flow.run((), Some(run_options)).await;

    // 6. Assert that the model was correctly notified
    assert!(
        *streaming_was_requested.lock().unwrap(),
        "The model action was not notified of the streaming request."
    );
}

#[rstest]
#[tokio::test]
async fn test_streaming_strips_out_noop_streaming_callback(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, helpers::ProgrammableModel),
) {
    let (genkit, pm_handle) = genkit_with_programmable_model.await;

    let was_called = Arc::new(Mutex::new(false));
    let was_called_clone = was_called.clone();

    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(
            move |_, streaming_callback: Option<StreamingCallback>| {
                let was_called_clone_2 = was_called_clone.clone();
                Box::pin(async move {
                    if streaming_callback.is_some() {
                        *was_called_clone_2.lock().unwrap() = true;
                    }
                    Ok(Default::default())
                })
            },
        ));
    }

    let _: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(*was_called.lock().unwrap());
}
