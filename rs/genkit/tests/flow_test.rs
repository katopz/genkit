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

use genkit::{define_flow, error::Result, flow::Flow, ActionContext, Genkit, GenkitOptions};
use genkit_core::action::ActionRunOptions;
use rstest::*;
use serde_json::json;
use std::sync::{Arc, Mutex};

// This fixture function acts as the "beforeEach" block.
#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    let default_context = ActionContext {
        auth: None,
        additional_context: [("something".to_string(), json!("extra"))]
            .iter()
            .cloned()
            .collect(),
        ..Default::default()
    };

    Genkit::init(GenkitOptions {
        context: Some(default_context),
        ..Default::default()
    })
    .await
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn test_simple_flow(#[future] genkit_instance: Arc<Genkit>) -> Result<()> {
    let genkit = genkit_instance.await;
    let registry = genkit.registry().clone();

    let banana_flow: Flow<(), &str, ()> =
        define_flow(&registry, "banana", |_: (), _| async { Ok("banana") });

    let result = banana_flow.run((), None).await?;
    assert_eq!(result.result, "banana");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_streaming_flow(#[future] genkit_instance: Arc<Genkit>) -> Result<()> {
    let genkit = genkit_instance.await;
    let registry = genkit.registry().clone();

    let streaming_banana_flow: Flow<String, String, char> =
        define_flow(&registry, "banana", |input: String, args| async move {
            for char in input.chars() {
                if args.chunk_sender.send(char).is_err() {
                    // Receiver is gone, probably the client disconnected.
                    // Stop sending chunks.
                    break;
                }
            }
            Ok(input)
        });

    // Test streaming call
    let mut stream_resp = streaming_banana_flow.stream("banana".to_string(), None);
    let chunks = Arc::new(Mutex::new(Vec::new()));

    let chunks_clone = chunks.clone();
    let stream_consumer = async move {
        while let Some(chunk_result) = futures_util::StreamExt::next(&mut stream_resp.stream).await
        {
            chunks_clone.lock().unwrap().push(chunk_result.unwrap());
        }
        Ok(()) as Result<()>
    };

    let (output_res, _) = tokio::try_join!(stream_resp.output, stream_consumer)?;
    let output = output_res;

    assert_eq!(output, "banana");
    assert_eq!(*chunks.lock().unwrap(), vec!['b', 'a', 'n', 'a', 'n', 'a']);

    // Test non-streaming call to a streaming flow
    let non_streaming_result = streaming_banana_flow
        .run("banana2".to_string(), None)
        .await?;
    assert_eq!(non_streaming_result.result, "banana2");

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_context_passing(#[future] genkit_instance: Arc<Genkit>) -> Result<()> {
    let genkit = genkit_instance.await;
    let registry = genkit.registry().clone();

    let context_flow: Flow<(), String, ()> =
        define_flow(&registry, "contextFlow", |_, args| async {
            let context = args.context.unwrap_or_default();
            serde_json::to_string(&context.additional_context)
                .map_err(|e| genkit::Error::new_internal(e.to_string()))
        });

    // Merge default context with call-site context
    let mut final_context = genkit.context().cloned().unwrap_or_default();
    final_context
        .additional_context
        .insert("foo".to_string(), json!("bar"));

    let result = context_flow
        .run(
            (),
            Some(ActionRunOptions {
                context: Some(final_context),
                ..Default::default()
            }),
        )
        .await?;

    // Use json! to ensure consistent serialization (e.g., key order)
    let expected_json = json!({
        "something": "extra",
        "foo": "bar"
    });
    let actual_json: serde_json::Value = serde_json::from_str(&result.result)
        .map_err(|e| genkit::Error::new_internal(e.to_string()))?;
    assert_eq!(actual_json, expected_json);

    Ok(())
}
