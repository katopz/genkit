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

use futures_util::StreamExt;
use genkit::{genkit, Genkit};
use genkit_core::error::Result;
use serde_json::{json, Value};
use std::collections::HashMap;

fn setup_with_context() -> Genkit {
    let mut config = Config::default();
    let mut context = HashMap::new();
    context.insert("something".to_string(), json!("extra"));
    config.flow_context = Some(context);
    genkit(config)
}

#[tokio::test]
async fn calls_simple_flow() -> Result<()> {
    let mut ai = genkit(Default::default());

    ai.define_flow("banana", move |_: (), _ctx: FlowContext| async move {
        Ok(json!("banana"))
    });

    let result: String = ai.run("banana", ()).await?;
    assert_eq!(result, "banana");
    Ok(())
}

#[tokio::test]
async fn streams_simple_chunks_with_schema_defined() -> Result<()> {
    let mut ai = genkit(Default::default());

    ai.define_flow(
        "streaming_banana",
        move |input: String, ctx: FlowContext| async move {
            if let Some(sender) = ctx.stream() {
                for char in input.chars() {
                    sender.send(json!(char.to_string())).await.unwrap();
                }
            }
            Ok(json!(input))
        },
    );

    // Test streaming call
    let (mut stream, output_handle) = ai
        .run_stream::<_, String, String>("streaming_banana", "banana".to_string())
        .await?;

    let mut chunks = Vec::new();
    while let Some(chunk_result) = stream.next().await {
        chunks.push(chunk_result?);
    }
    let output: String = serde_json::from_value(output_handle.await??)?;

    assert_eq!(output, "banana");
    let expected_chunks: Vec<String> = "banana".chars().map(|c| c.to_string()).collect();
    assert_eq!(chunks, expected_chunks);

    // a "streaming" flow can be invoked in non-streaming mode.
    let non_streaming_result: String = ai.run("streaming_banana", "banana2".to_string()).await?;
    assert_eq!(non_streaming_result, "banana2");

    Ok(())
}

#[tokio::test]
async fn passes_through_the_context() -> Result<()> {
    // NOTE: This test is slightly different from the TS version.
    // The current Rust `ai.run` API doesn't support passing per-call context.
    // This test only verifies that the globally configured context is available.
    let mut ai = setup_with_context();

    ai.define_flow("context_flow", move |_: (), ctx: FlowContext| async move {
        // The context in FlowContext is the merged context.
        Ok(serde_json::to_value(ctx.context()).unwrap())
    });

    // In the TS test, a call-site context is also passed: `{ context: { foo: 'bar' } }`
    // which gets merged with the global context. We can't do that here yet.
    let response: Value = ai.run("context_flow", ()).await?;

    let mut expected_context = HashMap::new();
    expected_context.insert("something".to_string(), json!("extra"));

    assert_eq!(response, serde_json::to_value(expected_context)?);

    Ok(())
}
