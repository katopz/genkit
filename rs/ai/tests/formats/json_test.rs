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

use genkit_ai::{
    document::Part,
    formats::json::json_formatter,
    generate::{
        chunk::{GenerateResponseChunk, GenerateResponseChunkOptions},
        generate_stream, GenerateOptions,
    },
    message::{Message, MessageData, Role},
    model::GenerateResponseChunkData,
};
use rstest::*;
use serde::Deserialize;
use serde_json::{from_value, json, Value};

// Import test helpers
#[path = "../helpers.rs"]
mod helpers;

#[cfg(test)]
mod json_format_tests {
    use super::*;

    //-////////////////////////////////////////////////////////////////////////-
    // Streaming Parser Tests
    //-////////////////////////////////////////////////////////////////////////-

    #[derive(Debug, Deserialize)]
    struct StreamingTestCase {
        desc: String,
        chunks: Vec<StreamingTestChunk>,
    }

    #[derive(Debug, Deserialize)]
    struct StreamingTestChunk {
        text: String,
        want: Option<Value>,
    }

    #[rstest]
    fn test_streaming_parser() {
        let test_cases: Vec<StreamingTestCase> = from_value(json!([
            {
                "desc": "parses complete JSON object",
                "chunks": [
                    {
                        "text": "{\"id\": 1, \"name\": \"test\"}",
                        "want": { "id": 1, "name": "test" }
                    }
                ]
            },
            {
                "desc": "handles partial JSON",
                "chunks": [
                    {
                        "text": "{\"id\": 1",
                        "want": null // Rust's extract_json only parses complete JSON.
                    },
                    {
                        "text": ", \"name\": \"test\"}",
                        "want": { "id": 1, "name": "test" }
                    }
                ]
            },
            {
                "desc": "handles preamble with code fence",
                "chunks": [
                    {
                        "text": "Here is the JSON:\n\n```json\n",
                        "want": null
                    },
                    {
                        "text": "{\"id\": 1}\n```",
                        "want": { "id": 1 }
                    }
                ]
            }
        ]))
        .unwrap();

        for case in test_cases {
            let formatter = json_formatter();
            let handler = (formatter.handler)(None);
            let mut chunks: Vec<GenerateResponseChunkData> = Vec::new();

            for (i, chunk_def) in case.chunks.iter().enumerate() {
                let new_chunk_data = GenerateResponseChunkData {
                    index: 0,
                    role: Some(Role::Model),
                    content: vec![Part::text(chunk_def.text.as_str())],
                    ..Default::default()
                };

                let response_chunk = GenerateResponseChunk::new(
                    new_chunk_data.clone(),
                    GenerateResponseChunkOptions {
                        previous_chunks: chunks.clone(),
                        ..Default::default()
                    },
                );
                chunks.push(new_chunk_data);

                let result = handler.parse_chunk(&response_chunk);
                assert_eq!(
                    result, chunk_def.want,
                    "Test: '{}', Chunk: {}",
                    case.desc, i
                );
            }
        }
    }

    //-////////////////////////////////////////////////////////////////////////-
    // Message Parser Tests
    //-////////////////////////////////////////////////////////////////////////-

    #[derive(Debug, Deserialize)]
    struct MessageTestCase {
        desc: String,
        message: MessageData,
        want: Option<Value>,
    }

    #[rstest]
    fn test_message_parser() {
        let test_cases: Vec<MessageTestCase> = from_value(json!([
            {
                "desc": "parses complete JSON response",
                "message": { "role": "model", "content": [{ "text": "{\"id\": 1, \"name\": \"test\"}" }] },
                "want": { "id": 1, "name": "test" }
            },
            {
                "desc": "handles empty response",
                "message": { "role": "model", "content": [{ "text": "" }] },
                "want": null
            },
            {
                "desc": "parses JSON with preamble and code fence",
                "message": { "role": "model", "content": [{ "text": "Here is the JSON:\n\n```json\n{\"id\": 1}\n```" }] },
                "want": { "id": 1 }
            }
        ]))
        .unwrap();

        for case in test_cases {
            let formatter = json_formatter();
            let handler = (formatter.handler)(None);
            let message = Message::new(case.message.clone(), None);
            let result = handler.parse_message(&message);
            let want = case.want.clone().unwrap_or(Value::Null);
            assert_eq!(result, want, "Test: '{}'", case.desc);
        }
    }
}

#[cfg(test)]
mod json_format_e2e_tests {
    use super::*;
    use futures_util::StreamExt;
    use genkit_ai::{
        model::{CandidateData, FinishReason, GenerateResponseData},
        Part, Role,
    };
    use schemars::JsonSchema;
    use std::sync::Arc;

    #[derive(JsonSchema, Deserialize, PartialEq, Debug)]
    struct TestSchema {
        foo: String,
    }

    #[rstest]
    #[tokio::test]
    async fn test_e2e_injects_instructions_into_request() {
        let (registry, pm_handle) = helpers::registry_with_programmable_model().await;

        let pm_handler: helpers::ProgrammableModelHandler =
            Arc::new(Box::new(|_req, streaming_callback| {
                Box::pin(async move {
                    if let Some(cb) = streaming_callback {
                        // Simulate streaming a JSON object in parts
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                        cb(GenerateResponseChunkData {
                            index: 0,
                            content: vec![Part::text("```\n{")],
                            ..Default::default()
                        });
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                        cb(GenerateResponseChunkData {
                            index: 0,
                            content: vec![Part::text("\"foo\": \"b")],
                            ..Default::default()
                        });
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                        cb(GenerateResponseChunkData {
                            index: 0,
                            content: vec![Part::text("ar\"")],
                            ..Default::default()
                        });
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                        cb(GenerateResponseChunkData {
                            index: 0,
                            content: vec![Part::text("}\n```")],
                            ..Default::default()
                        });
                    }

                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            index: 0,
                            finish_reason: Some(FinishReason::Stop),
                            message: MessageData {
                                role: Role::Model,
                                content: vec![Part::text("```\n{\"foo\": \"bar\"}\n```")],
                                ..Default::default()
                            },
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                })
            }));

        *pm_handle.handler.lock().unwrap() = pm_handler;

        let schema = schemars::schema_for!(TestSchema);
        let schema_value = serde_json::to_value(&schema).unwrap();

        let generate_options = from_value::<GenerateOptions<Value>>(json!({
            "model": "programmableModel",
            "prompt": [{ "text": "generate json" }],
            "output": {
                "format": "json",
                "schema": schema_value,
            }
        }))
        .unwrap();

        let mut stream = generate_stream(&registry, generate_options).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk_result) = stream.stream.next().await {
            let chunk = chunk_result.unwrap();
            // The default `output()` on chunk will try to parse using extract_json,
            // which is what we want to test.
            if let Ok(output) = chunk.output() {
                chunks.push(output);
            }
        }

        let final_response = stream.response.await.unwrap().unwrap();

        assert_eq!(
            final_response.output().unwrap(),
            json!({"foo": "bar"}),
            "Final response is incorrect"
        );

        // The Rust `extract_json` function (using json5) is not as lenient as the JS `partial-json`
        // library. It only emits a value when a complete, valid JSON object has been accumulated.
        // Therefore, we expect to see `None` for all partial chunks and then the final object
        // once the closing brace is received. The `generate` stream filters out failed `output()` calls.
        assert_eq!(
            chunks,
            vec![json!({"foo": "bar"})],
            "Streamed chunks do not match expectation"
        );
    }
}
