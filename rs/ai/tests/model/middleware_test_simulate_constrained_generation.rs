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

//! # Model Middleware Tests

use genkit_ai::model::middleware::simulate_constrained_generation;
use genkit_ai::model::{GenerateRequest, GenerateResponseData};
use genkit_core::error::Result;
use rstest::rstest;
use serde_json::{from_value, json, Value};
use std::sync::{Arc, Mutex};

#[path = "../helpers.rs"]
mod helpers;

#[cfg(test)]
mod test {
    use super::*;

    // Helper to invoke the middleware and capture the modified request.
    async fn test_constrained_request(req: GenerateRequest) -> GenerateRequest {
        let captured_req = Arc::new(Mutex::new(None));
        let captured_req_clone = captured_req.clone();

        let next = move |req: GenerateRequest| -> genkit_ai::model::BoxFuture<'static, Result<GenerateResponseData>> {
            let mut guard = captured_req_clone.lock().unwrap();
            *guard = Some(req);
            Box::pin(async { Ok(Default::default()) })
        };

        let middleware = simulate_constrained_generation(None);
        middleware(req, Box::new(next)).await.unwrap();

        let took_captured_req = captured_req.lock().unwrap().take().unwrap();
        took_captured_req
    }

    #[rstest]
    #[tokio::test]
    async fn test_simulates_constrained_generation() {
        let output_json = json!({
            "constrained": true,
            "schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                }
            }
        });
        let req: GenerateRequest = from_value(json!({
            "messages": [{ "role": "user", "content": [{ "text": "hello" }] }],
            "output": output_json.to_string(),
        }))
        .unwrap();

        let _modified_req = test_constrained_request(req).await;
        // Just running this should trigger the log output.
    }

    #[rstest]
    #[tokio::test]
    async fn test_injects_instructions_into_request() {
        use self::helpers::{registry_with_programmable_model, ProgrammableModelHandler};
        use genkit_ai::document::Part;
        use genkit_ai::generate::{generate, GenerateOptions, OutputOptions};
        use genkit_ai::message::{MessageData, Role};
        use genkit_ai::model::{CandidateData, Model};

        let (registry, pm_handle) = registry_with_programmable_model().await;

        let handler: ProgrammableModelHandler =
            Arc::new(Box::new(move |_req, _streaming_callback| {
                Box::pin(async {
                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
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

        *pm_handle.handler.lock().unwrap() = handler;

        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Foo {
            foo: String,
        }

        let schema_value = json!({
            "type": "object",
            "properties": {
                "foo": {
                    "type": "string"
                }
            },
            "required": ["foo"],
            "additionalProperties": true,
            "$schema": "http://json-schema.org/draft-07/schema#"
        });

        let result = generate(
            &registry,
            GenerateOptions::<Value> {
                model: Some(Model::Name("programmableModel".to_string())),
                prompt: Some(vec![Part::text("generate json")]),
                output: Some(OutputOptions {
                    schema: Some(schema_value.clone()),
                    format: Some("json".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let output_val: Foo = serde_json::from_value(result.output().unwrap()).unwrap();
        assert_eq!(
            output_val,
            Foo {
                foo: "bar".to_string()
            }
        );

        let last_req = pm_handle.last_request.lock().unwrap().clone().unwrap();

        let expected_text = format!(
            "Output should be in JSON format and conform to the following schema:\n\n```\n{}\n```\n",
            serde_json::to_string(&schema_value).unwrap()
        );

        let mut expected_req_val = json!({
          "messages": [
            {
              "role": "user",
              "content": [
                { "text": "generate json" },
                {
                  "metadata": { "purpose": "output" },
                  "text": expected_text
                }
              ]
            }
          ],
          "output": {
            "constrained": false
          },
          "tools": []
        });

        let mut last_req_val = serde_json::to_value(&last_req).unwrap();
        // Remove config from comparison as it's not in the expected output
        last_req_val.as_object_mut().unwrap().remove("config");

        // To avoid flaky tests due to pretty-printing differences,
        // we extract the schema JSON from the text and compare it as a Value.
        let last_req_content = last_req_val["messages"][0]["content"][1]["text"]
            .as_str()
            .unwrap();
        let expected_req_content = expected_req_val["messages"][0]["content"][1]["text"]
            .as_str()
            .unwrap();

        let last_schema_str = last_req_content.split("```").nth(1).unwrap().trim();
        let expected_schema_str = expected_req_content.split("```").nth(1).unwrap().trim();

        let last_schema: Value = serde_json::from_str(last_schema_str).unwrap();
        let expected_schema: Value = serde_json::from_str(expected_schema_str).unwrap();

        assert_eq!(last_schema, expected_schema);

        // Now that we've compared the schema text, replace it with a placeholder
        // to compare the rest of the request structure.
        let placeholder = Value::String("SCHEMA_TEXT_PLACEHOLDER".to_string());
        last_req_val["messages"][0]["content"][1]["text"] = placeholder.clone();
        expected_req_val["messages"][0]["content"][1]["text"] = placeholder;

        assert_eq!(last_req_val, expected_req_val);
    }
}
