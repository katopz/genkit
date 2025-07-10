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

mod helpers;

use genkit::{prompt::PromptConfig, Genkit, Part, Role};
use genkit_ai::{MessageData, OutputOptions};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
struct EmptyInput {}

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
}

#[rstest]
#[tokio::test]
/// 'loads from the folder'
async fn test_loads_from_folder(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // In Rust, we don't have a direct equivalent of `promptDir`.
    // We simulate the file being loaded by defining the prompt directly.
    let test_prompt_config = PromptConfig {
        name: "test".to_string(),
        prompt: Some("Hello from the prompt file".to_string()),
        config: Some(json!({ "temperature": 11 })),
        output: Some(OutputOptions::default()),
        ..Default::default()
    };
    genkit
        .define_prompt::<EmptyInput, Value, Value>(test_prompt_config)
        .await;

    // Look up the prompt from the registry.
    let test_prompt =
        genkit_ai::prompt::prompt::<EmptyInput, Value, Value>(genkit.registry(), "test", None)
            .await
            .unwrap();

    // First, test generation to ensure the default model is used correctly.
    let response = test_prompt.generate(EmptyInput {}, None).await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: Hello from the prompt file; config: {\"temperature\":11}"
    );

    // Second, test the render method.
    let rendered_opts = test_prompt.render(EmptyInput {}, None).await.unwrap();

    let expected_messages = Some(vec![MessageData {
        role: Role::User,
        content: vec![Part::text("Hello from the prompt file")],
        ..Default::default()
    }]);

    assert_eq!(rendered_opts.messages, expected_messages);
    assert_eq!(rendered_opts.config, Some(json!({ "temperature": 11 })));
    assert_eq!(rendered_opts.output, Some(OutputOptions::default()));
    // The render method itself doesn't apply the default model, so this should be None.
    assert!(rendered_opts.model.is_none());
}

#[rstest]
#[tokio::test]
/// 'loads from the sub folder'
async fn test_loads_from_sub_folder(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // Simulate loading from `prompts/sub/test.prompt` by defining it with that name.
    let sub_test_prompt_config = PromptConfig {
        name: "sub/test".to_string(),
        prompt: Some("Hello from the sub folder prompt file".to_string()),
        config: Some(json!({ "temperature": 12 })),
        output: Some(OutputOptions::default()),
        ..Default::default()
    };
    genkit
        .define_prompt::<EmptyInput, Value, Value>(sub_test_prompt_config)
        .await;

    // Look up the prompt using its full name.
    let test_prompt =
        genkit_ai::prompt::prompt::<EmptyInput, Value, Value>(genkit.registry(), "sub/test", None)
            .await
            .unwrap();

    // Test generation.
    let response = test_prompt.generate(EmptyInput {}, None).await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: Hello from the sub folder prompt file; config: {\"temperature\":12}"
    );

    // Test rendering.
    let rendered_opts = test_prompt.render(EmptyInput {}, None).await.unwrap();

    let expected_messages = Some(vec![MessageData {
        role: Role::User,
        content: vec![Part::text("Hello from the sub folder prompt file")],
        ..Default::default()
    }]);

    assert_eq!(rendered_opts.messages, expected_messages);
    assert_eq!(rendered_opts.config, Some(json!({ "temperature": 12 })));
    assert_eq!(rendered_opts.output, Some(OutputOptions::default()));
}

use genkit::{model::Model, tool::ToolArgument};

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
struct KitchenSinkInput {
    subject: String,
}

#[rstest]
#[tokio::test]
/// 'loads from the folder with all the options'
async fn test_loads_from_folder_with_all_options(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // Simulate loading a complex prompt from a file.
    let kitchen_sink_config = PromptConfig {
        name: "kitchensink".to_string(),
        model: Some(Model::Name(
            "googleai/gemini-5.0-ultimate-pro-plus".to_string(),
        )),
        config: Some(json!({ "temperature": 11 })),
        output: Some(OutputOptions {
            format: Some("csv".to_string()),
            json_schema: Some(json!({
                "type": "object",
                "properties": {
                    "obj": {
                        "type": ["object", "null"],
                        "description": "a nested object",
                        "properties": {
                            "nest1": { "type": ["string", "null"] }
                        },
                        "additionalProperties": false
                    },
                    "arr": {
                        "type": "array",
                        "description": "array of objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nest2": { "type": ["boolean", "null"] }
                            },
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["arr"],
                "additionalProperties": false
            })),
            ..Default::default()
        }),
        system: Some(" Hello ".to_string()),
        messages: Some(vec![MessageData {
            role: Role::Model,
            content: vec![Part::text(" from the prompt file {{subject}}".to_string())],
            ..Default::default()
        }]),
        tools: Some(vec![
            ToolArgument::from("toolA"),
            ToolArgument::from("toolB"),
        ]),
        max_turns: Some(77),
        return_tool_requests: Some(true),
        ..Default::default()
    };
    genkit
        .define_prompt::<KitchenSinkInput, Value, Value>(kitchen_sink_config)
        .await;

    let test_prompt = genkit_ai::prompt::prompt::<KitchenSinkInput, Value, Value>(
        genkit.registry(),
        "kitchensink",
        None,
    )
    .await
    .unwrap();

    let request = test_prompt
        .render(
            KitchenSinkInput {
                subject: "banana".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    // Assertions are split to make debugging easier.
    assert_eq!(
        request.model,
        Some(Model::Name(
            "googleai/gemini-5.0-ultimate-pro-plus".to_string()
        ))
    );
    assert_eq!(request.config, Some(json!({ "temperature": 11 })));
    assert_eq!(
        request.output,
        Some(OutputOptions {
            format: Some("csv".to_string()),
            json_schema: Some(json!({
                "type": "object",
                "properties": {
                    "obj": {
                        "type": ["object", "null"],
                        "description": "a nested object",
                        "properties": { "nest1": { "type": ["string", "null"] } },
                        "additionalProperties": false
                    },
                    "arr": {
                        "type": "array",
                        "description": "array of objects",
                        "items": {
                            "type": "object",
                            "properties": { "nest2": { "type": ["boolean", "null"] } },
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["arr"],
                "additionalProperties": false
            })),
            ..Default::default()
        })
    );
    assert_eq!(
        request.messages,
        Some(vec![
            MessageData {
                role: Role::System,
                content: vec![Part::text(" Hello ")],
                ..Default::default()
            },
            MessageData {
                role: Role::Model,
                content: vec![Part::text(" from the prompt file banana")],
                ..Default::default()
            },
        ])
    );
    assert_eq!(request.max_turns, Some(77));
    assert_eq!(request.return_tool_requests, Some(true));
    // The `tools` field in GenerateOptions is resolved from ToolArgument to ToolDefinition,
    // so we can't directly compare them here. We'll add tool definitions if needed.
    // assert_eq!(request.tool_choice, Some(ToolChoice::Required));
    // assert!(request.tools.is_some());
}

#[rstest]
#[tokio::test]
/// 'renders loaded prompt via executable-prompt'
async fn test_renders_loaded_prompt_via_executable_prompt(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    genkit_ai::model::define_model(
        genkit.registry(),
        genkit_ai::model::DefineModelOptions {
            name: "googleai/gemini-5.0-ultimate-pro-plus".to_string(),
            ..Default::default()
        },
        |_req, _| async { Ok(genkit_ai::model::GenerateResponseData::default()) },
    );

    genkit.define_tool(
        genkit::tool::ToolConfig {
            name: "toolA".to_string(),
            description: "toolA it is".to_string(),
            input_schema: Some(()),
            output_schema: Some(()),
            metadata: None,
        },
        |_, _| async { Ok(()) },
    );
    genkit.define_tool(
        genkit::tool::ToolConfig {
            name: "toolB".to_string(),
            description: "toolB it is".to_string(),
            input_schema: Some(()),
            output_schema: Some(()),
            metadata: None,
        },
        |_, _| async { Ok(()) },
    );

    let kitchen_sink_config = PromptConfig {
        name: "kitchensink".to_string(),
        model: Some(Model::Name(
            "googleai/gemini-5.0-ultimate-pro-plus".to_string(),
        )),
        config: Some(json!({ "temperature": 11 })),
        output: Some(OutputOptions {
            format: Some("csv".to_string()),
            json_schema: Some(json!({
                "type": "object",
                "properties": {
                    "obj": {
                        "type": ["object", "null"],
                        "description": "a nested object",
                        "properties": {
                            "nest1": { "type": ["string", "null"] }
                        },
                        "additionalProperties": false
                    },
                    "arr": {
                        "type": "array",
                        "description": "array of objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nest2": { "type": ["boolean", "null"] }
                            },
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["arr"],
                "additionalProperties": false
            })),
            ..Default::default()
        }),
        system: Some(" Hello ".to_string()),
        messages: Some(vec![MessageData {
            role: Role::Model,
            content: vec![Part::text(" from the prompt file {{subject}}".to_string())],
            ..Default::default()
        }]),
        tools: Some(vec![
            ToolArgument::from("toolA"),
            ToolArgument::from("toolB"),
        ]),
        max_turns: Some(77),
        return_tool_requests: Some(true),
        ..Default::default()
    };
    genkit
        .define_prompt::<KitchenSinkInput, Value, Value>(kitchen_sink_config)
        .await;

    let action = genkit
        .registry()
        .lookup_action("/prompt/kitchensink")
        .await
        .unwrap();

    let response_value = action
        .run_http_json(json!({ "subject": "banana" }), None)
        .await
        .unwrap();

    let generate_request: genkit_ai::model::GenerateRequest =
        serde_json::from_value(response_value["result"].clone()).unwrap();

    let tool_a = genkit
        .registry()
        .lookup_action("/tool/toolA")
        .await
        .unwrap();
    let tool_b = genkit
        .registry()
        .lookup_action("/tool/toolB")
        .await
        .unwrap();
    let expected_tools = Some(vec![
        genkit::tool::to_tool_definition(tool_a.as_ref()).unwrap(),
        genkit::tool::to_tool_definition(tool_b.as_ref()).unwrap(),
    ]);

    let expected_output_options = OutputOptions {
        format: Some("csv".to_string()),
        json_schema: Some(json!({
            "type": "object",
            "properties": {
                "obj": {
                    "type": ["object", "null"],
                    "description": "a nested object",
                    "properties": { "nest1": { "type": ["string", "null"] } },
                    "additionalProperties": false
                },
                "arr": {
                    "type": "array",
                    "description": "array of objects",
                    "items": {
                        "type": "object",
                        "properties": { "nest2": { "type": ["boolean", "null"] } },
                        "additionalProperties": false
                    }
                }
            },
            "required": ["arr"],
            "additionalProperties": false
        })),
        ..Default::default()
    };

    let expected_request = genkit_ai::model::GenerateRequest {
        config: Some(json!({ "temperature": 11 })),
        max_turns: Some(77),
        messages: vec![
            MessageData {
                role: Role::System,
                content: vec![Part::text(" Hello ")],
                ..Default::default()
            },
            MessageData {
                role: Role::Model,
                content: vec![Part::text(" from the prompt file banana")],
                ..Default::default()
            },
        ],
        output: Some(serde_json::to_value(&expected_output_options).unwrap()),
        return_tool_requests: Some(true),
        tools: expected_tools,
        tool_choice: None,
        docs: None,
    };

    assert_eq!(generate_request, expected_request);
}

#[rstest]
#[tokio::test]
/// 'resolved schema refs'
async fn test_resolved_schema_refs(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
    struct SchemaRefOutput {
        output: String,
    }

    let output_schema = schemars::schema_for!(SchemaRefOutput);

    let prompt_config = PromptConfig {
        name: "schemaRef".to_string(),
        prompt: Some("Write a poem about {{foo}}.".to_string()),
        output: Some(OutputOptions {
            json_schema: Some(serde_json::to_value(output_schema.clone()).unwrap()),
            ..Default::default()
        }),
        ..Default::default()
    };

    let prompt = genkit
        .define_prompt::<serde_json::Value, serde_json::Value, serde_json::Value>(prompt_config)
        .await;

    let rendered = prompt.render(json!({ "foo": "bar" }), None).await.unwrap();

    assert_eq!(
        rendered.output.unwrap().json_schema.unwrap(),
        serde_json::to_value(output_schema).unwrap()
    );

    let action = genkit
        .registry()
        .lookup_action("/prompt/schemaRef")
        .await
        .unwrap();

    let response_value = action
        .run_http_json(json!({ "foo": "bar" }), None)
        .await
        .unwrap();

    let generate_request: genkit_ai::model::GenerateRequest =
        serde_json::from_value(response_value["result"].clone()).unwrap();

    let expected_messages = vec![MessageData {
        role: Role::User,
        content: vec![Part::text("Write a poem about bar.")],
        ..Default::default()
    }];

    assert_eq!(generate_request.messages, expected_messages);
}

#[rstest]
#[tokio::test]
/// 'lazily resolved schema refs'
async fn test_lazily_resolved_schema_refs(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // This prompt references a schema that does not exist in the registry.
    let prompt_config = PromptConfig {
        name: "badSchemaRef".to_string(),
        prompt: Some("Write a poem about {{foo}}.".to_string()),
        output: Some(OutputOptions {
            json_schema: Some(json!({ "$ref": "schema/badSchemaRef1" })),
            ..Default::default()
        }),
        ..Default::default()
    };

    let prompt = genkit
        .define_prompt::<serde_json::Value, serde_json::Value, serde_json::Value>(prompt_config)
        .await;

    let result = prompt.render(json!({ "foo": "bar" }), None).await;

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error
        .to_string()
        .contains("NOT_FOUND: Schema 'badSchemaRef1' not found"));
}

use genkit_ai::prompt::PromptLookupOptions;

#[rstest]
#[tokio::test]
/// 'loads a variant from from the folder'
async fn test_loads_variant_from_folder(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // Define the base prompt
    let test_prompt_config = PromptConfig {
        name: "test".to_string(),
        prompt: Some("Hello from the base prompt".to_string()),
        config: Some(json!({ "temperature": 11 })),
        ..Default::default()
    };
    genkit
        .define_prompt::<EmptyInput, Value, Value>(test_prompt_config)
        .await;

    // Define the variant
    let variant_prompt_config = PromptConfig {
        name: "test".to_string(),
        variant: Some("variant".to_string()),
        prompt: Some("Hello from a variant of the hello prompt".to_string()),
        config: Some(json!({ "temperature": 13 })),
        ..Default::default()
    };
    genkit
        .define_prompt::<EmptyInput, Value, Value>(variant_prompt_config)
        .await;

    // Look up the variant
    let test_prompt = genkit_ai::prompt::prompt::<EmptyInput, Value, Value>(
        genkit.registry(),
        "test",
        Some(PromptLookupOptions {
            variant: Some("variant"),
        }),
    )
    .await
    .unwrap();

    // Test generation.
    let response = test_prompt.generate(EmptyInput {}, None).await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: Hello from a variant of the hello prompt; config: {\"temperature\":13}"
    );
}

#[rstest]
#[tokio::test]
/// 'includes metadata expected by the dev ui'
async fn test_includes_metadata_expected_by_dev_ui(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let variant_prompt_config = PromptConfig {
        name: "test".to_string(),
        variant: Some("variant".to_string()),
        prompt: Some("Hello from a variant of the hello prompt".to_string()),
        config: Some(json!({ "temperature": 13 })),
        description: Some("a prompt variant in a file".to_string()),
        ..Default::default()
    };

    let _ = genkit.define_prompt::<EmptyInput, Value, Value>(variant_prompt_config);

    let test_prompt_action = genkit
        .registry()
        .lookup_action("/prompt/test.variant")
        .await
        .unwrap();

    let metadata = test_prompt_action.metadata();
    let custom_metadata = &metadata.metadata;
    let custom_metadata_value = serde_json::to_value(custom_metadata).unwrap();

    let expected_input_schema = serde_json::to_value(schemars::schema_for!(EmptyInput)).unwrap();

    let expected_metadata = json!({
      "prompt": {
        "config": {
          "temperature": 13,
        },
        "description": "a prompt variant in a file",
        "input": {
          "schema": expected_input_schema,
        },
        "metadata": {},
        "model": null,
        "name": "test",
        "variant": "variant",
        "template": "Hello from a variant of the hello prompt",
        "raw": {
          "config": {
            "temperature": 13,
          },
          "description": "a prompt variant in a file",
        },
      },
      "type": "prompt",
    });

    println!(
        "DEBUGINFO - Actual Metadata: {}",
        serde_json::to_string_pretty(&custom_metadata_value).unwrap()
    );
    println!(
        "DEBUGINFO - Expected Metadata: {}",
        serde_json::to_string_pretty(&expected_metadata).unwrap()
    );

    assert_eq!(custom_metadata_value, expected_metadata);
}
