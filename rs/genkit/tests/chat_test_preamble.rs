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
    model::{Part, Role},
    Genkit, PromptConfig,
};
use genkit_ai::{session::ChatOptions, MessageData};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use genkit_ai::session::SessionUpdater;
use std::sync::Arc;

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
}

#[cfg(test)]
/// 'preamble'
mod preamble_test {
    use super::*;
    use futures::lock::Mutex;
    use genkit::common::ToolChoice;
    use genkit::CreateSessionOptions;
    use genkit::GenerateOptions;
    use genkit::Model;
    use genkit::Result;
    use genkit::ToolArgument;
    use genkit_ai::generate::BaseGenerateOptions;
    use genkit_ai::{CandidateData, GenerateResponseData};

    #[tokio::test]
    #[ignore = "Ignoring until prompt-as-a-tool and preamble swapping features are fully implemented"]
    /// 'swaps out preamble on prompt tool invocation'
    async fn test_swaps_preamble_on_prompt_tool_invocation() -> Result<()> {
        let (genkit, pm) = helpers::genkit_with_programmable_model().await;

        let _agent_b_prompt = genkit.define_prompt(PromptConfig::<(), (), Value> {
            name: "agentB".to_string(),
            tools: Some(vec![ToolArgument::Name("agentA".to_string())]),
            tool_choice: Some(ToolChoice::Required),
            system: Some("agent b".to_string()),
            config: Some(json!({"temperature": 1})),
            description: Some("Agent B description".to_string()),
            ..Default::default()
        });

        let agent_a_prompt = genkit.define_prompt(PromptConfig::<(), Value, Value> {
            name: "agentA".to_string(),
            tools: Some(vec![ToolArgument::Name("agentB".to_string())]),
            tool_choice: Some(ToolChoice::Required),
            messages_fn: Some(Arc::new(|_, _, _| {
                Box::pin(async {
                    Ok(vec![MessageData {
                        role: Role::System,
                        content: vec![Part::text(" agent a")],
                        ..Default::default()
                    }])
                })
            })),
            config: Some(json!({"temperature": 2})),
            description: Some("Agent A description".to_string()),
            ..Default::default()
        });

        // === Step 1: Simple "hi" to agent A ===
        *pm.handler.lock().unwrap() = Arc::new(Box::new(|req, _| {
            let tool_choice = req
                .tool_choice
                .as_ref()
                .and_then(|v| v.as_str())
                .unwrap_or("undefined");
            let response_text = format!("hi from agent a (toolChoice: {})", tool_choice);
            Box::pin(async move {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text(response_text)],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));

        let session = genkit
            .create_session(CreateSessionOptions::<Value>::default())
            .await?;
        let chat = session
            .chat(Some(ChatOptions {
                preamble: Some(&agent_a_prompt),
                ..Default::default()
            }))
            .await?;

        let response = chat.send("hi").await?;
        assert_eq!(response.text()?, "hi from agent a (toolChoice: required)");

        let last_req = pm.last_request.lock().unwrap().clone().unwrap();
        assert_eq!(last_req.config, Some(json!({"temperature": 2})));
        assert_eq!(
            last_req.messages,
            vec![
                MessageData {
                    role: Role::System,
                    content: vec![Part::text(" agent a")],
                    metadata: Some([("preamble".to_string(), json!(true))].into())
                },
                MessageData::user(vec![Part::text("hi")]),
            ]
        );

        // === Step 2: Transfer to agent B ===
        let req_counter_b = Arc::new(Mutex::new(0));
        *pm.handler.lock().unwrap() = Arc::new(Box::new(move |req, _| {
            let tool_choice = req
                .tool_choice
                .as_ref()
                .and_then(|v| v.as_str())
                .unwrap_or("undefined")
                .to_string();
            let counter = Arc::clone(&req_counter_b);
            Box::pin(async move {
                let mut guard = counter.lock().await;
                let content = if *guard == 0 {
                    vec![Part::tool_request(
                        "agentB",
                        Some(json!({})),
                        Some("ref123".to_string()),
                    )]
                } else {
                    vec![Part::text(format!(
                        "hi from agent b (toolChoice: {})",
                        tool_choice
                    ))]
                };
                *guard += 1;
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content,
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));

        let response = chat.send("pls transfer to b").await?;
        assert_eq!(response.text()?, "hi from agent b (toolChoice: required)");

        let last_req_b = pm.last_request.lock().unwrap().clone().unwrap();
        assert_eq!(last_req_b.config, Some(json!({"temperature": 1})));
        assert_eq!(
            last_req_b.messages,
            vec![
                MessageData {
                    // Note: Preamble is now agent b's
                    role: Role::System,
                    content: vec![Part::text("agent b")],
                    metadata: Some([("preamble".to_string(), json!(true))].into())
                },
                MessageData::user(vec![Part::text("hi")]),
                MessageData {
                    role: Role::Model,
                    content: vec![Part::text("hi from agent a (toolChoice: required)")],
                    ..Default::default()
                },
                MessageData::user(vec![Part::text("pls transfer to b")]),
                MessageData {
                    // Model requests tool call
                    role: Role::Model,
                    content: vec![Part::tool_request(
                        "agentB",
                        Some(json!({})),
                        Some("ref123".to_string())
                    )],
                    ..Default::default()
                },
                MessageData {
                    // Framework provides tool response
                    role: Role::Tool,
                    content: vec![Part::tool_response(
                        "agentB",
                        Some(json!("transferred to agentB")),
                        Some("ref123".to_string())
                    )],
                    ..Default::default()
                }
            ]
        );

        // === Step 3: Transfer back to agent A ===
        let req_counter_a = Arc::new(Mutex::new(0));
        *pm.handler.lock().unwrap() = Arc::new(Box::new(move |_, _| {
            let counter = Arc::clone(&req_counter_a);
            Box::pin(async move {
                let mut guard = counter.lock().await;
                let content = if *guard == 0 {
                    vec![Part::tool_request(
                        "agentA",
                        Some(json!({})),
                        Some("ref123".to_string()),
                    )]
                } else {
                    vec![Part::text("hi from agent a")]
                };
                *guard += 1;
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content,
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));

        let response = chat.send("pls transfer to a").await?;
        assert_eq!(response.text()?, "hi from agent a");

        let last_req_a = pm.last_request.lock().unwrap().clone().unwrap();
        assert_eq!(last_req_a.config, Some(json!({"temperature": 2})));
        assert_eq!(
            last_req_a
                .messages
                .first()
                .unwrap()
                .content
                .first()
                .unwrap()
                .text,
            Some(" agent a".to_string()),
            "Preamble should have been restored to agent a's"
        );

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    /// 'updates the preamble on fresh chat instance'
    async fn test_updates_preamble_on_fresh_chat_instance(#[future] genkit_instance: Arc<Genkit>) {
        #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone, Default)]
        struct MyState {
            name: String,
        }

        let genkit = genkit_instance.await;

        // The agent prompt definition is the same.
        let agent = genkit.define_prompt::<(), Value, Value>(PromptConfig {
            name: "agent".to_string(),
            config: Some(json!({ "temperature": 2 })),
            messages: Some(vec![MessageData {
                role: Role::System,
                content: vec![Part::text("greet {{ @state.name }}")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            }]),
            ..Default::default()
        });

        // Create the session with initial state.
        let session = genkit_ai::Session::new(
            Arc::new(genkit.registry().clone()),
            None,
            None,
            Some(MyState {
                name: "Pavel".to_string(),
            }),
        )
        .await
        .unwrap();
        let session = Arc::new(session);

        // === First Turn ===
        let chat = session
            .chat(Some(ChatOptions {
                preamble: Some(&agent),
                base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                    model: Some(genkit::model::Model::Name("echoModel".to_string())),
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .await
            .unwrap();

        let response1 = chat.send("hi").await.unwrap();

        // Assertion 1: Matches the first assertion in the TS test.
        let expected_messages1 = vec![
            MessageData {
                role: Role::System,
                content: vec![Part::text("greet Pavel")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
            MessageData::user(vec![Part::text("hi")]),
            MessageData {
                role: Role::Model,
                content: vec![
                    Part::text("Echo: system:  greet Pavel,hi"),
                    Part::text("; config: {\"temperature\":2}"),
                ],
                ..Default::default()
            },
        ];
        assert_eq!(
            serde_json::to_value(response1.messages().unwrap()).unwrap(),
            serde_json::to_value(expected_messages1.clone()).unwrap()
        );

        // Update the session state.
        session
            .update_state(MyState {
                name: "Michael".to_string(),
            })
            .await
            .unwrap();

        // === Second Turn ===
        let fresh_chat = session
            .chat(Some(ChatOptions {
                preamble: Some(&agent),
                base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                    model: Some(genkit::model::Model::Name("echoModel".to_string())),
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .await
            .unwrap();

        // The TS test sends "hi" again.
        let response2 = fresh_chat.send("hi").await.unwrap();

        // Assertion 2: This is the cumulative history, matching the second TS assertion.
        let mut expected_messages2 = vec![MessageData {
            role: Role::System,
            content: vec![Part::text("greet Michael")], // New preamble
            metadata: Some([("preamble".to_string(), json!(true))].into()),
        }];
        // Add the history from the first response, but without its old preamble.
        expected_messages2.extend(
            expected_messages1
                .into_iter()
                .filter(|m| m.role != Role::System),
        );
        // Add the new user message for the second turn.
        expected_messages2.push(MessageData::user(vec![Part::text("hi")]));
        // Add the final model response for the second turn.
        expected_messages2.push(MessageData {
            role: Role::Model,
            content: vec![
                Part::text("Echo: system:  greet Michael,hi,Echo: system:  greet Pavel,hi,; config: {\"temperature\":2},hi"),
                Part::text("; config: {\"temperature\":2}"),
            ],
            ..Default::default()
        });

        assert_eq!(
            serde_json::to_value(response2.messages().unwrap()).unwrap(),
            serde_json::to_value(expected_messages2).unwrap()
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'initializes chat with history'
    async fn test_initializes_chat_with_history(#[future] genkit_instance: Arc<Genkit>) {
        let genkit = genkit_instance.await;

        // 1. Define the history that should already be in the session.
        let history = vec![
            MessageData::user(vec![Part::text("hi")]),
            MessageData {
                role: Role::Model,
                content: vec![Part::text("bye")],
                ..Default::default()
            },
        ];

        // 2. Create a session and manually save the history to its store.
        let session =
            genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
                .await
                .unwrap();
        session
            .update_messages(genkit_ai::chat::MAIN_THREAD, &history)
            .await
            .unwrap();

        // 3. Define the new system prompt for this chat instance.
        let system_prompt = MessageData {
            role: Role::System,
            content: vec![Part::text("system instructions")],
            metadata: Some([("preamble".to_string(), json!(true))].into()),
        };

        // 4. Create a new chat. It will load the history from the session store
        //    and prepend the new system prompt.
        let chat = Arc::new(session)
            .chat::<()>(Some(ChatOptions {
                base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                    model: Some(genkit::model::Model::Name("echoModel".to_string())),
                    messages: vec![system_prompt.clone()],
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .await
            .unwrap();

        // 5. Send a new message.
        let response = chat.send("hi again").await.unwrap();

        // 6. Construct the expected final history.
        let mut expected_messages = vec![system_prompt];
        expected_messages.extend(history);
        expected_messages.push(MessageData::user(vec![Part::text("hi again")]));
        expected_messages.push(MessageData {
            role: Role::Model,
            content: vec![
                Part::text("Echo: system: system instructions,hi,bye,hi again"),
                Part::text("; config: {}"),
            ],
            ..Default::default()
        });

        // 7. Compare the JSON Value representation for a robust test.
        let actual_json = serde_json::to_value(response.messages().unwrap()).unwrap();
        let expected_json = serde_json::to_value(expected_messages).unwrap();

        assert_eq!(actual_json, expected_json);
    }

    #[tokio::test]
    #[ignore = "Ignoring until history can be provided as part of the preamble during chat creation."]
    async fn test_initializes_chat_with_history_in_preamble() -> Result<()> {
        let (genkit, _last_request) = helpers::genkit_instance_for_test().await;

        #[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
        struct NameInput {
            name: String,
        }

        let hi_prompt = genkit.define_prompt::<NameInput, Value, Value>(PromptConfig {
            name: "hi".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            system: Some("system instructions".to_string()),
            prompt: Some("hi {{name}}".to_string()),
            input: Some(NameInput::default()),
            ..Default::default()
        });

        let history = vec![
            MessageData {
                role: Role::User,
                content: vec![Part::text("hi")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
            MessageData {
                role: Role::Model,
                content: vec![Part::text("bye")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
        ];

        // Manually simulate what the TS ai.chat(prompt, {messages}) does:
        // 1. Render the prompt.
        let rendered_opts: GenerateOptions = hi_prompt
            .render(
                NameInput {
                    name: "Genkit".to_string(),
                },
                None,
            )
            .await?;

        // 2. Combine history and the rendered prompt messages.
        let mut combined_messages = history.clone();
        // Mark the rendered prompt message as part of the preamble.
        let mut rendered_user_message = rendered_opts.messages.unwrap().pop().unwrap();
        rendered_user_message.metadata = Some([("preamble".to_string(), json!(true))].into());
        combined_messages.push(rendered_user_message);

        // 3. The system message from the prompt is also part of the preamble.
        combined_messages.insert(
            0,
            MessageData {
                role: Role::System,
                content: vec![Part::text("system instructions")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
        );

        let chat_opts: ChatOptions<'_, Value, Value> = ChatOptions {
            base_options: Some(BaseGenerateOptions {
                model: rendered_opts.model,
                messages: combined_messages,
                ..Default::default()
            }),
            ..Default::default()
        };

        let session = genkit.create_session(Default::default()).await?;
        let chat = session.chat(Some(chat_opts)).await?;

        let response = chat.send("hi again").await?;

        // Now, construct the expected final message list for assertion.
        let expected_messages = vec![
            MessageData {
                role: Role::System,
                content: vec![Part::text("system instructions")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
            MessageData {
                role: Role::User,
                content: vec![Part::text("hi")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
            MessageData {
                role: Role::Model,
                content: vec![Part::text("bye")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
            MessageData {
                role: Role::User,
                content: vec![Part::text("hi Genkit")],
                metadata: Some([("preamble".to_string(), json!(true))].into()),
            },
            MessageData {
                role: Role::User,
                content: vec![Part::text("hi again")],
                ..Default::default()
            },
            MessageData {
                role: Role::Model,
                content: vec![
                    Part::text("Echo: system: system instructions,hi,bye,hi Genkit,hi again"),
                    Part::text("; config: {}"),
                ],
                ..Default::default()
            },
        ];

        assert_eq!(response.messages()?, expected_messages);

        Ok(())
    }
}
