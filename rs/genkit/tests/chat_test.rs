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
    Genkit, PromptConfig, PromptGenerateOptions,
};
use genkit_ai::{session::ChatOptions, MessageData};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use genkit_ai::session::SessionUpdater;
use std::sync::Arc;
use tokio_stream::StreamExt;

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
}

#[rstest]
#[tokio::test]
/// 'maintains history in the session'
async fn test_maintains_history_in_session(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;
    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat::<()>(Some(ChatOptions {
            base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                model: Some(genkit::model::Model::Name("echoModel".to_string())),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let mut response = chat.send("hi").await.unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi; config: {}");

    response = chat.send("bye").await.unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi,Echo: hi,; config: {},bye; config: {}"
    );

    let expected_messages = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hi")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Echo: hi"), Part::text("; config: {}")],
            ..Default::default()
        },
        MessageData {
            role: Role::User,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![
                Part::text("Echo: hi,Echo: hi,; config: {},bye"),
                Part::text("; config: {}"),
            ],
            ..Default::default()
        },
    ];
    assert_eq!(response.messages().unwrap(), expected_messages);
}

#[rstest]
#[tokio::test]
/// 'maintains history in the session with streaming'
async fn test_maintains_history_in_session_with_streaming(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;
    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat::<()>(Some(ChatOptions {
            base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                model: Some(genkit::model::Model::Name("echoModel".to_string())),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let stream_response = chat.send_stream("hi").await.unwrap();
    let mut response_handle = stream_response.response;
    let mut stream = stream_response.stream;

    let mut chunks: Vec<String> = Vec::new();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunks.push(chunk.text());
    }
    let mut final_response = response_handle.await.unwrap().unwrap();
    assert_eq!(final_response.text().unwrap(), "Echo: hi; config: {}");
    assert_eq!(chunks, vec!["3", "2", "1"]);

    let stream_response_2 = chat.send_stream("bye").await.unwrap();
    response_handle = stream_response_2.response;
    stream = stream_response_2.stream;

    chunks.clear();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        chunks.push(chunk.text());
    }
    final_response = response_handle.await.unwrap().unwrap();

    assert_eq!(chunks, vec!["3", "2", "1"]);
    assert_eq!(
        final_response.text().unwrap(),
        "Echo: hi,Echo: hi,; config: {},bye; config: {}"
    );

    let expected_messages = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hi")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Echo: hi"), Part::text("; config: {}")],
            ..Default::default()
        },
        MessageData {
            role: Role::User,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![
                Part::text("Echo: hi,Echo: hi,; config: {},bye"),
                Part::text("; config: {}"),
            ],
            ..Default::default()
        },
    ];
    assert_eq!(final_response.messages().unwrap(), expected_messages);
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone, Default)]
struct NameInput {
    name: String,
}

#[rstest]
#[tokio::test]
/// 'can init a session with a prompt'
async fn test_init_session_with_prompt(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let prompt = genkit.define_prompt::<NameInput, Value, Value>(PromptConfig {
        name: "hi".to_string(),
        prompt: Some("hi {{name}}".to_string()),
        ..Default::default()
    });

    let rendered = prompt
        .render(
            NameInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                config: Some(json!({ "temperature": 11 })),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat::<()>(Some(ChatOptions {
            base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                model: rendered.model,
                docs: rendered.docs,
                messages: rendered.messages.unwrap_or_default(),
                tools: rendered.tools,
                tool_choice: rendered.tool_choice,
                config: rendered.config,
                output: rendered.output,
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let response = chat.send("hi").await.unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi Genkit,hi; config: {\"temperature\":11}"
    );
}

#[rstest]
#[tokio::test]
/// 'can start chat from a prompt'
async fn test_start_chat_from_prompt(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let preamble = genkit.define_prompt::<(), Value, Value>(PromptConfig {
        name: "hi".to_string(),
        config: Some(json!({ "version": "abc" })),
        messages: Some(vec![MessageData {
            role: Role::User,
            content: vec![Part::text("hi from template")],
            ..Default::default()
        }]),
        ..Default::default()
    });

    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat::<()>(Some(ChatOptions {
            preamble: Some(&preamble),
            ..Default::default()
        }))
        .await
        .unwrap();

    let response = chat.send("send it").await.unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi from template,send it; config: {\"version\":\"abc\"}"
    );
}

#[rstest]
#[tokio::test]
/// 'can start chat from a prompt with input'
async fn test_start_chat_from_prompt_with_input(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let preamble = genkit.define_prompt::<NameInput, Value, Value>(PromptConfig {
        name: "hi".to_string(),
        config: Some(json!({ "version": "abc" })),
        prompt: Some("hi {{name}} from template".to_string()),
        ..Default::default()
    });

    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat(Some(ChatOptions {
            preamble: Some(&preamble),
            prompt_render_input: Some(NameInput {
                name: "Genkit".to_string(),
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let response = chat.send("send it").await.unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi Genkit from template,send it; config: {\"version\":\"abc\"}"
    );
}

#[rstest]
#[tokio::test]
/// 'can start chat from a prompt file with input'
async fn test_start_chat_from_prompt_file_with_input(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // Define the prompt to simulate it being loaded from a file
    genkit.define_prompt::<NameInput, Value, Value>(PromptConfig {
        name: "chat_preamble".to_string(),
        config: Some(json!({ "version": "abc" })),
        prompt: Some("hi {{name}} from template".to_string()),
        ..Default::default()
    });

    // Look up the prompt
    let preamble = genkit_ai::prompt::prompt(genkit.registry(), "chat_preamble", None)
        .await
        .unwrap();

    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat(Some(ChatOptions {
            preamble: Some(&preamble),
            prompt_render_input: Some(NameInput {
                name: "Genkit".to_string(),
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let response = chat.send("send it").await.unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi Genkit from template,send it; config: {\"version\":\"abc\"}"
    );
}

#[rstest]
#[tokio::test]
/// 'can send a rendered prompt to chat'
async fn test_send_rendered_prompt_to_chat(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let prompt = genkit.define_prompt::<NameInput, Value, Value>(PromptConfig {
        name: "hi".to_string(),
        config: Some(json!({ "version": "abc" })),
        prompt: Some("hi {{name}}".to_string()),
        ..Default::default()
    });

    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    let chat = Arc::new(session)
        .chat::<()>(Some(ChatOptions {
            base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                model: Some(genkit::model::Model::Name("echoModel".to_string())),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let rendered = prompt
        .render(
            NameInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                config: Some(json!({ "temperature": 11 })),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let response = chat.send(rendered).await.unwrap();

    let right_str = "Echo: hi Genkit; config: {\"temperature\":11,\"version\":\"abc\"}";
    let left_str = response.text().unwrap();

    let prefix = "Echo: hi Genkit; config: ";
    assert!(
        left_str.starts_with(prefix),
        "Response prefix did not match. Got: '{}'",
        left_str
    );

    let left_json_str = left_str.strip_prefix(prefix).unwrap();
    let right_json_str = right_str.strip_prefix(prefix).unwrap();

    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    assert_eq!(left_value, right_value);
}

#[rstest]
#[tokio::test]
/// 'initializes chat with history'
async fn test_initializes_chat_with_history(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // The history we want to load into the chat.
    let history = vec![
        MessageData::user(vec![Part::text("hi")]),
        MessageData {
            role: Role::Model,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
    ];

    let session =
        genkit_ai::Session::<()>::new(Arc::new(genkit.registry().clone()), None, None, None)
            .await
            .unwrap();

    // Manually update the session's message store to simulate a pre-existing conversation.
    session
        .update_messages(genkit_ai::chat::MAIN_THREAD, &history)
        .await
        .unwrap();

    // The system prompt to start the new chat with.
    let system_prompt = MessageData {
        role: Role::System,
        content: vec![Part::text("system instructions")],
        metadata: Some([("preamble".to_string(), json!(true))].into()),
    };

    // Create the chat. It should automatically load the history we just saved.
    let chat = Arc::new(session)
        .chat::<()>(Some(ChatOptions {
            base_options: Some(genkit_ai::generate::BaseGenerateOptions {
                model: Some(genkit::model::Model::Name("echoModel".to_string())),
                messages: vec![system_prompt], // This becomes the new preamble
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();

    let response = chat.send("hi again").await.unwrap();

    // The `echoModel` doesn't support system prompts, so middleware simulates it.
    let mut expected_messages = vec![
        MessageData {
            role: Role::User,
            content: vec![
                Part::text("SYSTEM INSTRUCTIONS:\n"),
                Part::text("system instructions"),
            ],
            metadata: Some([("preamble".to_string(), json!(true))].into()),
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Understood.")],
            metadata: Some([("preamble".to_string(), json!(true))].into()),
        },
    ];
    // The original history follows the simulated system prompt.
    expected_messages.extend(history);
    // Finally, add the latest turn.
    expected_messages.push(MessageData::user(vec![Part::text("hi again")]));
    expected_messages.push(MessageData {
        role: Role::Model,
        content: vec![
            Part::text(
                "Echo: SYSTEM INSTRUCTIONS:\nsystem instructions,Understood.,hi,bye,hi again",
            ),
            Part::text("; config: {}"),
        ],
        ..Default::default()
    });

    assert_eq!(response.messages().unwrap(), expected_messages);
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone, Default)]
struct MyState {
    name: String,
}

#[rstest]
#[tokio::test]
/// 'updates the preamble on fresh chat instance'
async fn test_updates_preamble_on_fresh_chat_instance(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let agent = genkit.define_prompt::<(), Value, Value>(PromptConfig {
        name: "agent".to_string(),
        messages: Some(vec![MessageData {
            role: Role::System,
            content: vec![Part::text("greet {{ @state.name }}")],
            ..Default::default()
        }]),
        config: Some(json!({ "temperature": 2 })),
        ..Default::default()
    });

    let session = genkit_ai::Session::new(
        Arc::new(genkit.registry().clone()),
        None, // use in-memory store
        None, // new session id
        Some(MyState {
            name: "Pavel".to_string(),
        }),
    )
    .await
    .unwrap();
    let session = Arc::new(session);

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

    // Middleware simulates the system prompt
    let expected_messages1 = vec![
        MessageData::user(vec![
            Part::text("SYSTEM INSTRUCTIONS:\n"),
            Part::text("greet Pavel"),
        ]),
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Understood.")],
            ..Default::default()
        },
        MessageData::user(vec![Part::text("hi")]),
        MessageData {
            role: Role::Model,
            content: vec![
                Part::text("Echo: SYSTEM INSTRUCTIONS:\ngreet Pavel,Understood.,hi"),
                Part::text("; config: {\"temperature\":2}"),
            ],
            ..Default::default()
        },
    ];
    assert_eq!(response1.messages().unwrap(), expected_messages1);

    session
        .update_state(MyState {
            name: "Michael".to_string(),
        })
        .await
        .unwrap();

    // Create a new chat from the same session to pick up the state change.
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

    let response2 = fresh_chat.send("hi again").await.unwrap();

    let mut expected_messages2 = vec![
        MessageData::user(vec![
            Part::text("SYSTEM INSTRUCTIONS:\n"),
            Part::text("greet Michael"), // Preamble is updated
        ]),
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Understood.")],
            ..Default::default()
        },
    ];
    // History from the first chat is preserved
    expected_messages2.extend(response1.messages().unwrap());
    expected_messages2.push(MessageData::user(vec![Part::text("hi again")]));
    expected_messages2.push(MessageData {
        role: Role::Model,
        content: vec![
            Part::text("Echo: SYSTEM INSTRUCTIONS:\ngreet Michael,Understood.,SYSTEM INSTRUCTIONS:\ngreet Pavel,Understood.,hi,Echo: SYSTEM INSTRUCTIONS:\ngreet Pavel,Understood.,hi; config: {\"temperature\":2},hi again"),
            Part::text("; config: {\"temperature\":2}"),
        ],
        ..Default::default()
    });

    assert_eq!(response2.messages().unwrap(), expected_messages2);
}
