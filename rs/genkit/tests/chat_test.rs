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

use genkit::{
    genkit,
    model::{Part, Role},
    registry::Registry,
    GenerateRequest, Genkit,
};
use genkit_ai::{
    define_model, model::DefineModelOptions, CandidateData, GenerateResponseChunkData,
    GenerateResponseData, MessageData,
};
use genkit_core::{error::Result, registry::ErasedAction, ActionFnArg};
use std::sync::Arc;

// Defines and registers a simple model that echoes its input.
fn define_echo_model(registry: &mut Registry) {
    let model_action = define_model(
        DefineModelOptions {
            name: "echoModel".to_string(),
            ..Default::default()
        },
        move |req: GenerateRequest, _args: ActionFnArg<GenerateResponseChunkData>| {
            let req_clone = req.clone();
            Box::pin(async move {
                let text_to_echo = req_clone
                    .messages
                    .iter()
                    .map(|m| {
                        let content_text = m
                            .content
                            .iter()
                            .filter_map(|p| p.text.as_ref())
                            .map(|s| s.as_str())
                            .collect::<Vec<&str>>()
                            .join(",");

                        match m.role {
                            Role::User | Role::Model => content_text.to_string(),
                            _ => format!(
                                "{}: {}",
                                serde_json::to_string(&m.role)
                                    .unwrap_or_default()
                                    .trim_matches('"'),
                                content_text
                            ),
                        }
                    })
                    .collect::<Vec<String>>()
                    .join(",");

                let config_str =
                    serde_json::to_string(&req_clone.config).unwrap_or("{}".to_string());

                let response_message = MessageData {
                    role: Role::Model,
                    content: vec![
                        Part::text(format!("Echo: {}", text_to_echo)),
                        Part::text(format!("; config: {}", config_str)),
                    ],
                    ..Default::default()
                };

                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        finish_reason: Some(genkit::model::FinishReason::Stop),
                        message: response_message,
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        },
    );

    let action_name = model_action.name().to_string();
    registry.register_action(Arc::new(model_action)).unwrap();
}

fn setup() -> Genkit {
    let mut ai = genkit(Default::default());
    define_echo_model(&mut ai.registry());
    ai
}

#[tokio::test]
async fn maintains_history_in_the_session() -> Result<()> {
    let ai = setup();
    let mut session = new_session(&ai, None, None).await?;
    let mut chat = session.chat(None).await?;

    let response1 = chat.send("hi").await?;
    assert_eq!(
        response1.text()?,
        "Echo: hi; config: {\"model\":\"echoModel\"}"
    );

    let response2 = chat.send("bye").await?;
    assert_eq!(
        response2.text()?,
        "Echo: hi,Echo: hi,; config: {\"model\":\"echoModel\"},bye; config: {\"model\":\"echoModel\"}"
    );

    let last_request_messages = response2.request.unwrap().messages;

    let expected_history = vec![
        MessageData::user(vec![Part::text("hi")]),
        MessageData::model(vec![
            Part::text("Echo: hi"),
            Part::text("; config: {\"model\":\"echoModel\"}"),
        ]),
        MessageData::user(vec![Part::text("bye")]),
    ];

    assert_eq!(last_request_messages.len(), expected_history.len());
    for (i, msg) in last_request_messages.iter().enumerate() {
        assert_eq!(msg.role, expected_history[i].role);
        assert_eq!(msg.content, expected_history[i].content);
    }

    Ok(())
}
