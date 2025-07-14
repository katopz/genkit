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

use genkit::SessionStore;
use genkit::{Genkit, Session};
use genkit_ai::{
    self as genkit_ai,
    model::GenerateRequest,
    session::{ChatOptions, InMemorySessionStore},
};
use rstest::{fixture, rstest};
use serde_json::{json, to_value, Value};
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[rstest]
#[tokio::test]
/// 'loads session from store'
async fn test_loads_session_from_store(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    // Create a store
    let store = Arc::new(InMemorySessionStore::<Value>::new());

    // Create a session with the store
    let original_session = Session::<Value>::new(
        genkit.registry().clone().into(),
        Some(store.clone()),
        None,
        None,
    )
    .await
    .unwrap();
    let original_session = Arc::new(original_session);

    // Create a chat with config and send messages
    let chat_options = ChatOptions {
        config: Some(json!({ "temperature": 1 })),
        ..Default::default()
    };
    let chat = original_session
        .chat::<Value>(Some(chat_options))
        .await
        .unwrap();
    chat.send("hi").await.unwrap();
    chat.send("bye").await.unwrap();

    let session_id = original_session.id.clone();

    // Load the session from the store
    let loaded_session = Session::<Value>::new(
        genkit.registry().clone().into(),
        Some(store.clone()),
        Some(session_id.clone()),
        None,
    )
    .await
    .unwrap();
    let loaded_session = Arc::new(loaded_session);

    let main_chat = loaded_session.chat::<Value>(None).await.unwrap();

    // Verify message history is loaded
    assert_eq!(
        to_value(main_chat.messages().await).unwrap(),
        json!([
          { "content": [{ "text": "hi" }], "role": "user" },
          {
            "content": [{ "text": "Echo: hi" }, { "text": "; config: {\"temperature\":1}" }],
            "role": "model",
          },
          { "content": [{ "text": "bye" }], "role": "user" },
          {
            "content": [
              { "text": "Echo: hi,Echo: hi; config: {\"temperature\":1},bye" },
              { "text": "; config: {\"temperature\":1}" },
            ],
            "role": "model",
          },
        ])
    );

    // Send another message and verify state
    let response = main_chat.send("hi again").await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: hi,Echo: hi; config: {\"temperature\":1},bye,Echo: hi,Echo: hi; config: {\"temperature\":1},bye; config: {\"temperature\":1},hi again; config: {}"
    );

    // Verify final message history
    assert_eq!(
        to_value(main_chat.messages().await).unwrap(),
        json!([
          { "content": [{ "text": "hi" }], "role": "user" },
          {
            "content": [
              { "text": "Echo: hi" },
              { "text": "; config: {\"temperature\":1}" },
            ],
            "role": "model",
          },
          { "content": [{ "text": "bye" }], "role": "user" },
          {
            "content": [
              { "text": "Echo: hi,Echo: hi; config: {\"temperature\":1},bye" },
              { "text": "; config: {\"temperature\":1}" },
            ],
            "role": "model",
          },
          { "content": [{ "text": "hi again" }], "role": "user" },
          {
            "content": [
              {
                "text": "Echo: hi,Echo: hi; config: {\"temperature\":1},bye,Echo: hi,Echo: hi; config: {\"temperature\":1},bye; config: {\"temperature\":1},hi again",
              },
              { "text": "; config: {}" },
            ],
            "role": "model",
          },
        ])
    );

    // Verify state in store
    let state = store.get(&session_id).await.unwrap().unwrap();
    assert_eq!(
        to_value(state.threads).unwrap(),
        json!({
            "main": [
              { "content": [{ "text": "hi" }], "role": "user" },
              {
                "content": [
                  { "text": "Echo: hi" },
                  { "text": "; config: {\"temperature\":1}" },
                ],
                "role": "model",
              },
              { "content": [{ "text": "bye" }], "role": "user" },
              {
                "content": [
                  { "text": "Echo: hi,Echo: hi; config: {\"temperature\":1},bye" },
                  { "text": "; config: {\"temperature\":1}" },
                ],
                "role": "model",
              },
              { "content": [{ "text": "hi again" }], "role": "user" },
              {
                "content": [
                  {
                    "text": "Echo: hi,Echo: hi; config: {\"temperature\":1},bye,Echo: hi,Echo: hi; config: {\"temperature\":1},bye; config: {\"temperature\":1},hi again",
                  },
                  { "text": "; config: {}" },
                ],
                "role": "model",
              },
            ]
        })
    );
}
