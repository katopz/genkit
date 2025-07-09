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

use async_trait::async_trait;
use futures_util::StreamExt;
use genkit::Genkit;
use genkit_ai::document::Part;
use genkit_ai::generate::BaseGenerateOptions;
use genkit_ai::message::{MessageData, Role};
use genkit_ai::model::GenerateRequest;
use genkit_ai::session::{
    ChatOptions, InMemorySessionStore, Session, SessionData, SessionStore, SessionUpdater,
};
use rstest::{fixture, rstest};
use serde::{Deserialize, Serialize};
use serde_json::{json, to_value, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[rstest]
#[tokio::test]
async fn test_maintains_history_in_the_session(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let session = Arc::new(
        Session::<Value>::new(genkit.registry().clone().into(), None, None, None)
            .await
            .unwrap(),
    );

    let chat = session.chat::<Value>(None).await.unwrap();

    // First message
    let response1 = chat.send("hi").await.unwrap();
    assert_eq!(response1.text().unwrap(), "Echo: hi; config: {}");

    // Second message
    let response2 = chat.send("bye").await.unwrap();
    assert_eq!(
        response2.text().unwrap(),
        "Echo: hi,Echo: hi,; config: {},bye; config: {}"
    );

    // Verify message history
    assert_eq!(
        to_value(response2.messages().unwrap()).unwrap(),
        json!([
          { "content": [{ "text": "hi" }], "role": "user" },
          {
            "content": [{ "text": "Echo: hi" }, { "text": "; config: {}" }],
            "role": "model",
          },
          { "content": [{ "text": "bye" }], "role": "user" },
          {
            "content": [
              { "text": "Echo: hi,Echo: hi,; config: {},bye" },
              { "text": "; config: {}" },
            ],
            "role": "model",
          },
        ])
    );
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct MyState {
    foo: String,
}

#[derive(Clone)]
struct CapturingStore<S> {
    saved_data: Arc<Mutex<Option<SessionData<S>>>>,
}

impl<S> CapturingStore<S> {
    fn new(saved_data: Arc<Mutex<Option<SessionData<S>>>>) -> Self {
        Self { saved_data }
    }
}

#[async_trait]
impl<S: Clone + Send + Sync + 'static> SessionStore<S> for CapturingStore<S> {
    async fn get(&self, _session_id: &str) -> genkit::error::Result<Option<SessionData<S>>> {
        Ok(self.saved_data.lock().unwrap().clone())
    }

    async fn save(&self, _session_id: &str, data: &SessionData<S>) -> genkit::error::Result<()> {
        *self.saved_data.lock().unwrap() = Some(data.clone());
        Ok(())
    }
}

#[rstest]
#[tokio::test]
async fn test_sends_ready_to_serialize_data_to_session_store(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    let saved_data = Arc::new(Mutex::new(None));
    let store = Arc::new(CapturingStore::new(saved_data.clone()));

    let session = Session::new(
        genkit.registry().clone().into(),
        Some(store),
        None,
        Some(MyState {
            foo: "bar".to_string(),
        }),
    )
    .await
    .unwrap();

    let messages = vec![MessageData {
        role: Role::User,
        content: vec![Part::text("hello there")],
        ..Default::default()
    }];

    session.update_messages("main", &messages).await.unwrap();

    let captured_data = saved_data.lock().unwrap().clone().unwrap();

    let mut expected_threads = HashMap::new();
    expected_threads.insert("main".to_string(), messages);

    assert_eq!(
        captured_data.state,
        Some(MyState {
            foo: "bar".to_string()
        })
    );
    assert_eq!(captured_data.threads, expected_threads);
}

#[rstest]
#[tokio::test]
async fn test_maintains_multithreaded_history_in_the_session(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let store = Arc::new(InMemorySessionStore::<Value>::new());
    let session = Arc::new(
        Session::new(
            genkit.registry().clone().into(),
            Some(store.clone()),
            None,
            Some(json!({ "name": "Genkit" })),
        )
        .await
        .unwrap(),
    );

    let main_chat = session.chat::<Value>(None).await.unwrap();
    let response = main_chat.send("hi main").await.unwrap();
    assert_eq!(response.text().unwrap(), "Echo: hi main; config: {}");

    let mut lawyer_chat_opts: ChatOptions<Value, Value> = ChatOptions::default();
    lawyer_chat_opts.thread_name = Some("lawyerChat".to_string());
    lawyer_chat_opts.base_options = Some(BaseGenerateOptions {
        messages: vec![MessageData {
            role: Role::System,
            content: vec![Part::text("talk like a lawyer")],
            metadata: Some(
                [("preamble".to_string(), json!(true))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
        }],
        ..Default::default()
    });
    let lawyer_chat = session.chat::<Value>(Some(lawyer_chat_opts)).await.unwrap();
    let response = lawyer_chat.send("hi lawyerChat").await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: system: talk like a lawyer,hi lawyerChat; config: {}"
    );

    let mut pirate_chat_opts: ChatOptions<Value, Value> = ChatOptions::default();
    pirate_chat_opts.thread_name = Some("pirateChat".to_string());
    pirate_chat_opts.base_options = Some(BaseGenerateOptions {
        messages: vec![MessageData {
            role: Role::System,
            content: vec![Part::text("talk like a pirate")],
            metadata: Some(
                [("preamble".to_string(), json!(true))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
        }],
        ..Default::default()
    });
    let pirate_chat = session.chat::<Value>(Some(pirate_chat_opts)).await.unwrap();
    let response = pirate_chat.send("hi pirateChat").await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: system: talk like a pirate,hi pirateChat; config: {}"
    );

    let mut got_state = store.get(&session.id).await.unwrap().unwrap();
    got_state.id = "".to_string(); // ignore id

    assert_eq!(
        to_value(got_state).unwrap(),
        json!({
            "id": "",
            "state": {
                "name": "Genkit"
            },
            "threads": {
                "main": [
                    { "content": [{ "text": "hi main" }], "role": "user" },
                    {
                        "content": [{ "text": "Echo: hi main" }, { "text": "; config: {}" }],
                        "role": "model",
                    },
                ],
                "lawyerChat": [
                    {
                        "content": [{ "text": "talk like a lawyer" }],
                        "role": "system",
                        "metadata": { "preamble": true },
                    },
                    { "content": [{ "text": "hi lawyerChat" }], "role": "user" },
                    {
                        "content": [
                            { "text": "Echo: system: talk like a lawyer,hi lawyerChat" },
                            { "text": "; config: {}" },
                        ],
                        "role": "model",
                    },
                ],
                "pirateChat": [
                    {
                        "content": [{ "text": "talk like a pirate" }],
                        "role": "system",
                        "metadata": { "preamble": true },
                    },
                    { "content": [{ "text": "hi pirateChat" }], "role": "user" },
                    {
                        "content": [
                            { "text": "Echo: system: talk like a pirate,hi pirateChat" },
                            { "text": "; config: {}" },
                        ],
                        "role": "model",
                    },
                ],
            },
        })
    );
}

#[rstest]
#[tokio::test]
async fn test_maintains_history_in_the_session_with_streaming(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let session = Arc::new(
        Session::<Value>::new(genkit.registry().clone().into(), None, None, None)
            .await
            .unwrap(),
    );

    let chat = session.chat::<Value>(None).await.unwrap();

    let stream_resp1 = chat.send_stream("hi").await.unwrap();
    let mut chunks1 = Vec::new();
    let mut stream1 = stream_resp1.stream;
    while let Some(chunk_result) = stream1.next().await {
        chunks1.push(chunk_result.unwrap().text());
    }
    let response1 = stream_resp1.response.await.unwrap().unwrap();

    assert_eq!(response1.text().unwrap(), "Echo: hi; config: {}");
    assert_eq!(chunks1, vec!["3", "2", "1"]);

    let stream_resp2 = chat.send_stream("bye").await.unwrap();
    let mut chunks2 = Vec::new();
    let mut stream2 = stream_resp2.stream;
    while let Some(chunk_result) = stream2.next().await {
        chunks2.push(chunk_result.unwrap().text());
    }
    let response2 = stream_resp2.response.await.unwrap().unwrap();

    assert_eq!(chunks2, vec!["3", "2", "1"]);
    assert_eq!(
        response2.text().unwrap(),
        "Echo: hi,Echo: hi,; config: {},bye; config: {}"
    );

    assert_eq!(
        to_value(response2.messages().unwrap()).unwrap(),
        json!([
          { "content": [{ "text": "hi" }], "role": "user" },
          {
            "content": [{ "text": "Echo: hi" }, { "text": "; config: {}" }],
            "role": "model",
          },
          { "content": [{ "text": "bye" }], "role": "user" },
          {
            "content": [
              { "text": "Echo: hi,Echo: hi,; config: {},bye" },
              { "text": "; config: {}" },
            ],
            "role": "model",
          },
        ])
    );
}

#[rstest]
#[tokio::test]
async fn test_stores_state_and_messages_in_the_store(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let store = Arc::new(InMemorySessionStore::<Value>::new());

    let session = Arc::new(
        Session::new(
            genkit.registry().clone().into(),
            Some(store.clone()),
            None,
            Some(json!({"foo": "bar"})),
        )
        .await
        .unwrap(),
    );
    let chat = session.chat::<Value>(None).await.unwrap();

    chat.send("hi").await.unwrap();
    chat.send("bye").await.unwrap();

    let mut state = store.get(&session.id).await.unwrap().unwrap();
    state.id = "".to_string(); // ignore id

    assert_eq!(
        to_value(state).unwrap(),
        json!({
            "id": "",
            "state": {
                "foo": "bar"
            },
            "threads": {
                "main": [
                  { "content": [{ "text": "hi" }], "role": "user" },
                  {
                    "content": [{ "text": "Echo: hi" }, { "text": "; config: {}" }],
                    "role": "model",
                  },
                  { "content": [{ "text": "bye" }], "role": "user" },
                  {
                    "content": [
                      { "text": "Echo: hi,Echo: hi,; config: {},bye" },
                      { "text": "; config: {}" },
                    ],
                    "role": "model",
                  },
                ]
            }
        })
    );
}
