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

use genkit::{
    model::{Part, Role},
    Genkit,
};
use genkit_ai::{session::ChatOptions, MessageData};
use rstest::{fixture, rstest};

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
        "Echo: hiEcho: hi; config: {}bye; config: {}"
    );

    let expected_messages = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hi")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Echo: hi; config: {}")],
            ..Default::default()
        },
        MessageData {
            role: Role::User,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Echo: hiEcho: hi; config: {}bye; config: {}")],
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
        "Echo: hiEcho: hi; config: {}bye; config: {}"
    );

    let expected_messages = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hi")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Echo: hi; config: {}")],
            ..Default::default()
        },
        MessageData {
            role: Role::User,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("Echo: hiEcho: hi; config: {}bye; config: {}")],
            ..Default::default()
        },
    ];
    assert_eq!(final_response.messages().unwrap(), expected_messages);
}
