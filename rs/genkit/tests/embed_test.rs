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
    document::{Document, Part},
    embed::{
        define_embedder, embed, DefineEmbedderOptions, EmbedRequest, EmbedResponse, EmbedderConfig,
        EmbedderRef, EmbedderRequest as GenkitEmbedderRequest, EmbeddingData,
    },
    genkit,
    registry::Registry,
    Genkit,
};
use genkit_core::action::ActionFnArg;
use genkit_core::error::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

// This module encapsulates the test helpers.
mod helpers {
    use super::*;

    // To store the last request for inspection.
    #[derive(Default, Clone)]
    pub struct LastRequest {
        pub input: Vec<Document>,
        pub options: Option<EmbedderConfig>,
    }

    pub fn define_test_embedder(
        registry: &mut Registry,
        last_request: Arc<std::sync::Mutex<LastRequest>>,
    ) {
        let embedder_action = define_embedder(
            DefineEmbedderOptions {
                name: "echoEmbedder".to_string(),
                ..Default::default()
            },
            move |req: GenkitEmbedderRequest,
                  _args: ActionFnArg<genkit::embed::EmbedResponseChunk>| {
                let last_request_clone = last_request.clone();
                Box::pin(async move {
                    let mut last_req = last_request_clone.lock().unwrap();
                    last_req.input = req.input;
                    last_req.options = req.options;

                    Ok(EmbedResponse {
                        embeddings: vec![EmbeddingData {
                            embedding: vec![1.0, 2.0, 3.0, 4.0],
                        }],
                    })
                })
            },
        );
        let action_name = embedder_action.name().to_string();
        registry.register_action(Arc::new(embedder_action)).unwrap();
    }
}

struct TestContext {
    ai: Genkit,
    last_request: Arc<std::sync::Mutex<helpers::LastRequest>>,
}

fn setup() -> TestContext {
    let mut ai = genkit(Default::default());
    let last_request = Arc::new(std::sync::Mutex::new(helpers::LastRequest::default()));
    helpers::define_test_embedder(&mut ai.registry(), last_request.clone());
    TestContext { ai, last_request }
}

#[tokio::test]
async fn passes_string_content_as_docs() -> Result<()> {
    let context = setup();
    let response = embed(
        &context.ai,
        EmbedRequest {
            embedder: "echoEmbedder".into(),
            content: "hi".into(),
            options: None,
        },
    )
    .await?;

    let last_req = context.last_request.lock().unwrap();
    assert_eq!(last_req.input.len(), 1);
    assert_eq!(last_req.input[0], Document::from_text("hi"));
    assert!(last_req.options.is_none());

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[tokio::test]
async fn passes_docs_content_as_docs() -> Result<()> {
    let context = setup();
    let response = embed(
        &context.ai,
        EmbedRequest {
            embedder: "echoEmbedder".into(),
            content: Document::from_text("hi").into(),
            options: None,
        },
    )
    .await?;

    let last_req = context.last_request.lock().unwrap();
    assert_eq!(last_req.input.len(), 1);
    assert_eq!(last_req.input[0], Document::from_text("hi"));
    assert!(last_req.options.is_none());

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[tokio::test]
async fn takes_config_passed_to_embed() -> Result<()> {
    let context = setup();
    let mut options = HashMap::new();
    options.insert("temperature".to_string(), Value::from(11));

    let response = embed(
        &context.ai,
        EmbedRequest {
            embedder: "echoEmbedder".into(),
            content: "hi".into(),
            options: Some(EmbedderConfig {
                options: Some(options),
                ..Default::default()
            }),
        },
    )
    .await?;

    let last_req = context.last_request.lock().unwrap();
    let config = last_req.options.as_ref().unwrap();
    assert_eq!(
        config.options.as_ref().unwrap().get("temperature").unwrap(),
        &Value::from(11)
    );

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[tokio::test]
async fn merges_config_from_the_ref() -> Result<()> {
    let context = setup();
    let mut options = HashMap::new();
    options.insert("temperature".to_string(), Value::from(11));

    let mut embedder_ref_config = EmbedderConfig::default();
    let mut ref_options = HashMap::new();
    ref_options.insert("version".to_string(), json!("abc"));
    embedder_ref_config.options = Some(ref_options);

    let response = embed(
        &context.ai,
        EmbedRequest {
            embedder: EmbedderRef::new("echoEmbedder")
                .with_config(embedder_ref_config)
                .into(),
            content: "hi".into(),
            options: Some(EmbedderConfig {
                options: Some(options),
                ..Default::default()
            }),
        },
    )
    .await?;

    let last_req = context.last_request.lock().unwrap();
    let config = last_req.options.as_ref().unwrap();
    assert_eq!(
        config.options.as_ref().unwrap().get("temperature").unwrap(),
        &Value::from(11)
    );
    assert_eq!(
        config.options.as_ref().unwrap().get("version").unwrap(),
        &json!("abc")
    );

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[tokio::test]
async fn picks_up_the_top_level_version_from_the_ref() -> Result<()> {
    let context = setup();
    let mut options = HashMap::new();
    options.insert("temperature".to_string(), Value::from(11));

    let mut embedder_ref = EmbedderRef::new("echoEmbedder");
    embedder_ref.version = Some("abc".to_string());

    let response = embed(
        &context.ai,
        EmbedRequest {
            embedder: embedder_ref.into(),
            content: "hi".into(),
            options: Some(EmbedderConfig {
                options: Some(options),
                ..Default::default()
            }),
        },
    )
    .await?;

    let last_req = context.last_request.lock().unwrap();
    let config = last_req.options.as_ref().unwrap();
    assert_eq!(
        config.options.as_ref().unwrap().get("temperature").unwrap(),
        &Value::from(11)
    );
    assert_eq!(config.version.as_ref().unwrap(), "abc");

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}
