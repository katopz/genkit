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

//! # `augment_with_context` Middleware Tests

use genkit_ai::model::{
    middleware::augment_with_context, AugmentWithContextOptions, BoxFuture, GenerateRequest,
    GenerateResponseData,
};
use genkit_core::error::Result;
use rstest::rstest;
use serde_json::{from_value, json};
use std::sync::{Arc, Mutex};

// Helper to invoke the middleware and capture the modified request.
async fn test_augment_request(
    req: GenerateRequest,
    options: Option<AugmentWithContextOptions>,
) -> GenerateRequest {
    let captured_req = Arc::new(Mutex::new(None));
    let captured_req_clone = captured_req.clone();

    let next = move |req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
        let mut guard = captured_req_clone.lock().unwrap();
        *guard = Some(req);
        Box::pin(async { Ok(Default::default()) })
    };

    let middleware = augment_with_context(options);
    middleware(req, Box::new(next)).await.unwrap();

    let took_captured_req = captured_req.lock().unwrap().take().unwrap();
    took_captured_req
}

#[rstest]
#[tokio::test]
async fn test_should_not_change_message_with_empty_context() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
    }))
    .unwrap();

    // Test with undefined context (docs is None)
    let req_with_no_docs = original_req.clone();
    let modified_req_no_docs = test_augment_request(req_with_no_docs, None).await;
    assert_eq!(modified_req_no_docs, original_req);

    // Test with empty context (docs is Some([]))
    let mut req_with_empty_docs = original_req.clone();
    req_with_empty_docs.docs = Some(vec![]);
    let modified_req_empty_docs = test_augment_request(req_with_empty_docs.clone(), None).await;
    assert_eq!(
        modified_req_empty_docs.messages,
        req_with_empty_docs.messages
    );
}

#[rstest]
#[tokio::test]
async fn test_should_not_change_a_message_that_already_has_a_context_part_with_content() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{
            "role": "user",
            "content": [{ "text": "first part", "metadata": { "purpose": "context" } }]
        }],
        "docs": [{ "content": [{ "text": "i am context" }] }]
    }))
    .unwrap();

    let modified_req = test_augment_request(original_req.clone(), None).await;
    assert_eq!(modified_req, original_req);
}

#[rstest]
#[tokio::test]
async fn test_should_augment_a_message_that_has_a_pending_context_part() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{
            "role": "user",
            "content": [{ "metadata": { "purpose": "context", "pending": true } }]
        }],
        "docs": [{ "content": [{ "text": "i am context" }] }]
    }))
    .unwrap();

    let modified_req = test_augment_request(original_req.clone(), None).await;
    let expected_text =
        "\n\nUse the following information to complete your task:\n\n- [0]: i am context\n\n";
    let last_message = modified_req.messages.last().unwrap();
    let last_part = last_message.content.last().unwrap();

    assert_eq!(last_message.content.len(), 1);
    assert_eq!(last_part.text.as_deref(), Some(expected_text));
    assert_eq!(
        last_part.metadata.as_ref().unwrap().get("purpose"),
        Some(&json!("context"))
    );
}

#[rstest]
#[tokio::test]
async fn test_should_append_a_new_text_part() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
        "docs": [
            { "content": [{ "text": "i am context" }] },
            { "content": [{ "text": "i am more context" }] }
        ]
    }))
    .unwrap();

    let modified_req = test_augment_request(original_req.clone(), None).await;
    let expected_text = "\n\nUse the following information to complete your task:\n\n- [0]: i am context\n- [1]: i am more context\n\n";
    let last_message = modified_req.messages.last().unwrap();
    let last_part = last_message.content.last().unwrap();

    assert_eq!(last_message.content.len(), 2);
    assert_eq!(last_part.text.as_deref(), Some(expected_text));
    assert_eq!(
        last_part.metadata.as_ref().unwrap().get("purpose"),
        Some(&json!("context"))
    );
}

#[rstest]
#[tokio::test]
async fn test_should_use_citation_key() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
        "docs": [
            { "content": [{ "text": "i am context" }], "metadata": { "uid": "first" } },
            { "content": [{ "text": "i am more context" }], "metadata": { "uid": "second" } }
        ]
    }))
    .unwrap();

    let options = Some(AugmentWithContextOptions {
        citation_key: Some("uid".to_string()),
        ..Default::default()
    });

    let modified_req = test_augment_request(original_req.clone(), options).await;
    let last_part = modified_req.messages[0].content.last().unwrap();
    let expected_text = "\n\nUse the following information to complete your task:\n\n- [first]: i am context\n- [second]: i am more context\n\n";

    assert_eq!(last_part.text.as_deref(), Some(expected_text));
}

#[rstest]
#[tokio::test]
async fn test_should_use_citation_fallback_order() {
    let original_req: GenerateRequest = from_value(json!({
                "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
                "docs": [
                    { "content": [{ "text": "i am context" }], "metadata": { "ref": "first", "id": "wrong" } },
                    { "content": [{ "text": "i am more context" }], "metadata": { "id": "second" } },
                    { "content": [{ "text": "i am even more context" }] }
                ]
            }))
            .unwrap();

    let modified_req = test_augment_request(original_req.clone(), None).await;
    let last_part = modified_req.messages[0].content.last().unwrap();
    let expected_text = "\n\nUse the following information to complete your task:\n\n- [first]: i am context\n- [second]: i am more context\n- [2]: i am even more context\n\n";

    assert_eq!(last_part.text.as_deref(), Some(expected_text));
}

#[rstest]
#[tokio::test]
async fn test_should_use_custom_item_template() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
        "docs": [
            { "content": [{ "text": "i am context" }], "metadata": { "uid": "first" } },
            { "content": [{ "text": "i am more context" }], "metadata": { "uid": "second" } }
        ]
    }))
    .unwrap();

    let options = Some(AugmentWithContextOptions {
        item_template: Some(Box::new(|d, _| {
            format!(
                "* ({}) -- {}\n",
                d.metadata
                    .as_ref()
                    .unwrap()
                    .get("uid")
                    .unwrap()
                    .as_str()
                    .unwrap(),
                d.text()
            )
        })),
        ..Default::default()
    });

    let modified_req = test_augment_request(original_req.clone(), options).await;
    let last_part = modified_req.messages[0].content.last().unwrap();
    let expected_text =
                "\n\nUse the following information to complete your task:\n\n* (first) -- i am context\n* (second) -- i am more context\n\n";

    assert_eq!(last_part.text.as_deref(), Some(expected_text));
}

#[rstest]
#[tokio::test]
async fn test_should_append_to_last_user_message() {
    let original_req: GenerateRequest = from_value(json!({
            "messages": [
                { "role": "user", "content": [{ "text": "first part" }] },
                { "role": "tool", "content": [{ "toolResponse": { "name": "testTool", "output": { "abc": 123 } } }] }
            ],
            "docs": [
                { "content": [{ "text": "i am context" }] },
                { "content": [{ "text": "i am more context" }] }
            ]
        }))
        .unwrap();

    let modified_req = test_augment_request(original_req.clone(), None).await;
    let last_user_message = modified_req
        .messages
        .iter()
        .rfind(|m| m.role == genkit_ai::Role::User)
        .unwrap();
    let last_part = last_user_message.content.last().unwrap();
    let expected_text = "\n\nUse the following information to complete your task:\n\n- [0]: i am context\n- [1]: i am more context\n\n";

    assert_eq!(last_part.text.as_deref(), Some(expected_text));
    assert_eq!(
        last_part.metadata.as_ref().unwrap().get("purpose"),
        Some(&json!("context"))
    );
}

#[rstest]
#[tokio::test]
async fn test_should_use_custom_preface() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
        "docs": [
            { "content": [{ "text": "i am context" }] },
            { "content": [{ "text": "i am more context" }] }
        ]
    }))
    .unwrap();

    let options = Some(AugmentWithContextOptions {
        preface: Some("\n\nCheck this out:\n\n".to_string()),
        ..Default::default()
    });

    let modified_req = test_augment_request(original_req.clone(), options).await;
    let last_part = modified_req.messages[0].content.last().unwrap();
    let expected_text = "\n\nCheck this out:\n\n- [0]: i am context\n- [1]: i am more context\n\n";

    assert_eq!(last_part.text.as_deref(), Some(expected_text));
    assert_eq!(
        last_part.metadata.as_ref().unwrap().get("purpose"),
        Some(&json!("context"))
    );
}

#[rstest]
#[tokio::test]
async fn test_should_elide_null_preface() {
    let original_req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "first part" }] }],
        "docs": [
            { "content": [{ "text": "i am context" }] },
            { "content": [{ "text": "i am more context" }] }
        ]
    }))
    .unwrap();

    let options = Some(AugmentWithContextOptions {
        preface: Some("".to_string()),
        ..Default::default()
    });

    let modified_req = test_augment_request(original_req.clone(), options).await;
    let last_part = modified_req.messages[0].content.last().unwrap();
    let expected_text = "- [0]: i am context\n- [1]: i am more context\n\n";

    assert_eq!(last_part.text.as_deref(), Some(expected_text));
    assert_eq!(
        last_part.metadata.as_ref().unwrap().get("purpose"),
        Some(&json!("context"))
    );
}
