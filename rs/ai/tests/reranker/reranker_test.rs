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

use genkit_ai::document::Document;
use genkit_ai::reranker::{
    define_reranker, rerank, RankedDocument, RerankerArgument, RerankerParams, RerankerRequest,
    RerankerResponse,
};
use genkit_core::registry::Registry;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{from_value, json};

#[fixture]
fn registry() -> Registry {
    Registry::new()
}

fn ranked_doc_text(doc: &RankedDocument) -> String {
    doc.content
        .iter()
        .filter_map(|p| p.text.as_deref())
        .collect::<Vec<&str>>()
        .join("")
}

#[rstest]
#[tokio::test]
async fn test_reranks_documents_based_on_custom_logic(mut registry: Registry) {
    #[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
    struct TestOptions {
        k: Option<usize>,
    }

    let custom_reranker = define_reranker(
        &mut registry,
        "customReranker",
        |req: RerankerRequest<TestOptions>, _| async move {
            let query_len = req.query.text().len();
            let mut reranked_docs = req
                .documents
                .iter()
                .map(|doc| {
                    let score = (query_len as i32 - doc.text().len() as i32).abs() as f64;
                    RankedDocument {
                        content: doc.content.clone(),
                        metadata: genkit_ai::reranker::RankedDocumentMetadata {
                            score,
                            other_metadata: doc
                                .metadata
                                .as_ref()
                                .map(|m| serde_json::to_value(m).unwrap()),
                        },
                    }
                })
                .collect::<Vec<_>>();

            reranked_docs.sort_by(|a, b| a.score().partial_cmp(&b.score()).unwrap());

            if let Some(k) = req.options.as_ref().and_then(|o| o.k) {
                reranked_docs.truncate(k);
            }

            Ok(RerankerResponse {
                documents: reranked_docs,
            })
        },
    );

    let documents = vec![
        Document::from_text("short", None),
        Document::from_text("a bit longer", None),
        Document::from_text("this is a very long document", None),
    ];

    let query = Document::from_text("medium length", None);
    let reranked_documents = rerank(
        &registry,
        RerankerParams {
            reranker: RerankerArgument::Action(custom_reranker),
            query,
            documents,
            options: Some(TestOptions { k: Some(2) }),
        },
    )
    .await
    .unwrap();

    assert_eq!(reranked_documents.len(), 2);
    assert_eq!(ranked_doc_text(&reranked_documents[0]), "a bit longer");
    assert_eq!(ranked_doc_text(&reranked_documents[1]), "short");
}

#[rstest]
#[tokio::test]
async fn test_handles_missing_options_gracefully(mut registry: Registry) {
    #[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
    struct TestOptions {
        k: Option<usize>,
    }

    let custom_reranker = define_reranker(
        &mut registry,
        "reranker",
        |req: RerankerRequest<TestOptions>, _| async move {
            let reranked_docs = req
                .documents
                .iter()
                .enumerate()
                .map(|(i, doc)| RankedDocument {
                    content: doc.content.clone(),
                    metadata: genkit_ai::reranker::RankedDocumentMetadata {
                        score: 1.0 - (i as f64 * 0.1), // simplified deterministic scoring
                        other_metadata: doc
                            .metadata
                            .as_ref()
                            .map(|m| serde_json::to_value(m).unwrap()),
                    },
                })
                .collect::<Vec<_>>();

            Ok(RerankerResponse {
                documents: reranked_docs,
            })
        },
    );
    let documents = vec![
        Document::from_text("doc1", None),
        Document::from_text("doc2", None),
    ];

    let query = Document::from_text("test query", None);
    let reranked_documents = rerank(
        &registry,
        RerankerParams {
            reranker: RerankerArgument::Action(custom_reranker),
            query,
            documents,
            options: Some(TestOptions { k: Some(2) }),
        },
    )
    .await
    .unwrap();
    assert_eq!(reranked_documents.len(), 2);
    assert!(reranked_documents[0].score() >= 0.0);
}

#[rstest]
#[tokio::test]
async fn test_preserves_document_metadata(mut registry: Registry) {
    let custom_reranker = define_reranker(
        &mut registry,
        "reranker",
        |req: RerankerRequest<()>, _| async move {
            let reranked_docs = req
                .documents
                .iter()
                .enumerate()
                .map(|(i, doc)| RankedDocument {
                    content: doc.content.clone(),
                    metadata: genkit_ai::reranker::RankedDocumentMetadata {
                        score: (2.0 - i as f64),
                        other_metadata: doc
                            .metadata
                            .as_ref()
                            .map(|m| serde_json::to_value(m).unwrap()),
                    },
                })
                .collect::<Vec<_>>();

            Ok(RerankerResponse {
                documents: reranked_docs,
            })
        },
    );
    let documents = vec![
        Document::new(
            vec![],
            Some(from_value(json!({ "originalField": "test1" })).unwrap()),
        ),
        Document::new(
            vec![],
            Some(from_value(json!({ "originalField": "test2" })).unwrap()),
        ),
    ];

    let query = Document::from_text("test query", None);
    let reranked_documents = rerank(
        &registry,
        RerankerParams {
            reranker: RerankerArgument::Action(custom_reranker),
            query,
            documents,
            options: None,
        },
    )
    .await
    .unwrap();
    let meta0 = reranked_documents[0]
        .metadata
        .other_metadata
        .as_ref()
        .unwrap()
        .as_object()
        .unwrap();
    assert_eq!(meta0["originalField"], "test1");
    let meta1 = reranked_documents[1]
        .metadata
        .other_metadata
        .as_ref()
        .unwrap()
        .as_object()
        .unwrap();
    assert_eq!(meta1["originalField"], "test2");
}

#[rstest]
#[tokio::test]
async fn test_handles_errors_thrown_by_the_reranker(mut registry: Registry) {
    let custom_reranker = define_reranker(&mut registry, "reranker", |_, _| async move {
        Err(genkit_core::error::Error::new_internal(
            "Something went wrong during reranking",
        ))
    });
    let documents = vec![
        Document::from_text("doc1", None),
        Document::from_text("doc2", None),
    ];
    let query = Document::from_text("test query", None);

    let result = rerank(
        &registry,
        RerankerParams::<()> {
            reranker: RerankerArgument::Action(custom_reranker),
            query,
            documents,
            options: None,
        },
    )
    .await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(
        err.to_string(),
        "INTERNAL: Something went wrong during reranking"
    );
}
