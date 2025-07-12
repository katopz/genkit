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
use genkit_ai::embedder::{define_embedder, EmbedRequest, EmbedResponse, Embedding};
use genkit_ai::generate::{generate, GenerateOptions};
use genkit_ai::model::{
    define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
};
use genkit_ai::retriever::{index, retrieve, CommonRetrieverOptions};
use genkit_core::plugin::Plugin;
use genkit_core::registry::Registry;
use genkit_plugins_dev_local_vectorstore::{
    local_indexer_ref, local_retriever_ref, DevLocalVectorStorePlugin, LocalVectorStoreConfig,
};
use std::sync::Arc;

// A simple mock embedder that returns predefined vectors for specific texts.
fn mock_embedder() -> genkit_ai::embedder::EmbedderAction {
    let registry = Registry::new();
    define_embedder(
        &registry,
        "mock-embedder",
        |req: EmbedRequest, _| async move {
            let embeddings = req
                .input
                .into_iter()
                .map(|doc| {
                    let text = doc.text();
                    let embedding_vec = match text.as_str() {
                        "How old is Bob?" => vec![1.0, 0.1, 0.2, 0.3],
                        "Bob is 42 years old." => vec![1.0, 0.1, 0.2, 0.3], // Same as question for max similarity
                        "Bob lives on the moon." => vec![0.1, 1.0, 0.2, 0.3],
                        "Bob likes bananas." => vec![0.2, 0.1, 1.0, 0.3],
                        "Bob has 11 cats." => vec![0.3, 0.2, 0.1, 1.0],
                        _ => vec![0.0; 4], // Default for any other text
                    };
                    Embedding {
                        embedding: embedding_vec,
                        metadata: None,
                    }
                })
                .collect();
            Ok(EmbedResponse { embeddings })
        },
    )
}

// A simple mock model that just returns the content of the first document it receives.
fn mock_model() -> genkit_ai::model::ModelAction {
    let registry = Registry::new();
    define_model(
        &registry,
        genkit_ai::model::DefineModelOptions {
            name: "mock-model".to_string(),
            label: Some("Mock Context Model".to_string()),
            supports: Some(genkit_ai::model::ModelInfoSupports {
                context: Some(true), // We need to signal that this model supports docs
                ..Default::default()
            }),
            ..Default::default()
        },
        |req: GenerateRequest, _| async move {
            let context_text = req
                .docs
                .unwrap_or_default()
                .first()
                .map(|d| d.text().clone())
                .unwrap_or_else(|| "No context provided.".to_string());

            let response_message = genkit_ai::message::MessageData {
                role: genkit_ai::message::Role::Model,
                content: vec![genkit_ai::document::Part {
                    text: Some(context_text),
                    ..Default::default()
                }],
                metadata: None,
            };
            Ok(GenerateResponseData {
                candidates: vec![CandidateData {
                    index: 0,
                    message: response_message,
                    finish_reason: Some(FinishReason::Stop),
                    finish_message: None,
                }],
                ..Default::default()
            })
        },
    )
}

#[tokio::test]
async fn test_dev_local_vectorstore_e2e() {
    // 1. Set up registry and register mock components.
    let registry = Registry::new();

    let embedder_action = mock_embedder();
    registry
        .register_action("mock-embedder", embedder_action)
        .unwrap();

    let model_action = mock_model();
    registry
        .register_action("mock-model", model_action)
        .unwrap();

    // 2. Configure and initialize the vector store plugin.
    // Using `data_path: None` will cause it to write to a temporary file.
    let vectorstore_plugin = DevLocalVectorStorePlugin::new(vec![LocalVectorStoreConfig {
        index_name: "BobFacts".to_string(),
        embedder: "mock-embedder".to_string(),
        data_path: None,
    }]);

    vectorstore_plugin.initialize(&registry).await.unwrap();

    let arc_registry = Arc::new(registry);

    // 3. Index documents.
    let indexer = local_indexer_ref("BobFacts");
    let documents_to_index = vec![
        Document::from_text("Bob lives on the moon.".to_string(), None),
        Document::from_text("Bob is 42 years old.".to_string(), None),
        Document::from_text("Bob likes bananas.".to_string(), None),
        Document::from_text("Bob has 11 cats.".to_string(), None),
    ];

    index(
        &arc_registry,
        genkit_ai::retriever::IndexerParams {
            indexer: genkit_ai::retriever::IndexerArgument::<()>::Name(indexer.name),
            documents: documents_to_index,
            options: None,
        },
    )
    .await
    .unwrap();

    // 4. Retrieve the most relevant document.
    let question = "How old is Bob?";
    let retriever = local_retriever_ref("BobFacts");
    let retrieve_options = CommonRetrieverOptions { k: Some(1) }; // Ask for the single most relevant doc.

    let retrieved_docs = retrieve(
        &arc_registry,
        genkit_ai::retriever::RetrieverParams {
            retriever: genkit_ai::retriever::RetrieverArgument::<CommonRetrieverOptions>::Name(
                retriever.name,
            ),
            query: Document::from_text(question.to_string(), None),
            options: Some(retrieve_options),
        },
    )
    .await
    .unwrap();

    assert_eq!(retrieved_docs.len(), 1);
    assert_eq!(retrieved_docs[0].text(), "Bob is 42 years old.");

    // 5. Generate a response using the retrieved document as context.
    let generate_result = generate(
        &arc_registry,
        GenerateOptions::<()> {
            model: Some(genkit_ai::model::Model::Name("mock-model".to_string())),
            prompt: Some(vec![genkit_ai::document::Part {
                text: Some(format!(
                    "Use the provided context to answer this query: {}",
                    question
                )),
                ..Default::default()
            }]),
            docs: Some(retrieved_docs),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // 6. Assert the final output.
    // Our mock model is designed to just return the text of the first document.
    assert_eq!(generate_result.text().unwrap(), "Bob is 42 years old.");
}
