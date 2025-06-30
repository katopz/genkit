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

//! # Genkit Development Local Vector Store Plugin
//!
//! This crate provides a simple, file-based vector store for local development.

use async_trait::async_trait;
use genkit_ai::{
    document::Document,
    embedder::EmbedRequest,
    retriever::{
        define_indexer, define_retriever, indexer_ref, retriever_ref, CommonRetrieverOptions,
        IndexerInfo, IndexerRef, RetrieverInfo, RetrieverRef, RetrieverRequest,
    },
    EmbedResponse,
};
use genkit_core::{error::Error, plugin::Plugin, registry::Registry, Result};
use semanticsimilarity_rs::cosine_similarity;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path, sync::Arc};

const DEV_LOCAL_VECTORSTORE_PREFIX: &str = "devLocalVectorstore";

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DbValue {
    doc: Document,
    embedding: Vec<f32>,
}

type DbContents = HashMap<String, DbValue>;

fn load_filestore(path: &str) -> Result<DbContents> {
    if !Path::new(path).exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(path).map_err(|e| Error::new_internal(e.to_string()))?;
    serde_json::from_str(&data).map_err(|e| Error::new_internal(e.to_string()))
}

fn save_filestore(path: &str, contents: &DbContents) -> Result<()> {
    let data =
        serde_json::to_string_pretty(contents).map_err(|e| Error::new_internal(e.to_string()))?;
    fs::write(path, data).map_err(|e| Error::new_internal(e.to_string()))?;
    Ok(())
}

fn add_document(embedding: Vec<f32>, doc: Document, contents: &mut DbContents) {
    let digest = md5::compute(serde_json::to_string(&doc).unwrap_or_default());
    let id = format!("{:x}", digest);
    contents.entry(id).or_insert(DbValue { doc, embedding });
}

fn get_closest_documents(query_embedding: &[f32], db: &DbContents, k: usize) -> Vec<Document> {
    let query_embedding_f64: Vec<f64> = query_embedding.iter().map(|&x| x as f64).collect();
    let mut scored_docs: Vec<(f64, Document)> = db
        .values()
        .map(|value| {
            let value_embedding_f64: Vec<f64> = value.embedding.iter().map(|&x| x as f64).collect();
            let score = cosine_similarity(&value_embedding_f64, &query_embedding_f64, false);
            (score, value.doc.clone())
        })
        .collect();
    scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored_docs
        .into_iter()
        .take(k)
        .map(|(_, doc)| doc)
        .collect()
}

#[derive(Debug, Deserialize)]
pub struct LocalVectorStoreConfig {
    /// The name of the index.
    pub index_name: String,
    /// The path to the directory where the vector store data will be saved.
    pub data_path: Option<String>,
    /// The name of the embedder to use.
    pub embedder: String,
}

/// The development-only local vector store plugin.
#[derive(Debug)]
pub struct DevLocalVectorStorePlugin {
    configs: Vec<LocalVectorStoreConfig>,
}

impl DevLocalVectorStorePlugin {
    pub fn new(configs: Vec<LocalVectorStoreConfig>) -> Self {
        Self { configs }
    }
}

#[async_trait]
impl Plugin for DevLocalVectorStorePlugin {
    fn name(&self) -> &'static str {
        DEV_LOCAL_VECTORSTORE_PREFIX
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        for config in &self.configs {
            let data_path = config.data_path.clone().unwrap_or_else(|| {
                let mut path = std::env::temp_dir();
                let filename = format!("__db_{}.json", config.index_name);
                path.push(filename);
                path.to_str().unwrap().to_string()
            });

            let embedder_name = config.embedder.clone();
            let embedder_action = registry
                .lookup_action(&format!("/embedder/{}", embedder_name))
                .await
                .ok_or_else(|| {
                    Error::new_internal(format!("Embedder '{}' not found", embedder_name))
                })?;

            let aname = format!("{}/{}", DEV_LOCAL_VECTORSTORE_PREFIX, config.index_name);
            let action_name: &'static str = Box::leak(aname.into_boxed_str());

            let retriever_data_path = data_path.clone();
            let retriever_embedder_action = embedder_action.clone();
            let retriever_action = define_retriever(
                action_name,
                move |req: RetrieverRequest<CommonRetrieverOptions>, _| {
                    let path = retriever_data_path.clone();
                    let embedder = retriever_embedder_action.clone();
                    async move {
                        let embed_req = serde_json::to_value(EmbedRequest {
                            input: vec![req.query],
                            options: None::<()>,
                        })
                        .map_err(|e| Error::new_internal(e.to_string()))?;
                        let mut embed_resp_value = embedder.run_http_json(embed_req, None).await?;
                        if let Some(result) = embed_resp_value.get_mut("result") {
                            embed_resp_value = result.take();
                        }
                        let embed_resp: EmbedResponse = serde_json::from_value(embed_resp_value)
                            .map_err(|e| Error::new_internal(e.to_string()))?;

                        if embed_resp.embeddings.is_empty() {
                            return Err(Error::new_internal("Embedder returned no embeddings"));
                        }
                        let query_embedding = &embed_resp.embeddings[0].embedding;

                        let db = load_filestore(&path)?;
                        let k = req.options.as_ref().and_then(|o| o.k).unwrap_or(3);
                        let docs = get_closest_documents(query_embedding, &db, k as usize);

                        Ok(genkit_ai::retriever::RetrieverResponse { documents: docs })
                    }
                },
            );
            registry.register_action(Arc::new(retriever_action))?;

            let indexer_data_path = data_path.clone();
            let indexer_embedder_action = embedder_action.clone();
            let indexer_action = define_indexer(
                action_name,
                move |req: genkit_ai::retriever::IndexerRequest<()>, _| {
                    let path = indexer_data_path.clone();
                    let embedder = indexer_embedder_action.clone();
                    async move {
                        let mut db = load_filestore(&path)?;
                        for doc in req.documents {
                            let embed_req = serde_json::to_value(EmbedRequest {
                                input: vec![doc.clone()],
                                options: None::<()>,
                            })
                            .map_err(|e| Error::new_internal(e.to_string()))?;
                            let mut embed_resp_value =
                                embedder.run_http_json(embed_req, None).await?;
                            if let Some(result) = embed_resp_value.get_mut("result") {
                                embed_resp_value = result.take();
                            }
                            let embed_resp: EmbedResponse =
                                serde_json::from_value(embed_resp_value)
                                    .map_err(|e| Error::new_internal(e.to_string()))?;

                            if embed_resp.embeddings.is_empty() {
                                continue;
                            }
                            let embedding_docs =
                                doc.get_embedding_documents(&embed_resp.embeddings);
                            for (i, chunked_doc) in embedding_docs.into_iter().enumerate() {
                                if let Some(embedding) = embed_resp.embeddings.get(i) {
                                    add_document(embedding.embedding.clone(), chunked_doc, &mut db);
                                }
                            }
                        }
                        save_filestore(&path, &db)
                    }
                },
            );
            registry.register_action(Arc::new(indexer_action))?;
        }
        Ok(())
    }
}

/// Creates a new local vector store plugin.
pub fn local_vector_store(configs: Vec<LocalVectorStoreConfig>) -> Arc<dyn Plugin> {
    Arc::new(DevLocalVectorStorePlugin::new(configs))
}

/// A reference to the local retriever.
pub fn local_retriever_ref(index_name: &str) -> RetrieverRef<CommonRetrieverOptions> {
    retriever_ref(
        format!("{}/{}", DEV_LOCAL_VECTORSTORE_PREFIX, index_name).as_str(),
        Some(RetrieverInfo {
            label: Some(format!("Local file-based Retriever - {}", index_name)),
        }),
    )
}

/// A reference to the local indexer.
pub fn local_indexer_ref(index_name: &str) -> IndexerRef<()> {
    indexer_ref(
        format!("{}/{}", DEV_LOCAL_VECTORSTORE_PREFIX, index_name).as_str(),
        Some(IndexerInfo {
            label: Some(format!("Local file-based Indexer - {}", index_name)),
        }),
    )
}
