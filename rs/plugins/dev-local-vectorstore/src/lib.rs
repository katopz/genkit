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
use genkit_ai::retriever::{
    define_indexer, define_retriever, indexer_ref, retriever_ref, IndexerRef, RetrieverRef,
};
use genkit_core::{plugin::Plugin, registry::Registry, Result};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct LocalVectorStoreConfig {
    /// The path to the directory where the vector store data will be saved.
    pub data_path: Option<String>,
}

/// The development-only local vector store plugin.
#[derive(Debug)]
pub struct DevLocalVectorStorePlugin {
    config: LocalVectorStoreConfig,
}

impl DevLocalVectorStorePlugin {
    pub fn new(config: LocalVectorStoreConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Plugin for DevLocalVectorStorePlugin {
    fn name(&self) -> &'static str {
        "dev-local-vectorstore"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let data_path = self.config.data_path.clone().unwrap_or_else(|| {
            let mut path = std::env::temp_dir();
            path.push("genkit_dev_store.json");
            path.to_str().unwrap().to_string()
        });

        let retriever_data_path = data_path.clone();
        let retriever_action = define_retriever(
            "local",
            move |_req: genkit_ai::retriever::RetrieverRequest<()>, _| {
                let path = retriever_data_path.clone();
                async move {
                    // TODO: Implement file reading and similarity search logic.
                    println!("Retrieving from local store at: {}", path);
                    Ok(genkit_ai::retriever::RetrieverResponse { documents: vec![] })
                }
            },
        );
        registry.register_action(retriever_action.0)?;

        let indexer_action = define_indexer(
            "local",
            move |_req: genkit_ai::retriever::IndexerRequest<()>, _| {
                let path = data_path.clone();
                async move {
                    // TODO: Implement file writing logic.
                    println!("Indexing to local store at: {}", path);
                    Ok(())
                }
            },
        );
        registry.register_action(indexer_action.0)?;

        Ok(())
    }
}

/// Creates a new local vector store plugin.
pub fn local_vector_store(config: LocalVectorStoreConfig) -> Arc<dyn Plugin> {
    Arc::new(DevLocalVectorStorePlugin::new(config))
}

/// A reference to the local retriever.
pub fn local_retriever_ref() -> RetrieverRef {
    retriever_ref("local")
}

/// A reference to the local indexer.
pub fn local_indexer_ref() -> IndexerRef {
    indexer_ref("local")
}
