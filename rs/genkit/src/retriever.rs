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

//! # Genkit Retriever
//!
//! This module re-exports the retriever and indexer functionality from the
//! `genkit-ai` crate.

pub use genkit_ai::document::Document;
pub use genkit_ai::retriever::{
    define_indexer, define_retriever, index, indexer_ref, retrieve, retriever_ref, IndexerAction,
    IndexerArgument, IndexerInfo, IndexerParams, IndexerRef, RetrieverAction, RetrieverArgument,
    RetrieverInfo, RetrieverParams, RetrieverRef,
};
