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

//! # Vertex AI Models
//!
//! This module contains the definitions for models supported by the Vertex AI plugin.

pub mod gemini;
pub mod imagen;
pub mod types;

// Lists of supported models, similar to the TS implementation.
pub const SUPPORTED_GEMINI_MODELS: &[&str] = &[
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17",
];

pub const SUPPORTED_IMAGEN_MODELS: &[&str] = &[
    "imagen-3.0-generate-001",
    "imagen-3.0-fast-generate-001",
    "imagen2",
    "imagen3",
    "imagen3-fast",
];

pub const SUPPORTED_EMBEDDER_MODELS: &[&str] = &[
    "textembedding-gecko@003",
    "text-embedding-004",
    "text-embedding-005",
    "textembedding-gecko-multilingual@001",
    "text-multilingual-embedding-002",
    "multimodalembedding@001",
];
