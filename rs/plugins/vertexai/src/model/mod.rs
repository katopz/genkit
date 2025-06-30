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

// Lists of supported models, similar to the TS implementation.
pub const SUPPORTED_GEMINI_MODELS: &[&str] = &[
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.0-pro",
];

pub const SUPPORTED_IMAGEN_MODELS: &[&str] = &["imagen-3.0-generate"];

pub const SUPPORTED_EMBEDDER_MODELS: &[&str] = &[
    "text-embedding-004",
    "text-multilingual-embedding-002",
    "text-embedding-gecko@003",
    "text-embedding-gecko-multilingual@001",
    "multimodalembedding@001",
];
