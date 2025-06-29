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

//! # Streaming Generation Chunk Tests

use genkit_ai::document::Part;
use genkit_ai::generate::chunk::{GenerateResponseChunk, GenerateResponseChunkOptions};
use genkit_ai::message::Role;
use genkit_ai::model::GenerateResponseChunkData;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_text_accumulation() {
        let options = GenerateResponseChunkOptions {
            previous_chunks: vec![
                GenerateResponseChunkData {
                    index: 0,
                    role: Some(Role::Model),
                    content: vec![Part {
                        text: Some("old1".to_string()),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                GenerateResponseChunkData {
                    index: 0,
                    role: Some(Role::Model),
                    content: vec![Part {
                        text: Some("old2".to_string()),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            role: Some(Role::Model),
            index: Some(0),
        };

        let current_chunk_data = GenerateResponseChunkData {
            index: 0,
            role: Some(Role::Model),
            content: vec![Part {
                text: Some("new".to_string()),
                ..Default::default()
            }],
            ..Default::default()
        };

        let test_chunk: GenerateResponseChunk =
            GenerateResponseChunk::new(current_chunk_data, options);

        assert_eq!(test_chunk.previous_text(), "old1old2");
        assert_eq!(test_chunk.accumulated_text(), "old1old2new");
    }
}
