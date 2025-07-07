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

use genkit_ai::generate::chunk::GenerateResponseChunk;

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_text_accumulation() {
        let test_chunk: GenerateResponseChunk<()> = GenerateResponseChunk::from_json(
            json!({
                "index": 0,
                "role": "model",
                "content": [{ "text": "new" }]
            }),
            json!({
                "previousChunks": [
                    { "index": 0, "role": "model", "content": [{ "text": "old1" }] },
                    { "index": 0, "role": "model", "content": [{ "text": "old2" }] },
                ]
            }),
        )
        .unwrap();

        assert_eq!(test_chunk.previous_text(), "old1old2");
        assert_eq!(test_chunk.accumulated_text(), "old1old2new");
    }
}
