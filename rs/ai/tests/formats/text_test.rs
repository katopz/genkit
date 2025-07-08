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

use genkit_ai::{
    document::Part,
    formats::text::text_formatter,
    generate::chunk::{GenerateResponseChunk, GenerateResponseChunkOptions},
    message::{Message, MessageData, Role},
    model::GenerateResponseChunkData,
};
use rstest::*;
use serde::Deserialize;
use serde_json::{from_value, json, Value};

#[cfg(test)]
mod text_format_tests {
    use super::*;

    //-////////////////////////////////////////////////////////////////////////-
    // Streaming Parser Tests
    //-////////////////////////////////////////////////////////////////////////-

    #[derive(Debug, Deserialize)]
    struct StreamingTestCase {
        desc: String,
        chunks: Vec<StreamingTestChunk>,
    }

    #[derive(Debug, Deserialize)]
    struct StreamingTestChunk {
        text: String,
        want: String,
    }

    #[rstest]
    #[case(from_value(json!({
        "desc": "emits text chunks as they arrive",
        "chunks": [
            { "text": "Hello", "want": "Hello" },
            { "text": " world", "want": " world" },
        ],
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "handles empty chunks",
        "chunks": [
            { "text": "", "want": "" },
        ],
    })).unwrap())]
    fn test_streaming_parser(#[case] case: StreamingTestCase) {
        let formatter = text_formatter();
        let handler = (formatter.handler)(None);

        for (i, chunk_def) in case.chunks.iter().enumerate() {
            let new_chunk_data = GenerateResponseChunkData {
                index: 0,
                role: Some(Role::Model),
                content: vec![Part::text(&chunk_def.text)],
                ..Default::default()
            };

            // For the text formatter, previous chunks are not needed,
            // so we can pass an empty Vec.
            let response_chunk =
                GenerateResponseChunk::new(new_chunk_data, GenerateResponseChunkOptions::default());

            let result = handler.parse_chunk(&response_chunk).unwrap();
            assert_eq!(
                result,
                Value::String(chunk_def.want.clone()),
                "Test: '{}', Chunk: {}",
                case.desc,
                i
            );
        }
    }

    //-////////////////////////////////////////////////////////////////////////-
    // Message Parser Tests
    //-////////////////////////////////////////////////////////////////////////-

    #[derive(Debug, Deserialize)]
    struct MessageTestCase {
        desc: String,
        message: MessageData,
        want: String,
    }

    #[rstest]
    #[case(from_value(json!({
        "desc": "parses complete text response",
        "message": {
            "role": "model",
            "content": [{ "text": "Hello world" }],
        },
        "want": "Hello world",
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "handles empty response",
        "message": {
            "role": "model",
            "content": [{ "text": "" }],
        },
        "want": "",
    })).unwrap())]
    fn test_message_parser(#[case] case: MessageTestCase) {
        let formatter = text_formatter();
        let handler = (formatter.handler)(None);
        let message = Message::new(case.message, None);
        let result = handler.parse_message(&message);
        assert_eq!(result, Value::String(case.want), "Test: '{}'", case.desc);
    }
}
