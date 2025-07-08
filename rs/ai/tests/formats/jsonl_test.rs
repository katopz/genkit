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
    formats::jsonl::jsonl_formatter,
    generate::chunk::{GenerateResponseChunk, GenerateResponseChunkOptions},
    message::{Message, MessageData, Role},
    model::GenerateResponseChunkData,
};
use rstest::*;
use serde::Deserialize;
use serde_json::{from_value, json, Value};

#[cfg(test)]
mod jsonl_format_tests {
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
        want: Value,
    }

    #[rstest]
    #[case(from_value(json!({
        "desc": "emits complete JSON objects as they arrive",
        "chunks": [
            {
                "text": "{\"id\": 1, \"name\": \"first\"}\n",
                "want": [{"id": 1, "name": "first"}],
            },
            {
                "text": "{\"id\": 2, \"name\": \"second\"}\n{\"id\": 3",
                "want": [{"id": 2, "name": "second"}],
            },
            {
                "text": ", \"name\": \"third\"}\n",
                "want": [{"id": 3, "name": "third"}],
            },
        ],
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "handles single object",
        "chunks": [
            {
                "text": "{\"id\": 1, \"name\": \"single\"}\n",
                "want": [{"id": 1, "name": "single"}],
            },
        ],
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "handles preamble with code fence",
        "chunks": [
            {
                "text": "Here are the objects:\n\n```\n",
                "want": [],
            },
            {
                "text": "{\"id\": 1, \"name\": \"item\"}\n```",
                "want": [{"id": 1, "name": "item"}],
            },
        ],
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "ignores non-object lines",
        "chunks": [
            {
                "text": "First object:\n{\"id\": 1}\nSecond object:\n{\"id\": 2}\n",
                "want": [{"id": 1}, {"id": 2}],
            },
        ],
    })).unwrap())]
    fn test_streaming_parser(#[case] case: StreamingTestCase) {
        let formatter = jsonl_formatter();
        let handler = (formatter.handler)(None);
        let mut chunks: Vec<GenerateResponseChunkData> = Vec::new();

        for (i, chunk_def) in case.chunks.iter().enumerate() {
            let new_chunk_data = GenerateResponseChunkData {
                index: 0,
                role: Some(Role::Model),
                content: vec![Part::text(&chunk_def.text)],
                ..Default::default()
            };

            let response_chunk = GenerateResponseChunk::new(
                new_chunk_data.clone(),
                GenerateResponseChunkOptions {
                    previous_chunks: chunks.clone(),
                    ..Default::default()
                },
            );
            chunks.push(new_chunk_data);

            let result = handler.parse_chunk(&response_chunk).unwrap();
            assert_eq!(
                result, chunk_def.want,
                "Test: '{}', Chunk: {}",
                case.desc, i
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
        want: Value,
    }

    #[rstest]
    #[case(from_value(json!({
        "desc": "parses complete JSONL response",
        "message": {
            "role": "model",
            "content": [{ "text": "{\"id\": 1, \"name\": \"test\"}\n{\"id\": 2}\n" }],
        },
        "want": [{ "id": 1, "name": "test" }, { "id": 2 }],
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "handles empty response",
        "message": {
            "role": "model",
            "content": [{ "text": "" }],
        },
        "want": [],
    })).unwrap())]
    #[case(from_value(json!({
        "desc": "parses JSONL with preamble and code fence",
        "message": {
            "role": "model",
            "content": [
                {
                    "text": "Here are the objects:\n\n```\n{\"id\": 1}\n{\"id\": 2}\n```",
                },
            ],
        },
        "want": [{ "id": 1 }, { "id": 2 }],
    })).unwrap())]
    fn test_message_parser(#[case] case: MessageTestCase) {
        let formatter = jsonl_formatter();
        let handler = (formatter.handler)(None);
        let message = Message::new(case.message, None);
        let result = handler.parse_message(&message);
        assert_eq!(result, case.want, "Test: '{}'", case.desc);
    }
}
