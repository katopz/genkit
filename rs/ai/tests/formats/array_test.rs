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
    formats::array::array_formatter,
    generate::chunk::{GenerateResponseChunk, GenerateResponseChunkOptions},
    message::{Message, MessageData, Role},
    model::GenerateResponseChunkData,
};
use serde_json::{json, Value};

#[cfg(test)]
mod tests {
    use super::*;

    struct StreamingTestCase<'a> {
        desc: &'a str,
        chunks: Vec<StreamingTestStep<'a>>,
    }

    struct StreamingTestStep<'a> {
        text: &'a str,
        want: Value,
    }

    #[test]
    fn test_streaming_parser() {
        let test_cases = vec![
            StreamingTestCase {
                desc: "emits complete array items as they arrive",
                chunks: vec![
                    StreamingTestStep {
                        text: r#"[{"id": 1,"#,
                        want: json!([]),
                    },
                    StreamingTestStep {
                        text: r#""name": "first"}"#,
                        want: json!([{"id": 1, "name": "first"}]),
                    },
                    StreamingTestStep {
                        text: r#", {"id": 2, "name": "second"}]"#,
                        want: json!([{"id": 2, "name": "second"}]),
                    },
                ],
            },
            StreamingTestCase {
                desc: "handles single item arrays",
                chunks: vec![StreamingTestStep {
                    text: r#"[{"id": 1, "name": "single"}]"#,
                    want: json!([{"id": 1, "name": "single"}]),
                }],
            },
            StreamingTestCase {
                desc: "handles preamble with code fence",
                chunks: vec![
                    StreamingTestStep {
                        text: "Here is the array you requested:\n\n```json\n[",
                        want: json!([]),
                    },
                    StreamingTestStep {
                        text: r#"{"id": 1, "name": "item"}]`"#,
                        want: json!([{"id": 1, "name": "item"}]),
                    },
                ],
            },
        ];

        let formatter = array_formatter();
        let handler = (formatter.handler)(None);

        for case in test_cases {
            let mut chunks: Vec<GenerateResponseChunkData> = Vec::new();
            for (i, step) in case.chunks.iter().enumerate() {
                let new_chunk_data = GenerateResponseChunkData {
                    index: 0,
                    role: Some(Role::Model),
                    content: vec![Part::text(step.text)],
                    ..Default::default()
                };

                let chunk = GenerateResponseChunk::new(
                    new_chunk_data.clone(),
                    GenerateResponseChunkOptions {
                        previous_chunks: chunks.clone(),
                        ..Default::default()
                    },
                );
                let result = handler.parse_chunk(&chunk).unwrap();
                assert_eq!(result, step.want, "failed test ({}): step {}", case.desc, i);
                chunks.push(new_chunk_data);
            }
        }
    }

    struct MessageTestCase {
        desc: &'static str,
        message: MessageData,
        want: Value,
    }

    #[test]
    fn test_message_parser() {
        let test_cases = vec![
            MessageTestCase {
                desc: "parses complete array response",
                message: MessageData {
                    role: Role::Model,
                    content: vec![Part::text(r#"[{"id": 1, "name": "test"}]"#)],
                    ..Default::default()
                },
                want: json!([{"id": 1, "name": "test"}]),
            },
            MessageTestCase {
                desc: "parses empty array",
                message: MessageData {
                    role: Role::Model,
                    content: vec![Part::text(r#"[]"#)],
                    ..Default::default()
                },
                want: json!([]),
            },
            MessageTestCase {
                desc: "parses array with preamble and code fence",
                message: MessageData {
                    role: Role::Model,
                    content: vec![Part::text(
                        "Here is the array:\n\n```json\n[{\"id\": 1}]\n```",
                    )],
                    ..Default::default()
                },
                want: json!([{"id": 1}]),
            },
        ];

        let formatter = array_formatter();
        let handler = (formatter.handler)(None);

        for case in test_cases {
            let message = Message::new(case.message, None);
            let result = handler.parse_message(&message);
            assert_eq!(result, case.want, "failed test: {}", case.desc);
        }
    }
}
