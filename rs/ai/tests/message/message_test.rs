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

use genkit_ai::message::{MessageData, Role};
use genkit_ai::Part;
use rstest::rstest;
use serde_json::{json, Value};
use std::collections::HashMap;

#[cfg(test)]
mod message_tests {
    use super::*;

    #[rstest]
    #[case(
        "convert string to user message",
        json!("i am a user message"),
        MessageData {
            role: Role::User,
            content: vec![Part::text("i am a user message")],
            ..Default::default()
        }
    )]
    #[case(
        "convert string content to Part[] content",
        json!({
          "role": "system",
          "content": "i am a system message",
          "metadata": { "extra": true },
        }),
        MessageData {
            role: Role::System,
            content: vec![Part::text("i am a system message")],
            metadata: Some({
                let mut map = HashMap::new();
                map.insert("extra".to_string(), Value::Bool(true));
                map
            }),
        }
    )]
    #[case(
        "leave valid MessageData alone",
        json!({ "role": "model", "content": [{ "text": "i am a model message" }] }),
        MessageData {
            role: Role::Model,
            content: vec![Part::text("i am a model message")],
            ..Default::default()
        }
    )]
    fn test_from_value(#[case] desc: &str, #[case] input: Value, #[case] want: MessageData) {
        let result = MessageData::from_value(input).unwrap();
        assert_eq!(result, want, "Test failed: {}", desc);
    }
}
