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
    formats::r#enum::enum_formatter,
    message::{Message, MessageData},
};
use serde::Deserialize;
use serde_json::{from_value, json, Value};

#[cfg(test)]
mod enum_format_tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MessageTestCase {
        desc: String,
        message: MessageData,
        want: Value,
    }

    #[test]
    fn test_message_parser() {
        let tests: Vec<MessageTestCase> = from_value(json!([
            {
                "desc": "parses simple enum value",
                "message": {
                    "role": "model",
                    "content": [{ "text": "VALUE1" }]
                },
                "want": "VALUE1"
            },
            {
                "desc": "trims whitespace",
                "message": {
                    "role": "model",
                    "content": [{ "text": "  VALUE2\n" }]
                },
                "want": "VALUE2"
            }
        ]))
        .unwrap();

        let formatter = enum_formatter();
        let handler = (formatter.handler)(None);

        for t in tests {
            let message = Message::new(t.message, None);
            let result = handler.parse_message(&message);
            assert_eq!(result, t.want, "Test: '{}'", t.desc);
        }
    }
}
