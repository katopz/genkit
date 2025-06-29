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

//! # JSON Extraction Tests

use genkit_ai::extract::{extract_items, extract_json, parse_partial_json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

#[cfg(test)]
mod test {
    use super::*;

    //-////////////////////////////////////////////////////////////////////////-
    // extract_json tests
    //-////////////////////////////////////////////////////////////////////////-

    #[derive(Deserialize, Debug, PartialEq)]
    struct SimpleObject {
        a: i32,
    }

    #[test]
    fn test_extract_json_simple_object() {
        let text = "prefix{\"a\":1}suffix";
        let result: Option<SimpleObject> = extract_json(text).unwrap();
        assert_eq!(result, Some(SimpleObject { a: 1 }));
    }

    #[test]
    fn test_extract_json_simple_array() {
        let text = "prefix[1,2,3]suffix";
        let result: Option<Vec<i32>> = extract_json(text).unwrap();
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_extract_json_nested() {
        #[derive(Deserialize, Debug, PartialEq)]
        struct Nested {
            b: Vec<i32>,
        }
        #[derive(Deserialize, Debug, PartialEq)]
        struct Outer {
            a: Nested,
        }
        let text = "text{\"a\":{\"b\":[1,2]}}more";
        let result: Option<Outer> = extract_json(text).unwrap();
        assert_eq!(
            result,
            Some(Outer {
                a: Nested { b: vec![1, 2] }
            })
        );
    }

    #[test]
    fn test_extract_json_with_braces_in_string() {
        let text = "{\"text\": \"not {a} json\"}";
        let result: Option<HashMap<String, String>> = extract_json(text).unwrap();
        assert_eq!(result.unwrap().get("text").unwrap(), "not {a} json");
    }

    #[test]
    fn test_extract_json_no_json() {
        let text = "not json at all";
        let result: Option<Value> = extract_json(text).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_json_malformed_throws() {
        // The Rust implementation returns an error, which we assert on.
        let text = "prefix{\"a\": 1,}suffix";
        let result: genkit_core::error::Result<Option<Value>> = extract_json(text);
        assert!(result.is_err());
    }

    //-////////////////////////////////////////////////////////////////////////-
    // parse_partial_json tests
    //-////////////////////////////////////////////////////////////////////////-

    #[test]
    fn test_parse_partial_json_complete_object() {
        let input = "{\"a\":1,\"b\":2}";
        let result: genkit_core::error::Result<Value> = parse_partial_json(input);
        assert_eq!(result.unwrap(), json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_parse_partial_json_incomplete_object() {
        // json5 handles trailing commas, so this should parse.
        let input = "{\"a\":1,";
        let result: genkit_core::error::Result<Value> = parse_partial_json(input);
        assert_eq!(result.unwrap(), json!({"a": 1}));
    }

    #[test]
    fn test_parse_partial_json_incomplete_array() {
        let input = "[1, 2, 3,";
        let result: genkit_core::error::Result<Value> = parse_partial_json(input);
        assert_eq!(result.unwrap(), json!([1, 2, 3]));
    }

    #[test]
    fn test_parse_partial_json_severely_malformed() {
        let input = "{\"a\":{\"b\":1,\"c\":]}}";
        let result: genkit_core::error::Result<Value> = parse_partial_json(input);
        assert!(result.is_err());
    }

    //-////////////////////////////////////////////////////////////////////////-
    // extract_items tests
    //-////////////////////////////////////////////////////////////////////////-

    #[test]
    fn test_extract_items_simple_array_in_chunks() {
        let mut text = String::new();
        let mut cursor = 0;

        // Chunk 1
        text.push_str("[{\"a\": 1},");
        let result1 = extract_items(&text, cursor);
        assert_eq!(result1.items, vec![json!({"a": 1})]);
        cursor = result1.cursor;

        // Chunk 2
        text.push_str(" {\"b\": 2}");
        let result2 = extract_items(&text, cursor);
        assert_eq!(result2.items, vec![json!({"b": 2})]);
        cursor = result2.cursor;

        // Chunk 3
        text.push(']');
        let result3 = extract_items(&text, cursor);
        assert!(result3.items.is_empty());
    }

    #[test]
    fn test_extract_items_nested_objects() {
        let mut text = String::new();
        let mut cursor = 0;

        // Chunk 1
        text.push_str("[{\"outer\": {\"inner\": \"value\"}},");
        let result1 = extract_items(&text, cursor);
        assert_eq!(result1.items, vec![json!({"outer": {"inner": "value"}})]);
        cursor = result1.cursor;

        // Chunk 2
        text.push_str("{\"next\": true}]");
        let result2 = extract_items(&text, cursor);
        assert_eq!(result2.items, vec![json!({"next": true})]);
    }

    #[test]
    fn test_extract_items_ignores_content_before_array() {
        let text =
            "Here is an array:\n```json\n\n[{\"a\": 1}, {\"b\": 2}]\n```\nDid you like my array?";
        let result = extract_items(text, 0);
        assert_eq!(result.items, vec![json!({"a": 1}), json!({"b": 2})]);
    }
}
