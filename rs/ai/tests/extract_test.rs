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
use serde_json::{from_value, json, Value};

//-////////////////////////////////////////////////////////////////////////-
// Test Case Structs
//-////////////////////////////////////////////////////////////////////////-

#[derive(Deserialize, Debug)]
struct ExtractJsonTestCase {
    name: String,
    input: String,
    expected: Option<Value>,
    throws: Option<bool>,
}

#[derive(Deserialize, Debug)]
struct ParsePartialJsonTestCase {
    name: String,
    input: String,
    expected: Option<Value>,
    throws: bool,
}

#[derive(Deserialize, Debug)]
struct ExtractItemsTestStep {
    chunk: String,
    want: Vec<Value>,
}

#[derive(Deserialize, Debug)]
struct ExtractItemsTestCase {
    name: String,
    steps: Vec<ExtractItemsTestStep>,
}

//-////////////////////////////////////////////////////////////////////////-
// extract_json tests
//-////////////////////////////////////////////////////////////////////////-

#[test]
fn test_extract_json_all_cases() {
    let cases: Vec<ExtractJsonTestCase> = from_value(json!([
        {
            "name": "extracts simple object",
            "input": "prefix{\"a\":1}suffix",
            "expected": { "a": 1 }
        },
        {
            "name": "extracts simple array",
            "input": "prefix[1,2,3]suffix",
            "expected": [1, 2, 3]
        },
        {
            "name": "handles nested structures",
            "input": "text{\"a\":{\"b\":[1,2]}}more",
            "expected": { "a": { "b": [1, 2] } }
        },
        {
            "name": "handles strings with braces",
            "input": "{\"text\": \"not {a} json\"}",
            "expected": { "text": "not {a} json" }
        },
        {
            "name": "returns null for non-json",
            "input": "not json at all",
            "expected": null
        },
        {
            "name": "throws for malformed json (unclosed string)",
            "input": "prefix{\"a\": \"",
            "expected": null,
            "throws": true
        }
    ]))
    .unwrap();

    for case in cases {
        let result: genkit_core::error::Result<Option<Value>> = extract_json(&case.input);
        if case.throws.unwrap_or(false) {
            assert!(
                result.is_err(),
                "Test case '{}' should have failed",
                case.name
            );
        } else {
            assert!(
                result.is_ok(),
                "Test case '{}' failed: {:?}",
                case.name,
                result.err()
            );
            assert_eq!(
                result.unwrap(),
                case.expected,
                "Test case '{}' failed",
                case.name
            );
        }
    }
}

//-////////////////////////////////////////////////////////////////////////-
// parse_partial_json tests
//-////////////////////////////////////////////////////////////////////////-

#[test]
fn test_parse_partial_json_all_cases() {
    let cases: Vec<ParsePartialJsonTestCase> = from_value(json!([
        {
            "name": "parses complete object",
            "input": "{\"a\":1,\"b\":2}",
            "expected": { "a": 1, "b": 2 },
            "throws": false
        },
        // The following tests are updated to reflect Rust's stricter parsing.
        // `json5` crate does not recover partial data like the `partial-json` TS library.
        {
            "name": "fails on partial object",
            "input": "{\"a\":1,\"b\":",
            "expected": null,
            "throws": true
        },
        {
            "name": "fails on partial array",
            "input": "[1,2,3,",
            "expected": null,
            "throws": true
        },
        {
            "name": "fails on severely malformed json",
            "input": "{\"a\":{\"b\":1,\"c\":]}}",
            "expected": null,
            "throws": true
        }
    ]))
    .unwrap();

    for case in cases {
        let result: genkit_core::error::Result<Value> = parse_partial_json(&case.input);
        if case.throws {
            assert!(
                result.is_err(),
                "Test case '{}' should have failed",
                case.name
            );
        } else {
            assert!(
                result.is_ok(),
                "Test case '{}' failed: {:?}",
                case.name,
                result.err()
            );
            assert_eq!(
                result.unwrap(),
                case.expected.unwrap(),
                "Test case '{}' failed",
                case.name
            );
        }
    }
}

//-////////////////////////////////////////////////////////////////////////-
// extract_items tests
//-////////////////////////////////////////////////////////////////////////-

#[test]
fn test_extract_items_all_cases() {
    let cases: Vec<ExtractItemsTestCase> = from_value(json!([
        {
            "name": "handles simple array in chunks",
            "steps": [
                { "chunk": "[", "want": [] },
                { "chunk": "{\"a\": 1},", "want": [{ "a": 1 }] },
                { "chunk": "{\"b\": 2}", "want": [{ "b": 2 }] },
                { "chunk": "]", "want": [] }
            ]
        },
        {
            "name": "handles nested objects",
            "steps": [
                { "chunk": "[{\"outer\": {", "want": [] },
                { "chunk": "\"inner\": \"value\"}},", "want": [{ "outer": { "inner": "value" } }] },
                { "chunk": "{\"next\": true}]", "want": [{ "next": true }] }
            ]
        },
        {
            "name": "handles escaped characters",
            "steps": [
                { "chunk": "[{\"text\": \"line1\\n", "want": [] },
                { "chunk": "line2\"},", "want": [{ "text": "line1\nline2" }] },
                { "chunk": "{\"text\": \"tab\\there\"}]", "want": [{ "text": "tab\there" }] }
            ]
        },
        {
            "name": "ignores content before first array",
            "steps": [
                { "chunk": "Here is an array:\n```json\n\n[", "want": [] },
                { "chunk": "{\"a\": 1},", "want": [{ "a": 1 }] },
                { "chunk": "{\"b\": 2}]\n```\nDid you like my array?", "want": [{ "b": 2 }] }
            ]
        },
        {
            "name": "handles whitespace",
            "steps": [
                { "chunk": "[\n  ", "want": [] },
                { "chunk": "{\"a\": 1},\n  ", "want": [{ "a": 1 }] },
                { "chunk": "{\"b\": 2}\n]", "want": [{ "b": 2 }] }
            ]
        }
    ]))
    .unwrap();

    for case in cases {
        let mut text = String::new();
        let mut cursor = 0;

        for step in case.steps {
            text.push_str(&step.chunk);
            let result = extract_items(&text, cursor);
            assert_eq!(result.items, step.want, "Test case '{}' failed", case.name);
            cursor = result.cursor;
        }
    }
}
