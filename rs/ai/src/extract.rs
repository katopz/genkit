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

//! # JSON Extraction Utilities
//!
//! This module provides functions for extracting and parsing JSON from text,
//! often from the output of large language models. It is the Rust equivalent
// of `extract.ts`.

use genkit_core::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Extracts a JSON object or array from a string.
///
/// This function scans the text to find the first occurrence of a top-level
/// JSON object (`{...}`) or array (`[...]`) and attempts to parse it. It is
/// designed to be lenient, ignoring any text that comes before or after the
/// JSON structure. This is particularly useful for cleaning up model outputs
/// that might include explanatory text around a JSON payload.
///
/// # Arguments
///
/// * `text` - The string to search for JSON.
///
/// # Returns
///
/// A `Result` containing an `Option<T>` where `T` is the deserialized type.
/// Returns `Ok(None)` if no valid JSON structure is found. Returns an `Err`
/// only if a potential JSON structure is found but is malformed.
pub fn extract_json<T: DeserializeOwned>(text: &str) -> Result<Option<T>> {
    let mut start_pos: Option<usize> = None;
    let mut opening_char: Option<char> = None;
    let mut closing_char: Option<char> = None;
    let mut nesting_count = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, char) in text.chars().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if char == '\\' {
            escape_next = true;
            continue;
        }

        if char == '"' {
            in_string = !in_string;
            continue;
        }

        if in_string {
            continue;
        }

        if start_pos.is_none() {
            if char == '{' || char == '[' {
                opening_char = Some(char);
                closing_char = if char == '{' { Some('}') } else { Some(']') };
                start_pos = Some(i);
                nesting_count = 1;
            }
        } else if Some(char) == opening_char {
            nesting_count += 1;
        } else if Some(char) == closing_char {
            nesting_count -= 1;
            if nesting_count == 0 {
                let json_str = &text[start_pos.unwrap()..=i];
                return json5::from_str(json_str)
                    .map(Some)
                    .map_err(|e| Error::new_internal(format!("JSON parsing error: {}", e)));
            }
        }
    }
    // If we exit the loop and found a start but no end, it's partial JSON.
    // Attempt to parse the partial string. If it fails, propagate the error.
    if let Some(start) = start_pos {
        if nesting_count > 0 {
            let partial_str = &text[start..];
            // Note: This specific check for `in_string` when exiting the loop (meaning
            // the JSON structure is incomplete and ends within a string literal)
            // is added to align with the stricter error-throwing behavior of the
            // TypeScript `partial-json` library. Without this, `json5` might leniently
            // parse such inputs (e.g., `{"a": "`) as valid (e.g., `{"a": null}`),
            // which would cause a discrepancy with the TS implementation's error expectation.
            if in_string {
                return Err(Error::new_internal(
                    "Unclosed string literal in partial JSON structure".to_string(),
                ));
            }
            return parse_partial_json(partial_str).map(Some);
        }
    }

    Ok(None)
}

/// Parses a partially complete JSON string.
///
/// This function uses a lenient JSON5 parser. Note that unlike the JavaScript
/// equivalent which uses the `partial-json` library, this implementation may
/// not be able to recover objects from severely truncated or malformed JSON.
/// It works best for JSON that is well-formed but simply incomplete (e.g.,
/// missing a final closing brace if the parser supports it) or has extra syntax
/// like trailing commas.
pub fn parse_partial_json<T: DeserializeOwned>(json_string: &str) -> Result<T> {
    json5::from_str(json_string)
        .map_err(|e| Error::new_internal(format!("Partial JSON parsing error: {}", e)))
}

/// The result of an `extract_items` operation.
pub struct ExtractItemsResult {
    /// The items that were successfully extracted and parsed.
    pub items: Vec<Value>,
    /// The new cursor position to use for the next call on an appended string.
    pub cursor: usize,
}

/// Extracts complete JSON objects from the first array found in the text.
///
/// This function is designed for streaming scenarios where text arrives in chunks.
/// It processes text from a given `cursor` position and returns any fully-formed
/// objects found within the first top-level array, along with the updated cursor
/// position.
pub fn extract_items(text: &str, mut cursor: usize) -> ExtractItemsResult {
    let mut items = Vec::new();

    // On the first run, find the start of the array.
    if cursor == 0 {
        if let Some(array_start) = text.find('[') {
            cursor = array_start + 1;
        } else {
            // No array found yet, return empty.
            return ExtractItemsResult {
                items,
                cursor: text.len(),
            };
        }
    }

    let mut object_start: Option<usize> = None;
    let mut brace_count = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, char) in text.chars().enumerate().skip(cursor) {
        if escape_next {
            escape_next = false;
            continue;
        }
        if char == '\\' {
            escape_next = true;
            continue;
        }
        if char == '"' {
            if object_start.is_some() {
                in_string = !in_string;
            }
            continue;
        }
        if in_string {
            continue;
        }

        if char == '{' {
            if brace_count == 0 {
                object_start = Some(i);
            }
            brace_count += 1;
        } else if char == '}' {
            if brace_count > 0 {
                brace_count -= 1;
                if brace_count == 0 {
                    if let Some(start) = object_start {
                        let obj_str = &text[start..=i];
                        if let Ok(item) = json5::from_str(obj_str) {
                            items.push(item);
                        }
                        // This object is now processed.
                        object_start = None;
                    }
                }
            }
        } else if char == ']' && brace_count == 0 {
            // End of the top-level array.
            cursor = i + 1;
            break;
        }
        cursor = i + 1;
    }

    ExtractItemsResult { items, cursor }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Deserialize, Debug, PartialEq)]
    struct SimpleObject {
        a: i32,
    }

    #[test]
    fn test_extract_json_simple_object() {
        let text = "Here is some JSON: {\"a\": 1}. That's all.";
        let result: Option<SimpleObject> = extract_json(text).unwrap();
        assert_eq!(result, Some(SimpleObject { a: 1 }));
    }

    #[test]
    fn test_extract_json_simple_array() {
        let text = "An array: [1, 2, 3].";
        let result: Option<Vec<i32>> = extract_json(text).unwrap();
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_extract_json_no_json() {
        let text = "There is no json here.";
        let result: Option<Value> = extract_json(text).unwrap();
        assert_eq!(result, None);
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
        let text = "{\"text\": \"this is {not} a nested object\"}";
        let result: Option<HashMap<String, String>> = extract_json(text).unwrap();
        assert_eq!(
            result.unwrap().get("text").unwrap(),
            "this is {not} a nested object"
        );
    }

    #[test]
    fn test_extract_json_malformed() {
        let text = "Here is bad json: {\"a\": \"";
        let result: Result<Option<Value>> = extract_json(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_items_streaming() {
        let mut text = String::new();
        let mut cursor = 0;

        // First chunk
        text.push_str("Some text... [{\"a\": 1},");
        let result1 = extract_items(&text, cursor);
        assert_eq!(result1.items.len(), 1);
        assert_eq!(result1.items[0], json!({"a": 1}));
        cursor = result1.cursor;

        // Second chunk
        text.push_str(" {\"b\": 2}");
        let result2 = extract_items(&text, cursor);
        assert_eq!(result2.items.len(), 1);
        assert_eq!(result2.items[0], json!({"b": 2}));
        cursor = result2.cursor;

        // Third chunk, closing the array
        text.push(']');
        let result3 = extract_items(&text, cursor);
        assert_eq!(result3.items.len(), 0);
        cursor = result3.cursor;
        assert!(text.get(cursor..).unwrap_or("").is_empty());
    }
}
