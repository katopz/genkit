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

//! # Schema Tests
//!
//! Integration tests for schema validation, ported from `schema_test.ts`.

use genkit_core::error::Error;
use genkit_core::schema::{parse_schema, schema_for, ProvidedSchema};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(JsonSchema, Deserialize, Debug, PartialEq)]
struct SimpleStruct {
    foo: bool,
}

#[derive(JsonSchema, Deserialize, Debug, PartialEq)]
struct ComplexStruct {
    items: Vec<SimpleStruct>,
    date: String,
}

mod test {
    use crate::*;

    #[test]
    fn test_parse_valid_schema() {
        let data = json!({ "foo": true });
        let schema = ProvidedSchema::FromType(schema_for::<SimpleStruct>());
        let parsed = parse_schema::<SimpleStruct>(data, schema).unwrap();
        assert_eq!(parsed, SimpleStruct { foo: true });
    }

    #[test]
    fn test_parse_invalid_type() {
        let data = json!({ "foo": 123 });
        let schema = ProvidedSchema::FromType(schema_for::<SimpleStruct>());
        let result = parse_schema::<SimpleStruct>(data, schema);
        assert!(result.is_err());

        if let Err(Error::Validation(e)) = result {
            let errors = e.errors();
            assert_eq!(errors.len(), 1);
            assert_eq!(errors[0].path, "/foo");
            assert!(errors[0].message.contains("is not of type \"boolean\""));
        } else {
            panic!("Expected a ValidationError, but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_missing_required_field() {
        let data = json!({ "bar": "baz" }); // 'foo' is missing
        let schema = ProvidedSchema::FromType(schema_for::<SimpleStruct>());
        let result = parse_schema::<SimpleStruct>(data, schema);
        assert!(result.is_err());

        if let Err(Error::Validation(e)) = result {
            let errors = e.errors();
            assert_eq!(errors.len(), 1);
            assert!(errors[0].message.contains("\"foo\" is a required property"));
        } else {
            panic!("Expected a ValidationError, but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_nested_invalid_type() {
        let data = json!({
            "items": [
                { "foo": true },
                { "foo": "not a boolean" }
            ],
            "date": "2024-01-01T12:00:00Z"
        });
        let schema = ProvidedSchema::FromType(schema_for::<ComplexStruct>());
        let result = parse_schema::<ComplexStruct>(data, schema);
        assert!(result.is_err());

        if let Err(Error::Validation(e)) = result {
            let errors = e.errors();
            assert_eq!(errors.len(), 1);
            assert_eq!(errors[0].path, "/items/1/foo");
            assert!(errors[0].message.contains("is not of type \"boolean\""));
        } else {
            panic!("Expected a ValidationError, but got {:?}", result);
        }
    }

    #[test]
    fn test_validation_error_display_format() {
        let data = json!({ "foo": 123 });
        let schema = ProvidedSchema::FromType(schema_for::<SimpleStruct>());
        let err = parse_schema::<SimpleStruct>(data, schema).unwrap_err();

        let err_string = err.to_string();
        assert!(err_string.contains("Schema validation failed"));
        assert!(err_string.contains("path: `/foo`"));
        assert!(err_string.contains("message: `123 is not of type \"boolean\"`"));
        assert!(err_string.contains("\"foo\": 123")); // Contains the invalid data
        assert!(err_string.contains("\"type\": \"boolean\"")); // Contains a snippet of the schema
    }
}
