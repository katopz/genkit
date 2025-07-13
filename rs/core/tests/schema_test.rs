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
//! Tests for schema definition and validation, ported from the TypeScript version.

use genkit_core::error::{Error, Result};
use genkit_core::schema::{parse_schema, schema_for, ProvidedSchema, ValidationErrorDetail};
use rstest::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

#[cfg(test)]
/// 'validate()'
mod validate_test {
    use chrono::{DateTime, Utc};
    use serde::Serialize;

    use super::*;

    /// This function acts as the test runner for various schema validation scenarios.
    /// It uses `rstest` to parameterize the tests, which is a Rust-idiomatic way
    /// to handle the list of test cases from the original TypeScript file.
    #[rstest]
    // Case 1: Valid raw JSON schema
    #[case(
        "should return true for a valid json schema",
        ProvidedSchema::Raw(json!({
            "type": "object",
            "properties": { "foo": { "type": "boolean" } },
        })),
        json!({ "foo": true }),
        true,
        None
    )]
    // Case 2: Invalid raw JSON schema
    #[case(
        "should return errors for an invalid json schema",
        ProvidedSchema::Raw(json!({
            "type": "object",
            "properties": { "foo": { "type": "boolean" } },
        })),
        json!({ "foo": 123 }),
        false,
        Some(vec![ValidationErrorDetail {
            path: "/foo".to_string(),
            message: "123 is not of type \"boolean\"".to_string()
        }])
    )]
    // Case 3: Top-level error for additional properties
    #[case(
        "should be understandable for top-level errors",
        ProvidedSchema::Raw(json!({ "type": "object", "properties": {}, "additionalProperties": false })),
        json!({ "foo": "bar" }),
        false,
        Some(vec![ValidationErrorDetail {
            path: "".to_string(),
            message: "Additional properties are not allowed ('foo' was unexpected)".to_string()
        }])
    )]
    // Case 4: Error for missing required field
    #[case(
        "should be understandable for required fields",
        ProvidedSchema::Raw(json!({
            "type": "object",
            "properties": { "foo": { "type": "string" } },
            "required": ["foo"],
        })),
        json!({}),
        false,
        Some(vec![ValidationErrorDetail {
            path: "".to_string(),
            message: "\"foo\" is a required property".to_string()
        }])
    )]
    fn test_schema_validation(
        #[case] name: &str,
        #[case] schema: ProvidedSchema,
        #[case] data: Value,
        #[case] should_be_valid: bool,
        #[case] expected_errors: Option<Vec<ValidationErrorDetail>>,
    ) {
        // We use `Result<Value, _>` because parse_schema needs a generic type `T` to
        // deserialize into. `Value` is a safe, generic choice when we only care
        // about the validation result, not the deserialized struct.
        let result: Result<Value> = parse_schema(data, schema);

        if should_be_valid {
            assert!(
                result.is_ok(),
                "Test '{}' failed: expected valid, but got error: {:?}",
                name,
                result.err()
            );
        } else {
            let err = result.expect_err(&format!(
                "Test '{}' failed: expected an error, but it was valid.",
                name
            ));
            match err {
                Error::Validation(validation_err) => {
                    let mut actual_errors = validation_err.errors().to_vec();
                    let mut expected = expected_errors.unwrap_or_default();

                    // Sort for consistent comparison, as error order is not guaranteed.
                    actual_errors.sort_by(|a, b| a.path.cmp(&b.path));
                    expected.sort_by(|a, b| a.path.cmp(&b.path));

                    assert_eq!(
                        actual_errors, expected,
                        "Test '{}': Validation errors do not match.",
                        name
                    );
                }
                other_err => {
                    panic!(
                        "Test '{}' failed: expected a ValidationError, but got a different error: {:?}",
                        name, other_err
                    );
                }
            }
        }
    }

    /// The following tests correspond to the `zod` schema tests in the original file.
    /// We use structs with `#[derive(JsonSchema)]` as the Rust equivalent.

    #[derive(JsonSchema, Deserialize, Debug)]
    #[allow(unused)]
    struct SimpleSchema {
        foo: bool,
    }

    #[test]
    fn test_struct_schema_valid() {
        let data = json!({ "foo": true });
        let schema = ProvidedSchema::FromType(schema_for::<SimpleSchema>());
        let result: Result<SimpleSchema> = parse_schema(data, schema);
        assert!(
            result.is_ok(),
            "Should be valid for a correct struct, but got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_struct_schema_invalid() {
        let data = json!({ "foo": 123 });
        let schema = ProvidedSchema::FromType(schema_for::<SimpleSchema>());
        let result: Result<SimpleSchema> = parse_schema(data, schema);
        let err = result.expect_err("Should be invalid for an incorrect struct type");
        match err {
            Error::Validation(e) => {
                assert_eq!(e.errors().len(), 1);
                assert_eq!(e.errors()[0].path, "/foo");
                assert!(e.errors()[0].message.contains("is not of type \"boolean\""));
            }
            _ => panic!("Expected a validation error"),
        }
    }

    #[derive(JsonSchema, Serialize, Deserialize, Debug)]
    struct DateTimeSchema {
        date: DateTime<Utc>,
    }

    #[test]
    fn test_datetime_schema_valid() {
        let data = json!({ "date": "2024-05-22T17:00:00Z" });
        let schema = ProvidedSchema::FromType(schema_for::<DateTimeSchema>());
        let result: Result<DateTimeSchema> = parse_schema(data, schema);
        assert!(
            result.is_ok(),
            "Should correctly validate date-time format, but got: {:?}",
            result.err()
        );
    }

    #[derive(JsonSchema, Deserialize, Debug)]
    #[allow(unused)]
    struct NestedSchemaItem {
        bar: bool,
    }

    #[derive(JsonSchema, Deserialize, Debug)]
    #[allow(unused)]
    struct NestedSchema {
        foo: Vec<NestedSchemaItem>,
    }

    #[test]
    fn test_nested_schema_error_path() {
        // This test case is adapted from the TS version. The original `data` was `{ foo: [{ bar: 123 }] }`
        // which corresponds to an invalid type for `bar`. The error path in TS was `foo.0.bar`.
        // The Rust `jsonschema` crate provides the JSON Pointer path `/foo/0/bar`.
        let data = json!({ "foo": [{ "bar": 123 }] });
        let schema = ProvidedSchema::FromType(schema_for::<NestedSchema>());
        let result: Result<NestedSchema> = parse_schema(data, schema);
        let err = result.expect_err("Should fail validation for nested object with wrong type");
        match err {
            Error::Validation(e) => {
                assert_eq!(e.errors().len(), 1, "Expected one validation error");
                let error_detail = &e.errors()[0];
                assert_eq!(error_detail.path, "/foo/0/bar");
                assert!(error_detail.message.contains("is not of type \"boolean\""));
            }
            _ => panic!("Expected a validation error"),
        }
    }
}

#[cfg(test)]
/// 'parse()'
mod parse_test {
    use super::*;

    #[derive(JsonSchema, Deserialize, Debug, PartialEq)]
    struct ParseSchema {
        foo: bool,
    }

    #[test]
    /// 'should return a ValidationError for invalid schema'
    fn test_returns_validation_error_for_invalid_schema() {
        let data = json!({ "foo": 123 });
        let schema = ProvidedSchema::FromType(schema_for::<ParseSchema>());
        let result: Result<ParseSchema> = parse_schema(data, schema);

        assert!(result.is_err(), "Expected parse_schema to return an error.");

        let err = result.unwrap_err();
        match err {
            Error::Validation(_) => {
                // Correct error type, test passes.
            }
            _ => panic!("Expected Error::Validation, but got a different error type."),
        }
    }

    #[test]
    /// 'should return the data if valid'
    fn test_returns_data_if_valid() {
        let data = json!({ "foo": true });
        let schema = ProvidedSchema::FromType(schema_for::<ParseSchema>());
        let result: Result<ParseSchema> = parse_schema(data, schema);

        assert!(
            result.is_ok(),
            "Expected parse_schema to succeed, but it failed: {:?}",
            result.err()
        );

        assert_eq!(
            result.unwrap(),
            ParseSchema { foo: true },
            "The parsed data did not match the expected data."
        );
    }
}

#[cfg(test)]
/// 'toJsonSchema'
mod to_json_schema_test {
    use super::*;
    use schemars::schema_for;
    use serde_json::{json, to_value, Value};

    #[test]
    /// 'converts struct to JSON schema'
    fn test_converts_struct_to_json_schema() {
        // The Rust equivalent of a Zod schema is a struct that derives `JsonSchema`.
        #[derive(JsonSchema)]
        #[allow(unused)]
        struct MySchema {
            output: String,
        }

        // `schema_for!` is the equivalent of the `toJsonSchema` function.
        let generated_schema = schema_for!(MySchema);
        let generated_json: Value =
            to_value(generated_schema).expect("Failed to serialize generated schema to JSON");

        // The expected schema now includes the `$schema` and `title` fields that
        // `schemars` adds by default.
        let expected_json = json!({
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "title": "MySchema",
          "type": "object",
          "properties": {
            "output": {
              "type": "string"
            }
          },
          "required": [
            "output"
          ]
        });

        assert_eq!(
            generated_json, expected_json,
            "The generated schema does not match the expected JSON schema."
        );
    }
}
