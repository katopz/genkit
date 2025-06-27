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

//! # Utilities
//!
//! This module provides various utility functions used throughout the Genkit framework.

use serde_json::{Map, Value};
use std::env;

/// Strips (non-destructively) any properties with `null` values from a `serde_json::Value`.
///
/// This is the Rust equivalent of JavaScript's `stripUndefinedProps`, as `undefined`
/// often serializes to `null` in JSON. This function recursively traverses the JSON
/// structure, removing `null` values from objects and leaving other types unchanged.
pub fn strip_null_values(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = Map::new();
            for (k, v) in map {
                if !v.is_null() {
                    new_map.insert(k, strip_null_values(v));
                }
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(strip_null_values).collect()),
        _ => value,
    }
}

/// Returns the current environment that the app code is running in.
///
/// It reads the `GENKIT_ENV` environment variable. If the variable is not set,
/// it defaults to `"prod"`.
pub fn get_current_env() -> String {
    env::var("GENKIT_ENV").unwrap_or_else(|_| "prod".to_string())
}

/// Checks whether the current environment is `dev`.
pub fn is_dev_env() -> bool {
    get_current_env() == "dev"
}

/// Creates a prefixed string for OpenTelemetry span attributes specific to a feature.
pub fn feature_metadata_prefix(name: &str) -> String {
    format!("feature:{}", name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_strip_null_values_from_object() {
        let input = json!({
            "a": 1,
            "b": null,
            "c": {
                "d": "hello",
                "e": null
            },
            "f": [1, null, 2]
        });
        let expected = json!({
            "a": 1,
            "c": {
                "d": "hello"
            },
            "f": [1, null, 2] // Note: nulls in arrays are preserved by this logic
        });
        assert_eq!(strip_null_values(input), expected);
    }

    #[test]
    fn test_strip_null_values_non_object() {
        assert_eq!(strip_null_values(json!("string")), json!("string"));
        assert_eq!(strip_null_values(json!(123)), json!(123));
        assert_eq!(strip_null_values(json!([1, 2, 3])), json!([1, 2, 3]));
    }

    #[test]
    fn test_get_current_env_default() {
        env::remove_var("GENKIT_ENV");
        assert_eq!(get_current_env(), "prod");
    }

    #[test]
    fn test_get_current_env_dev() {
        env::set_var("GENKIT_ENV", "dev");
        assert_eq!(get_current_env(), "dev");
        env::remove_var("GENKIT_ENV");
    }

    #[test]
    fn test_is_dev_env() {
        env::set_var("GENKIT_ENV", "dev");
        assert!(is_dev_env());
        env::set_var("GENKIT_ENV", "prod");
        assert!(!is_dev_env());
        env::remove_var("GENKIT_ENV");
        assert!(!is_dev_env());
    }

    #[test]
    fn test_feature_metadata_prefix() {
        assert_eq!(feature_metadata_prefix("myFeature"), "feature:myFeature");
    }
}
