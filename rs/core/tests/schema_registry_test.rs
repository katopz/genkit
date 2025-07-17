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

//! # Schema Registry Tests
//!
//! Tests for schema registration and lookup within the registry.

#[cfg(test)]
mod schema_registry_test {
    use genkit_core::{
        registry::Registry,
        schema::{define_json_schema, define_schema, schema_for, ProvidedSchema},
    };
    use rstest::*;
    use schemars::{JsonSchema, Schema};
    use serde::Deserialize;
    use serde_json::json;

    #[fixture]
    fn registry() -> Registry {
        Registry::new()
    }

    #[derive(JsonSchema, Deserialize, PartialEq, Debug)]
    struct TestSchema {
        name: String,
        count: i32,
    }

    #[rstest]
    fn test_define_and_lookup_schema(registry: Registry) {
        define_schema::<TestSchema>(&registry, "myTestSchema").unwrap();

        let looked_up = registry
            .lookup_schema("myTestSchema")
            .expect("Schema should be found in registry");

        let expected_schema = schema_for::<TestSchema>();

        if let ProvidedSchema::FromType(schema) = looked_up {
            let looked_up_json = serde_json::to_value(&schema).unwrap();
            let expected_json = serde_json::to_value(&expected_schema).unwrap();
            assert_eq!(looked_up_json, expected_json);
        } else {
            panic!("Expected ProvidedSchema::FromType"); //, but got {:?}", looked_up);
        }
    }

    #[rstest]
    fn test_define_and_lookup_json_schema(registry: Registry) {
        let raw_schema_json = json!({
            "type": "object",
            "properties": { "id": { "type": "string" } },
            "required": ["id"]
        });
        let schema_object: Schema = serde_json::from_value(raw_schema_json).unwrap();

        define_json_schema(&registry, "myJsonSchema", schema_object.clone()).unwrap();

        let looked_up = registry
            .lookup_schema("myJsonSchema")
            .expect("JSON Schema should be found in registry");

        if let ProvidedSchema::FromType(schema) = looked_up {
            let looked_up_json = serde_json::to_value(&schema).unwrap();
            let expected_json = serde_json::to_value(&schema_object).unwrap();
            assert_eq!(looked_up_json, expected_json);
        } else {
            panic!("Expected ProvidedSchema::FromType"); //, but got {:?}", looked_up);
        }
    }

    #[rstest]
    fn test_lookup_non_existent_schema(registry: Registry) {
        let looked_up = registry.lookup_schema("nonExistentSchema");
        assert!(
            looked_up.is_none(),
            "Should return None for a schema that is not registered"
        );
    }

    #[rstest]
    fn test_schema_inheritance(registry: Registry) {
        let parent_registry = registry;
        let child_registry = Registry::with_parent(&parent_registry);

        define_schema::<TestSchema>(&parent_registry, "parentSchema").unwrap();

        let looked_up_by_child = child_registry.lookup_schema("parentSchema");
        assert!(
            looked_up_by_child.is_some(),
            "Child should find schema registered on parent"
        );

        define_schema::<TestSchema>(&child_registry, "childSchema").unwrap();

        let looked_up_by_parent = parent_registry.lookup_schema("childSchema");
        assert!(
            looked_up_by_parent.is_none(),
            "Parent should not find schema registered on child"
        );
    }

    #[rstest]
    fn test_duplicate_schema_registration_fails(registry: Registry) {
        define_schema::<TestSchema>(&registry, "duplicateSchema").unwrap();

        let result = define_schema::<TestSchema>(&registry, "duplicateSchema");
        assert!(result.is_err(), "Should not allow duplicate registration");

        let error_message = result.err().unwrap().to_string();
        assert!(error_message.contains("Schema 'duplicateSchema' is already registered."));
    }
}
