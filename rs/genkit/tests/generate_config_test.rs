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

mod helpers;

use genkit::{model::Part, Genkit};
use genkit_ai::{model::GenerateRequest, GenerateOptions, ModelRef};
use rstest::{fixture, rstest};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

//
// Config Tests
//

#[rstest]
#[tokio::test]
async fn test_config_takes_config_passed_to_generate() {
    let (genkit, _) = genkit_instance_for_test().await;

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(
                ModelRef::new(json!({ "name": "echoModel" }))
                    .with_version("bcd")
                    .into(),
            ),
            prompt: Some(vec![Part::text("hi")]),
            config: Some(json!({
                "temperature": 11
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    let left_str = response.text().unwrap();
    let right_str = r#"Echo: hi; config: {"version":"bcd","temperature":11}"#;
    let prefix = "Echo: hi; config: ";

    // Extract the JSON part from each string
    let left_json_str = left_str.strip_prefix(prefix).expect("Prefix not found");
    let right_json_str = right_str.strip_prefix(prefix).expect("Prefix not found");

    // Parse them into serde_json::Value
    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    // Assert the JSON values are equal, ignoring key order
    assert_eq!(left_value, right_value);
}

#[rstest]
#[tokio::test]
async fn test_config_merges_config_from_the_ref() {
    let (genkit, _) = genkit_instance_for_test().await;

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(
                ModelRef::new(json!({
                    "name": "echoModel"
                }))
                .with_config(json!({
                    "version": "abc"
                }))
                .into(),
            ),
            prompt: Some(vec![Part::text("hi")]),
            config: Some(json!({
                "temperature": 11
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    let left_str = response.text().unwrap();
    let right_str = r#"Echo: hi; config: {"version":"abc","temperature":11}"#;
    let prefix = "Echo: hi; config: ";

    // 1. Extract the JSON part from each string
    println!("left_str:{}", left_str);
    let left_json_str = left_str.strip_prefix(prefix).expect("Prefix not found");
    let right_json_str = right_str.strip_prefix(prefix).expect("Prefix not found");

    // 2. Parse them into serde_json::Value
    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    // 3. Assert the JSON values are equal
    assert_eq!(left_value, right_value);
}

#[rstest]
#[tokio::test]
async fn test_config_picks_up_top_level_version_from_the_ref() {
    let (genkit, _) = genkit_instance_for_test().await;

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(
                ModelRef::new(json!({
                    "name": "echoModel"
                }))
                .with_config(json!({
                    "version": "abc"
                }))
                .into(),
            ),
            prompt: Some(vec![Part::text("hi")]),
            config: Some(json!({
                "temperature": 11
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    let left_str = response.text().unwrap();
    let right_str = r#"Echo: hi; config: {"version":"abc","temperature":11}"#;
    let prefix = "Echo: hi; config: ";

    // 1. Extract the JSON part from each string
    println!("left_str:{}", left_str);
    let left_json_str = left_str.strip_prefix(prefix).expect("Prefix not found");
    let right_json_str = right_str.strip_prefix(prefix).expect("Prefix not found");

    // 2. Parse them into serde_json::Value
    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    // 3. Assert the JSON values are equal
    assert_eq!(left_value, right_value);
}
