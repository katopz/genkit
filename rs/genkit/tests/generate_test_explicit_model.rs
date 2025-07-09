//! Copyright 2024 Google LLC
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

mod helpers;

use genkit::{model::Part, Genkit, Model};
use genkit_ai::{
    model::{GenerateRequest, GenerateResponse},
    GenerateOptions,
};

use rstest::{fixture, rstest};
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
// Explicit Model Tests
//

#[rstest]
#[tokio::test]
async fn test_explicit_model_calls_the_explicitly_passed_in_model(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("echoModel".to_string())),
            prompt: Some(vec![Part::text("hi")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi; config: {}");
}

#[rstest]
#[tokio::test]
async fn test_explicit_model_rejects_on_invalid_model() {
    let (genkit, _) = genkit_instance_for_test().await;
    let model = Some(Model::Name("modelThatDoesNotExist".to_string()));

    let result = genkit
        .generate_with_options(GenerateOptions::<GenerateResponse> {
            model,
            prompt: Some(vec![Part::text("hi".to_string())]),
            ..Default::default()
        })
        .await;

    // Assert that the operation returned an error
    assert!(result.is_err());

    // Optionally, inspect the error to be more specific
    let err = result.unwrap_err();
    assert!(err
        .to_string()
        .contains("Model 'modelThatDoesNotExist' not found"));
}
