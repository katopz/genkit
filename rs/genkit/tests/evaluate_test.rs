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

use genkit::{
    evaluator::{define_evaluator, evaluate, BaseEvalDataPoint, Evaluation, EvaluatorRequest},
    genkit,
    registry::{register_action, Mutex, Registry},
    Genkit,
};
use genkit_core::error::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

// Helper to define the bonkness evaluator.
fn bonkness_evaluator(registry: &mut Registry) {
    let evaluator_fn = move |datapoint: BaseEvalDataPoint| {
        Box::pin(async move {
            let mut details = HashMap::new();
            details.insert(
                "reasoning".to_string(),
                Value::String("Because I said so!".to_string()),
            );
            Ok(Evaluation {
                test_case_id: datapoint.test_case_id,
                score: "Much bonk".to_string(),
                details: Some(details),
                ..Default::default()
            })
        })
    };
    let evaluator = define_evaluator(
        "bonkness",
        "Bonkness",
        "Judges whether a statement is bonk",
        Box::new(evaluator_fn),
    );
    register_action("bonkness", Arc::new(Mutex::new(evaluator)), registry).unwrap();
}

fn setup() -> Genkit {
    let mut ai = genkit(Default::default());
    bonkness_evaluator(ai.registry_mut());
    ai
}

#[tokio::test]
async fn calls_evaluator() -> Result<()> {
    let ai = setup();
    let dataset = vec![BaseEvalDataPoint {
        test_case_id: "test-case-1".to_string(),
        input: Some(json!("Why did the chicken cross the road?")),
        output: Some(json!("To get to the other side")),
        ..Default::default()
    }];
    let request = EvaluatorRequest {
        evaluator: "bonkness".to_string(),
        dataset,
        eval_run_id: Some("my-dog-eval".to_string()),
        options: None,
    };

    let response = evaluate(&ai, request).await?;

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].score, "Much bonk");
    assert_eq!(
        response[0]
            .details
            .as_ref()
            .unwrap()
            .get("reasoning")
            .unwrap(),
        "Because I said so!"
    );

    Ok(())
}
