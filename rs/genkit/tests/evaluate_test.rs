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

use genkit::{error::Result, Genkit, GenkitOptions};
use genkit_ai::evaluator::{
    define_evaluator, evaluate, BaseDataPoint, EvalResponse, EvaluatorParams, Score,
};
use genkit_core::registry::Registry;
use rstest::*;
use serde_json::json;
use std::sync::Arc;

// A helper function to define the test evaluator, similar to `bonknessEvaluator` in TS.
fn define_bonkness_evaluator(registry: &mut Registry) {
    define_evaluator::<(), _, _>(registry, "bonkness", move |data_point| async move {
        Ok(EvalResponse {
            test_case_id: data_point.test_case_id,
            evaluation: Score {
                score: Some(json!("Much bonk")),
                status: None,
                error: None,
                details: Some(json!({ "reasoning": "Because I said so!" })),
            },
            trace_id: None,
        })
    });
}

// The rstest fixture, our "beforeEach" block.
struct TestFixture {
    genkit: Arc<Genkit>,
}

#[fixture]
async fn setup() -> TestFixture {
    let genkit = Genkit::init(GenkitOptions::default()).await.unwrap();
    define_bonkness_evaluator(&mut genkit.registry().clone());
    TestFixture { genkit }
}

#[rstest]
#[tokio::test]
async fn test_evaluate_calls_evaluator(#[future] setup: TestFixture) -> Result<()> {
    let fixture = setup.await;

    let dataset = vec![BaseDataPoint {
        test_case_id: Some("test-case-1".to_string()),
        input: json!("Why did the chicken cross the road?"),
        output: Some(json!("To get to the other side")),
        context: None,
        reference: None,
    }];

    let response = evaluate(
        fixture.genkit.registry(),
        EvaluatorParams {
            evaluator: genkit_ai::evaluator::EvaluatorArgument::Name("bonkness".to_string()),
            dataset,
            eval_run_id: Some("my-dog-eval".to_string()),
            options: None,
        },
    )
    .await?;

    assert_eq!(response.len(), 1);
    assert_eq!(response[0].test_case_id, "test-case-1");
    assert_eq!(response[0].evaluation.score, Some(json!("Much bonk")));
    assert_eq!(
        response[0]
            .evaluation
            .details
            .as_ref()
            .unwrap()
            .get("reasoning")
            .unwrap(),
        "Because I said so!"
    );

    Ok(())
}
