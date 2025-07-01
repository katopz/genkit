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
