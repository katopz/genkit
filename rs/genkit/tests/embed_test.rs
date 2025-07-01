use genkit::{Genkit, GenkitOptions};
use genkit_ai::document::Document;
use genkit_ai::embedder::{
    define_embedder, embed, EmbedParams, EmbedRequest, EmbedResponse, EmbedderRef,
};
use genkit_core::error::Result;
use genkit_core::registry::Registry;
use rstest::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

// Define a struct for our test embedder's options to match the TS test
#[derive(JsonSchema, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct TestEmbedderOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

// State shared between the test and the embedder implementation
struct TestState {
    last_request: Option<EmbedRequest<TestEmbedderOptions>>,
}

// This helper defines the test embedder and sets up state capture.
// It's the equivalent of `defineTestEmbedder` in the TS test.
fn define_test_embedder(registry: &mut Registry) -> Arc<Mutex<TestState>> {
    let shared_state = Arc::new(Mutex::new(TestState { last_request: None }));
    let state_for_closure = shared_state.clone();

    define_embedder(
        registry,
        "echoEmbedder",
        move |req: EmbedRequest<TestEmbedderOptions>, _args| {
            let state_clone = state_for_closure.clone();
            async move {
                // Capture the request for test assertions
                let mut state = state_clone.lock().unwrap();
                state.last_request = Some(req);

                // Return a fixed embedding response
                Ok(EmbedResponse {
                    embeddings: vec![genkit_ai::embedder::Embedding {
                        embedding: vec![1.0, 2.0, 3.0, 4.0],
                        metadata: None,
                    }],
                })
            }
        },
    );

    shared_state
}

// The rstest fixture, our "beforeEach"
struct TestFixture {
    genkit: Arc<Genkit>,
    state: Arc<Mutex<TestState>>,
}

#[fixture]
async fn setup() -> TestFixture {
    let genkit = Genkit::init(GenkitOptions::default()).await.unwrap();
    let state = define_test_embedder(&mut genkit.registry().clone());
    TestFixture { genkit, state }
}

#[rstest]
#[tokio::test]
async fn test_embed_string_content(#[future] setup: TestFixture) -> Result<()> {
    let fixture = setup.await;

    let response = embed(
        fixture.genkit.registry(),
        EmbedParams {
            embedder: genkit_ai::embedder::EmbedderArgument::Name("echoEmbedder".to_string()),
            content: vec![Document::from_text("hi", None)],
            options: None::<TestEmbedderOptions>,
        },
    )
    .await?;

    let state = fixture.state.lock().unwrap();
    let last_request = state.last_request.as_ref().unwrap();

    assert_eq!(last_request.input.len(), 1);
    assert_eq!(last_request.input[0].text(), "hi");
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_embed_document_content(#[future] setup: TestFixture) -> Result<()> {
    let fixture = setup.await;

    let doc = Document::from_text("hi", None);
    let response = embed(
        fixture.genkit.registry(),
        EmbedParams {
            embedder: genkit_ai::embedder::EmbedderArgument::Name("echoEmbedder".to_string()),
            content: vec![doc.clone()],
            options: None::<TestEmbedderOptions>,
        },
    )
    .await?;

    let state = fixture.state.lock().unwrap();
    let last_request = state.last_request.as_ref().unwrap();

    assert_eq!(last_request.input, vec![doc]);
    assert_eq!(response.len(), 1);
    assert_eq!(response[0].embedding, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_embed_with_options(#[future] setup: TestFixture) -> Result<()> {
    let fixture = setup.await;
    let options = TestEmbedderOptions {
        temperature: Some(11),
        ..Default::default()
    };

    let _response = embed(
        fixture.genkit.registry(),
        EmbedParams {
            embedder: genkit_ai::embedder::EmbedderArgument::Name("echoEmbedder".to_string()),
            content: vec![Document::from_text("hi", None)],
            options: Some(options.clone()),
        },
    )
    .await?;

    let state = fixture.state.lock().unwrap();
    let last_request = state.last_request.as_ref().unwrap();

    assert_eq!(last_request.options, Some(options));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_embed_with_ref_config(#[future] setup: TestFixture) -> Result<()> {
    let fixture = setup.await;

    let embedder_ref = EmbedderRef {
        name: "echoEmbedder".to_string(),
        config: Some(TestEmbedderOptions {
            version: Some("abc".to_string()),
            ..Default::default()
        }),
        version: None,
    };

    let _response = embed(
        fixture.genkit.registry(),
        EmbedParams {
            embedder: genkit_ai::embedder::EmbedderArgument::Ref(embedder_ref),
            content: vec![Document::from_text("hi", None)],
            options: Some(TestEmbedderOptions {
                temperature: Some(11),
                ..Default::default()
            }),
        },
    )
    .await?;

    let state = fixture.state.lock().unwrap();
    let last_request = state.last_request.as_ref().unwrap();

    let expected_options = TestEmbedderOptions {
        temperature: Some(11),
        version: Some("abc".to_string()),
    };
    assert_eq!(last_request.options, Some(expected_options));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_embed_with_ref_top_level_version(#[future] setup: TestFixture) -> Result<()> {
    let fixture = setup.await;

    let embedder_ref = EmbedderRef {
        name: "echoEmbedder".to_string(),
        config: None,
        version: Some("abc".to_string()),
    };

    let _response = embed(
        fixture.genkit.registry(),
        EmbedParams {
            embedder: genkit_ai::embedder::EmbedderArgument::Ref(embedder_ref),
            content: vec![Document::from_text("hi", None)],
            options: Some(TestEmbedderOptions {
                temperature: Some(11),
                ..Default::default()
            }),
        },
    )
    .await?;

    let state = fixture.state.lock().unwrap();
    let last_request = state.last_request.as_ref().unwrap();

    let expected_options = TestEmbedderOptions {
        temperature: Some(11),
        version: Some("abc".to_string()),
    };
    assert_eq!(last_request.options, Some(expected_options));
    Ok(())
}
