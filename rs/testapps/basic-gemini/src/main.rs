use genkit::{
    Flow, GenerateOptions, GenerateResponse, Genkit, GenkitOptions, Part, ToolArgument, ToolConfig,
};
use genkit_vertexai::{common::VertexAIPluginOptions, vertex_ai};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
pub struct JokeSubject {
    #[serde(rename = "jokeSubject")]
    pub joke_subject: String,
}

pub fn joke_subject_generator(genkit: &Genkit) {
    genkit.define_tool(
        ToolConfig {
            name: "jokeSubjectGenerator".to_string(),
            description: "Can be called to generate a subject for a joke".to_string(),
            input_schema: Some(String::new()),
            output_schema: Some(String::new()),
            ..Default::default()
        },
        |_, _| async { Ok("banana".to_string()) },
    )
}

#[tokio::main]
async fn main() -> genkit::Result<()> {
    env_logger::init();
    let vertexai_plugin = vertex_ai(VertexAIPluginOptions {
        project_id: None, //Some("talent-finder-437508".to_string()),
        location: None,   //Some("asia-northeast1".to_string()),
        service_account: None,
    });
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![vertexai_plugin],
        default_model: Some("gemini-1.5-flash".to_string()),
        ..Default::default()
    })
    .await?;

    let joke_flow: Flow<String, String, ()> =
        genkit.clone().define_flow("banana", move |input, _| {
            let genkit_cloned = genkit.clone();
            async move {
                let response: GenerateResponse = genkit_cloned
                    .generate_with_options(GenerateOptions {
                        prompt: Some(vec![Part::text(input)]),
                        tools: Some(vec![ToolArgument::Name("jokeSubjectGenerator".to_string())]),
                        ..Default::default()
                    })
                    .await?;

                response.text()
            }
        });

    println!("Running jokeFlow...");
    let result = joke_flow
        .run(
            "come up with a subject to joke about (using the function provided)".to_string(),
            None,
        )
        .await?;

    println!("Flow result: {:?}", result);

    Ok(())
}
