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
    tracing::{self, enable_telemetry, TelemetryConfig},
    Error, Flow, GenerateOptions, GenerateResponse, Genkit, GenkitOptions, Part, ToolArgument,
    ToolConfig,
};

use genkit_vertexai::{common::VertexAIPluginOptions, vertex_ai};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(JsonSchema, Deserialize, Serialize, Debug, Clone, Default)]
pub struct EmptyInput {}

pub fn joke_subject_generator(genkit: &Genkit) {
    genkit.define_tool(
        ToolConfig::<EmptyInput, String> {
            name: "jokeSubjectGenerator".to_string(),
            description: "Can be called to generate a subject for a joke".to_string(),
            ..Default::default()
        },
        |_, _| async { Ok("banana".to_string()) },
    )
}

#[tokio::main]
async fn main() -> genkit::Result<()> {
    enable_telemetry(TelemetryConfig::default()).map_err(|e| Error::new_internal(e.to_string()))?;
    tracing::enable_telemetry(TelemetryConfig::default())
        .map_err(|e| Error::new_internal(e.to_string()))?;
    env_logger::init();
    let vertexai_plugin = vertex_ai(VertexAIPluginOptions {
        project_id: None,
        location: None,
        service_account: None,
    });
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![vertexai_plugin],
        default_model: Some("vertexai/gemini-2.0-flash-lite-001".to_string()),
        ..Default::default()
    })
    .await?;

    joke_subject_generator(&genkit);

    let joke_flow: Flow<String, String, ()> =
        genkit.clone().define_flow("banana", move |input, _| {
            let genkit_cloned = genkit.clone();
            async move {
                log::info!("[joke_flow] Calling model with prompt: '{}'", input);

                let response: GenerateResponse = genkit_cloned
                    .generate_with_options(GenerateOptions {
                        prompt: Some(vec![Part::text(input)]),
                        tools: Some(vec![ToolArgument::Name("jokeSubjectGenerator".to_string())]),
                        ..Default::default()
                    })
                    .await?;

                log::info!("[joke_flow] Got response: {:#?}", response);

                response.text()
            }
        });

    println!("Running jokeFlow...");
    let action_result = joke_flow
        .run(
            "come up with a subject to joke about (using the function provided)".to_string(),
            None,
        )
        .await?;

    println!("Flow action_result: {:#?}", action_result);

    Ok(())
}
