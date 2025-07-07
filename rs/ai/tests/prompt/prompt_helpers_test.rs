// #![cfg(feature = "dotprompt-private")]
// // Copyright 2024 Google LLC
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// //! # Prompt Helper Tests

// use genkit_ai::{
//     model::{define_model, ModelInfo},
//     prompt::load_prompt_folder,
// };
// use genkit_core::{error::Result, registry::Registry};

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[tokio::test]
//     async fn test_load_prompt_folder_and_partials() -> Result<()> {
//         let mut registry = Registry::new();

//         // Define a dummy model so the prompt can resolve it.
//         define_model(
//             &mut registry,
//             crate::model::DefineModelOptions {
//                 name: "testModel".to_string(),
//                 ..Default::default()
//             },
//             |_, _| async { Ok(Default::default()) },
//         );

//         // Path to the test prompt files created earlier.
//         let prompt_dir = "./tests/prompt";

//         // Load the prompts and partials from the directory.
//         load_prompt_folder(&mut registry, prompt_dir, "dotprompt").await?;

//         // 1. Verify that the main prompt was loaded correctly.
//         let prompt_action = registry.lookup_action("/prompt/dotprompt/test").await;
//         assert!(
//             prompt_action.is_some(),
//             "Prompt '/prompt/dotprompt/test' should be loaded."
//         );
//         let prompt_meta = prompt_action.unwrap().metadata();
//         assert_eq!(prompt_meta.name, "dotprompt/test");
//         assert_eq!(
//             prompt_meta
//                 .metadata
//                 .get("model")
//                 .and_then(|v| v.as_str())
//                 .unwrap(),
//             "testModel"
//         );

//         // 2. Verify that the partial was defined correctly.
//         // We can check this by trying to render a template that uses the partial.
//         let handlebars = &registry.dotprompt.handlebars;
//         let rendered = handlebars
//             .render_template("{{> partial}}", &())
//             .expect("Failed to render template with partial");
//         assert_eq!(rendered, "...and a partial greeting.");

//         Ok(())
//     }
// }
