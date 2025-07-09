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

use genkit::model::Role;
use genkit::Genkit;
use genkit_ai::model::GenerateRequest;
use genkit_ai::session::Session;
use rstest::{fixture, rstest};
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[rstest]
#[tokio::test]
async fn test_maintains_history_in_the_session(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let session = Arc::new(
        Session::<Value>::new(genkit.registry().clone().into(), None, None, None)
            .await
            .unwrap(),
    );

    let chat = session.chat::<Value>(None).await.unwrap();

    // First message
    let response1 = chat.send("hi").await.unwrap();
    assert_eq!(response1.text().unwrap(), "Echo: hi; config: null");

    // Second message
    let response2 = chat.send("bye").await.unwrap();
    assert_eq!(
        response2.text().unwrap(),
        "Echo: hi,Echo: hi; config: null,bye; config: null"
    );

    // Verify message history
    let history = response2.messages().unwrap();
    assert_eq!(history.len(), 4);
    assert_eq!(history[0].role, Role::User);
    assert_eq!(history[0].text(), "hi");

    assert_eq!(history[1].role, Role::Model);
    assert_eq!(history[1].text(), "Echo: hi; config: null");

    assert_eq!(history[2].role, Role::User);
    assert_eq!(history[2].text(), "bye");

    assert_eq!(history[3].role, Role::Model);
    assert_eq!(
        history[3].text(),
        "Echo: hi,Echo: hi; config: null,bye; config: null"
    );
}
