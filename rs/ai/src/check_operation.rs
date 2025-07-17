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

//! # Check Operation
//!
//! This module provides the functionality to check the status of a long-running
//! background operation. It is the Rust equivalent of `check-operation.ts`.

use genkit_core::background_action::Operation;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use serde_json::Value;

/// Checks the status of a long-running operation.
///
/// This function looks up the associated `BackgroundAction`'s `check` action
/// from the registry and calls it to get the latest status.
pub async fn check_operation(
    registry: &Registry,
    operation: &Operation<Value>,
) -> Result<Operation<Value>> {
    let start_action_key = operation
        .action
        .as_ref()
        .ok_or_else(|| Error::new_internal("Operation is missing original action key"))?;

    let check_action_key = start_action_key.replace(
        "/background-model/",
        format!("/{}/", ActionType::CheckOperation).as_str(),
    ) + "/check";

    println!(
        "[check_operation] Looking for check action key: {}",
        check_action_key
    );
    let available_actions = registry
        .list_actions()
        .await
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    println!(
        "[check_operation] Available actions in registry: {:?}",
        available_actions
    );

    let check_action = registry
        .lookup_action(&check_action_key)
        .await
        .ok_or_else(|| {
            Error::new_internal(format!(
                "Check action '{}' not found for '{}'",
                check_action_key, start_action_key
            ))
        })?;

    let input_value = serde_json::to_value(operation)
        .map_err(|e| Error::new_internal(format!("Failed to serialize operation: {}", e)))?;
    let result_value = check_action.run_http_json(input_value, None).await?;
    println!(
        "[check_operation] Received value from action: {}",
        serde_json::to_string_pretty(&result_value).unwrap_or_default()
    );
    let result_field = result_value.get("result").ok_or_else(|| {
        Error::new_internal("Action response is missing 'result' field".to_string())
    })?;
    let checked_op: Operation<Value> = serde_json::from_value(result_field.clone())
        .map_err(|e| Error::new_internal(format!("Failed to deserialize operation: {}", e)))?;

    Ok(checked_op)
}
