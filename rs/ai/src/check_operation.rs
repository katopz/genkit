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

use genkit_core::action::Action;
use genkit_core::background_action::Operation;
use genkit_core::error::{Error, Result};
use genkit_core::registry::Registry;
use genkit_core::status::StatusCode;
use serde_json::Value;

/// Checks the status of a long-running operation.
///
/// This function looks up the associated `BackgroundAction`'s `check` action
/// from the registry and calls it to get the latest status.
pub async fn check_operation(
    registry: &Registry,
    operation: Operation<Value>,
) -> Result<Operation<Value>> {
    // 1. Extract the base action key from the operation.
    let start_action_key = operation.action.as_ref().ok_or_else(|| {
        Error::new_internal("Operation is missing original action key".to_string())
    })?;

    // 2. Derive the check action's key from the original action key.
    // e.g., /background-model/myModel -> /checkoperation/myModel/check
    let parts: Vec<&str> = start_action_key.split('/').collect();
    if parts.len() < 3 || parts[0] != "" {
        return Err(Error::new_internal(format!(
            "Invalid background action key format: {}",
            start_action_key
        )));
    }
    // The action type is at index 1, name at index 2.
    // e.g., ["", "background-model", "myModel"]
    let action_name = parts[2];

    let check_action_key = format!("/checkoperation/{}/check", action_name);

    // 3. Look up the check action in the registry.
    let check_action_erased = registry
        .lookup_action(&check_action_key.to_lowercase())
        .await
        .ok_or_else(|| {
            Error::new_user_facing(
                StatusCode::NotFound,
                format!(
                    "Check action '{}' not found for '{}'",
                    check_action_key, start_action_key
                ),
                None,
            )
        })?;

    // 4. Downcast the `ErasedAction` to its concrete type.
    // The check action is expected to have the signature `Action<Operation<Value>, Operation<Value>, ()>`.
    let concrete_action = check_action_erased
        .as_any()
        .downcast_ref::<Action<Operation<Value>, Operation<Value>, ()>>()
        .ok_or_else(|| {
            Error::new_internal(format!(
                "Mismatched type for check action '{}'.",
                check_action_key
            ))
        })?;

    // 5. Execute the action.
    let input_value = serde_json::to_value(&operation).map_err(|e| {
        Error::new_internal(format!("Failed to serialize operation to JSON: {}", e))
    })?;
    let action_result = concrete_action.run_http(input_value, None).await?;

    // 6. The result is already the correct type, so we can just return it.
    Ok(action_result.result)
}
