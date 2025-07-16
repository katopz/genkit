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

//! # Genkit Background Actions
//!
//! This module provides support for defining and managing long-running background
//! actions. It is the Rust equivalent of `background-action.ts`.
//!
//! A `BackgroundAction` is composed of three underlying regular `Action`s:
//! - `start`: To begin the long-running operation.
//! - `check`: To poll for the status of the operation.
//! - `cancel`: (Optional) To request cancellation of the operation.

use crate::action::{define_action, Action, ActionFnArg};
use crate::error::Result;
use crate::registry::{ActionType, Registry};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::sync::Arc;

/// Represents the state of a long-running background operation.
///
/// This struct is used to track the progress and result of a `BackgroundAction`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct Operation<O = Value> {
    /// The key of the action that this operation belongs to.
    pub action: Option<String>,
    /// A unique identifier for this operation.
    pub id: String,
    /// Indicates whether the operation has completed.
    #[serde(default)]
    pub done: bool,
    /// The output of the operation, available when `done` is true and there was no error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<O>,
    /// Details of an error, if one occurred.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    /// Additional metadata associated with the operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Represents a long-running background action.
///
/// Unlike a regular action, a background action can run for a long time.
/// The `start` method returns an `Operation` that can be used to check the
/// status and retrieve the eventual response.
pub struct BackgroundAction<I, O> {
    /// The underlying action that starts the operation.
    pub start_action: Action<I, Operation<O>, ()>,
    /// The underlying action that checks the operation's status.
    pub check_action: Action<Operation<O>, Operation<O>, ()>,
    /// The underlying action that cancels the operation, if supported.
    pub cancel_action: Option<Action<Operation<O>, Operation<O>, ()>>,
}

/// Defines a background action and registers its component actions with the given registry.
pub fn define_background_action<I, O, FStart, FCheck, FCancel, FutStart, FutCheck, FutCancel>(
    registry: &Registry,
    name: impl Into<String>,
    action_type: ActionType,
    start_fn: FStart,
    check_fn: FCheck,
    cancel_fn: Option<FCancel>,
) -> BackgroundAction<I, O>
where
    I: DeserializeOwned + JsonSchema + Serialize + Send + Sync + Clone + 'static,
    O: Serialize + DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    FStart: Fn(I, ActionFnArg<()>) -> FutStart + Send + Sync + 'static,
    FutStart: Future<Output = Result<Operation<O>>> + Send,
    FCheck: Fn(Operation<O>, ActionFnArg<()>) -> FutCheck + Send + Sync + 'static,
    FutCheck: Future<Output = Result<Operation<O>>> + Send,
    FCancel: Fn(Operation<O>, ActionFnArg<()>) -> FutCancel + Send + Sync + 'static,
    FutCancel: Future<Output = Result<Operation<O>>> + Send,
{
    let name = name.into();
    let action_key = Arc::new(format!(
        "/{}/{}",
        serde_json::to_value(action_type).unwrap().as_str().unwrap(),
        &name
    ));

    // Arc the functions to allow them to be captured by `Fn` closures.
    let start_fn = Arc::new(start_fn);
    let check_fn = Arc::new(check_fn);
    let cancel_fn = cancel_fn.map(Arc::new);

    // Define the start action, wrapping the original function to inject the operation key.
    let start_action = {
        let action_key = action_key.clone();
        let start_fn = start_fn;
        let wrapped_start_fn = move |input: I, args: ActionFnArg<()>| {
            let start_fn = start_fn.clone();
            let action_key = action_key.clone();
            async move {
                let mut operation = (*start_fn)(input, args).await?;
                operation.action = Some(action_key.to_string());
                Ok(operation)
            }
        };
        define_action(registry, action_type, name.clone(), wrapped_start_fn)
    };

    // Define the check action.
    let check_action = {
        let action_key = action_key.clone();
        let check_fn = check_fn;
        let wrapped_check_fn = move |input: Operation<O>, args: ActionFnArg<()>| {
            let check_fn = check_fn.clone();
            let action_key = action_key.clone();
            async move {
                let mut operation = (*check_fn)(input, args).await?;
                operation.action = Some(action_key.to_string());
                Ok(operation)
            }
        };
        let check_action_name = format!("{}/check", &name);
        define_action(
            registry,
            ActionType::CheckOperation,
            check_action_name,
            wrapped_check_fn,
        )
    };

    // Define the cancel action, if a function was provided.
    let cancel_action = cancel_fn.map(|cancel_fn| {
        let action_key = action_key.clone();
        let wrapped_cancel_fn = move |input: Operation<O>, args: ActionFnArg<()>| {
            let cancel_fn = cancel_fn.clone();
            let action_key = action_key.clone();
            async move {
                let mut operation = (*cancel_fn)(input, args).await?;
                operation.action = Some(action_key.to_string());
                Ok(operation)
            }
        };
        let cancel_action_name = format!("{}/cancel", &name);
        define_action(
            registry,
            ActionType::CancelOperation,
            cancel_action_name,
            wrapped_cancel_fn,
        )
    });

    BackgroundAction {
        start_action,
        check_action,
        cancel_action,
    }
}
