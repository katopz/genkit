// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is an "AS IS" BASIS,
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

use crate::action::{Action, ActionBuilder, ActionFnArg};
use crate::error::Result;
use crate::registry::ActionType;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::marker::PhantomData;

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

/// A builder for defining a `BackgroundAction`.
pub struct BackgroundActionBuilder<I, O, FStart, FCheck, FCancel> {
    name: String,
    start_fn: FStart,
    check_fn: FCheck,
    cancel_fn: Option<FCancel>,
    _marker: PhantomData<(I, O)>,
}

impl<I, O, FStart, FCheck, FCancel, FutStart, FutCheck, FutCancel>
    BackgroundActionBuilder<I, O, FStart, FCheck, FCancel>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    FStart: Fn(I, ActionFnArg<()>) -> FutStart + Send + Sync + 'static,
    FutStart: Future<Output = Result<Operation<O>>> + Send,
    FCheck: Fn(Operation<O>, ActionFnArg<()>) -> FutCheck + Send + Sync + 'static,
    FutCheck: Future<Output = Result<Operation<O>>> + Send,
    FCancel: Fn(Operation<O>, ActionFnArg<()>) -> FutCancel + Send + Sync + 'static,
    FutCancel: Future<Output = Result<Operation<O>>> + Send,
{
    /// Creates a new `BackgroundActionBuilder`.
    pub fn new(name: impl Into<String>, start_fn: FStart, check_fn: FCheck) -> Self {
        Self {
            name: name.into(),
            start_fn,
            check_fn,
            cancel_fn: None,
            _marker: PhantomData,
        }
    }

    /// Adds a cancellation function to the background action.
    pub fn with_cancel(mut self, cancel_fn: FCancel) -> Self {
        self.cancel_fn = Some(cancel_fn);
        self
    }

    /// Builds the `BackgroundAction` and registers its component actions.
    pub fn build(self) -> BackgroundAction<I, O> {
        let name = self.name;

        // Define the start action
        let start_action =
            ActionBuilder::new(ActionType::Custom, name.clone(), self.start_fn).build();

        // Define the check action
        let check_action_name = format!("{}/check", name);
        let check_action =
            ActionBuilder::new(ActionType::CheckOperation, check_action_name, self.check_fn)
                .build();

        // Define the cancel action, if provided
        let cancel_action = self.cancel_fn.map(|f| {
            let cancel_action_name = format!("{}/cancel", name);
            ActionBuilder::new(ActionType::CancelOperation, cancel_action_name, f).build()
        });

        // TODO: Register the actions with the registry.

        BackgroundAction {
            start_action,
            check_action,
            cancel_action,
        }
    }
}
