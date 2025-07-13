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

//! # Action Runtime Context
//!
//! This module provides mechanisms for managing the action execution context,
//! which allows for lazy initialization of plugins and other components.
//! It is analogous to the runtime context helpers in `action.ts`.

tokio::task_local! {
    /// Task-local flag to indicate if we are in an action's runtime context.
    static IS_RUNTIME_CONTEXT: bool;
}

/// Checks if the current execution is within an action's runtime context.
///
/// In a runtime context, certain operations like plugin initialization are permitted,
/// even if they were deferred during startup.
pub fn is_in_runtime_context() -> bool {
    IS_RUNTIME_CONTEXT.try_with(|&v| v).unwrap_or(false)
}

/// Executes a future within the scope of an action's runtime context.
///
/// This signals to other parts of the framework (like the `Registry`) that
/// it's safe to perform lazy initialization tasks.
pub async fn run_in_action_runtime_context<F, R>(future: F) -> R
where
    F: std::future::Future<Output = R>,
{
    IS_RUNTIME_CONTEXT.scope(true, future).await
}

/// Executes a future outside the scope of an action's runtime context.
pub async fn run_outside_action_runtime_context<F, R>(future: F) -> R
where
    F: std::future::Future<Output = R>,
{
    IS_RUNTIME_CONTEXT.scope(false, future).await
}
