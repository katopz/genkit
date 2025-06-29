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

//! # Genkit Flow
//!
//! This module provides the structures and traits for defining and executing
//! flows. A flow is a series of steps that can involve models, tools, and
//! custom logic to accomplish a task. It is the Rust equivalent of flow
//! definitions in `@genkit-ai/core`.

use crate::error::Result;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::future::Future;

/// Represents the metadata that defines a flow.
#[derive(Debug, Default, Clone)]
pub struct FlowInfo {
    /// The unique name of the flow.
    pub name: String,
    /// A description of what the flow does.
    pub description: String,
    /// A JSON schema for the flow's input.
    pub input_schema: Option<Value>,
    /// A JSON schema for the flow's output.
    pub output_schema: Option<Value>,
    /// A JSON schema for the chunks streamed by the flow.
    pub stream_schema: Option<Value>,
}

/// A trait representing an executable flow.
///
/// This is the core of the flow system. Implementations of this trait can be
/// registered and run by the Genkit framework.
// Note: In a full implementation, this might use `async_trait` for cleaner syntax.
pub trait Flow<I, O, S>: Send + Sync
where
    I: DeserializeOwned + Send,
    O: Serialize + Send,
    S: Serialize + Send,
{
    /// Returns the metadata for the flow.
    fn info(&self) -> &FlowInfo;

    /// Runs the flow with the given input and returns the final output.
    fn run<'a>(&'a self, input: I) -> impl Future<Output = Result<O>> + Send + 'a;

    // TODO: Add streaming support.
    // fn stream(...)
}

/// A simple struct to wrap a function and make it a `Flow`.
///
/// This is a stand-in for what would likely be a more complex macro-based
/// `define_flow` system.
#[derive(Clone)]
pub struct FunctionFlow<F, Fut, I, O, S>
where
    I: DeserializeOwned + Send,
    O: Serialize + Send,
    S: Serialize + Send,
    F: Fn(I) -> Fut + Send + Sync + Clone,
    Fut: Future<Output = Result<O>> + Send,
{
    info: FlowInfo,
    func: F,
    // PhantomData to hold the generic types, satisfying the compiler.
    _phantom: std::marker::PhantomData<(I, O, S)>,
}

impl<F, Fut, I, O, S> Flow<I, O, S> for FunctionFlow<F, Fut, I, O, S>
where
    I: DeserializeOwned + Send + Sync,
    O: Serialize + Send + Sync,
    S: Serialize + Send + Sync,
    F: Fn(I) -> Fut + Send + Sync + Clone,
    Fut: Future<Output = Result<O>> + Send,
{
    fn info(&self) -> &FlowInfo {
        &self.info
    }

    fn run<'a>(&'a self, input: I) -> impl Future<Output = Result<O>> + Send + 'a {
        (self.func)(input)
    }
}

/// Defines a new flow from a function.
///
/// This function acts as a constructor for a `FunctionFlow`, which wraps a
/// given function and associated metadata to make it compatible with the `Flow`
/// trait.
pub fn define_flow<I, O, S, F, Fut>(info: FlowInfo, func: F) -> FunctionFlow<F, Fut, I, O, S>
where
    I: DeserializeOwned + Send,
    O: Serialize + Send,
    S: Serialize + Send,
    F: Fn(I) -> Fut + Send + Sync + Clone,
    Fut: Future<Output = Result<O>> + Send,
{
    FunctionFlow {
        info,
        func,
        _phantom: std::marker::PhantomData,
    }
}
