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

//! # Genkit Actions
//!
//! This module defines the core `Action` abstraction for the Genkit framework,
//! providing a Rust-idiomatic equivalent of `action.ts`. An `Action` is a
//! self-describing, callable unit of work that forms the building block
//! for more complex flows and tool integrations.

use crate::async_utils::{channel, Channel};
use crate::context::{self, ActionContext};
use crate::error::{Error, Result};
use crate::registry::{ActionType, Registry};
use crate::schema::{parse_schema, ProvidedSchema};
use crate::status::StatusCode;

use crate::schema::schema_for;
use crate::tracing::{self, TraceContext, TRACE_PATH};
use async_trait::async_trait;
use futures::{Future, Stream, StreamExt};
use schemars::{JsonSchema, Schema};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Metadata describing a Genkit `Action`.
#[derive(Debug, Clone, Serialize)]
pub struct ActionMetadata {
    pub action_type: ActionType,
    pub name: String,
    pub description: Option<String>,
    pub subtype: Option<String>,
    pub input_schema: Option<Schema>,
    pub output_schema: Option<Schema>,
    pub stream_schema: Option<Schema>,
    pub metadata: HashMap<String, Value>,
}

impl ActionMetadata {
    /// Removes a key from the metadata map.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.metadata.remove(key)
    }
}

/// The result of a non-streaming action execution, including telemetry data.
#[derive(Debug, Serialize)]
pub struct ActionResult<O> {
    pub result: O,
    pub telemetry: TelemetryInfo,
}

/// Telemetry information associated with a single action execution.
#[derive(Debug, Serialize, Clone)]
pub struct TelemetryInfo {
    pub trace_id: String,
    pub span_id: String,
}

/// The response from a streaming action.
pub struct StreamingResponse<O, S> {
    /// A stream that yields chunks of type `S` or an `Error`.
    pub stream: Pin<Box<dyn Stream<Item = Result<S>> + Send>>,
    /// A future that resolves to the final output of type `O` when the stream is complete.
    pub output: Pin<Box<dyn Future<Output = Result<O>> + Send>>,
}

/// A callback for receiving streaming chunks.
pub type StreamingCallback<S> = Arc<dyn Fn(Result<S, Error>) + Send + Sync>;

/// Options for running an `Action`.
pub struct ActionRunOptions<S: Send + 'static> {
    /// A callback to receive streaming chunks.
    pub on_chunk: Option<StreamingCallback<S>>,
    /// Additional runtime context data.
    pub context: Option<ActionContext>,
    /// Additional attributes for telemetry spans.
    pub telemetry_labels: Option<HashMap<String, String>>,
    /// A token to signal cancellation.
    pub abort_signal: Option<CancellationToken>,
}

impl<S: Send + 'static> Default for ActionRunOptions<S> {
    fn default() -> Self {
        Self {
            on_chunk: None,
            context: None,
            telemetry_labels: None,
            abort_signal: None,
        }
    }
}

/// Arguments passed to the function that implements an action's logic.
#[derive(Default)]
pub struct ActionFnArg<S: Send + 'static> {
    /// Indicates whether the caller requested a streaming response.
    pub streaming_requested: bool,
    /// A channel sender for pushing streaming chunks back to the caller.
    pub chunk_sender: Channel<S>,
    /// Additional runtime context, such as authentication information.
    pub context: Option<ActionContext>,
    /// OpenTelemetry trace context.
    pub trace: TraceContext,
    /// A token that signals when the action should be cancelled.
    pub abort_signal: CancellationToken,
}

/// A trait that defines the executable logic of a Genkit action.
///
/// This allows for different action functions (e.g., closures) to be stored
/// and called polymorphically.
#[async_trait]
pub trait ActionFn<I, O, S: Send + 'static>: Send + Sync {
    async fn run(&self, input: I, args: ActionFnArg<S>) -> Result<O>;
}

/// Blanket implementation of `ActionFn` for any suitable async closure.
#[async_trait]
impl<F, Fut, I, O, S> ActionFn<I, O, S> for F
where
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
    I: Send + 'static,
    O: Send + 'static,
    S: Send + 'static,
{
    async fn run(&self, input: I, args: ActionFnArg<S>) -> Result<O> {
        (self)(input, args).await
    }
}

/// A self-describing, callable unit of work.
///
/// `Action` is the central abstraction in Genkit. It encapsulates metadata
/// and an executable function, and is managed by a `Registry`.
pub struct Action<I, O, S> {
    pub meta: Arc<ActionMetadata>,
    pub func: Arc<dyn ActionFn<I, O, S>>,
}

impl<I, O, S> Clone for Action<I, O, S> {
    fn clone(&self) -> Self {
        Self {
            meta: self.meta.clone(),
            func: self.func.clone(),
        }
    }
}

impl<I, O, S> Debug for Action<I, O, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Action")
            .field("meta", &self.meta)
            // Note: We can't print the function itself, so we just indicate its presence.
            .field("func", &"Arc<dyn ActionFn>")
            .finish()
    }
}

impl<I, O, S> Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
{
    /// Provides mutable access to the action's metadata.
    ///
    /// This will clone the metadata if it's shared across multiple references.
    pub fn metadata_mut(&mut self) -> &mut ActionMetadata {
        Arc::make_mut(&mut self.meta)
    }

    /// Attaches an action to a registry. In this Rust implementation, this is a
    /// conceptual operation that currently does nothing.
    pub fn attach(self, _registry: &Registry) -> Self {
        self
    }

    /// Executes the action with the given input and returns the final result.
    ///
    /// This method can handle streaming via the `on_chunk` callback in `ActionRunOptions`.
    pub async fn run(
        &self,
        input: I,
        options: Option<ActionRunOptions<S>>,
    ) -> Result<ActionResult<O>> {
        let mut opts = options.unwrap_or_else(|| ActionRunOptions {
            on_chunk: None,
            context: None,
            telemetry_labels: None,
            abort_signal: None,
        });

        let telemetry_labels = opts.telemetry_labels.map(|labels| {
            labels
                .into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect()
        });

        let (result, telemetry) =
            tracing::in_new_span(self.meta.name.clone(), telemetry_labels, |trace_context| {
                TRACE_PATH.scope(Vec::new(), async {
                    let (chunk_tx, mut chunk_rx) = channel();

                    let on_chunk_task = if let Some(on_chunk) = opts.on_chunk {
                        let task = tokio::spawn(async move {
                            while let Some(chunk_result) = chunk_rx.next().await {
                                on_chunk(
                                    chunk_result.map_err(|e| Error::new_internal(e.to_string())),
                                );
                            }
                        });
                        Some(task)
                    } else {
                        None
                    };

                    let args = ActionFnArg {
                        streaming_requested: on_chunk_task.is_some(),
                        chunk_sender: chunk_tx.clone(),
                        context: opts.context.clone(),
                        trace: trace_context,
                        abort_signal: opts.abort_signal.take().unwrap_or_default(),
                    };

                    let fut = self.func.run(input, args);

                    let run_result = if let Some(ctx) = opts.context {
                        context::run_with_context(ctx, fut).await
                    } else {
                        fut.await
                    };

                    chunk_tx.close();
                    if let Some(task) = on_chunk_task {
                        task.await.unwrap();
                    }

                    run_result
                })
            })
            .await?;

        // TODO: Add output schema validation.

        Ok(ActionResult { result, telemetry })
    }

    /// Executes the action and provides a streaming response.
    pub fn stream(
        &self,
        input: I,
        options: Option<ActionRunOptions<S>>,
    ) -> StreamingResponse<O, S> {
        let (chunk_tx, chunk_rx) = channel();
        let chunk_tx_clone = chunk_tx.clone();

        let mut opts = options.unwrap_or_default();
        opts.on_chunk = Some(Arc::new(move |chunk_result| {
            let _ = match chunk_result {
                Ok(chunk) => chunk_tx.send(Ok(chunk)),
                Err(e) => chunk_tx.send(Err(e)),
            };
        }));

        let self_clone = self.clone();
        let future = Box::pin(async move {
            let result = self_clone.run(input, Some(opts)).await.map(|ar| ar.result);
            chunk_tx_clone.close();
            result
        });

        let stream = chunk_rx.map(|res| match res {
            Ok(inner_result) => inner_result,
            Err(e) => Err(Error::new_internal(e.to_string())),
        });

        StreamingResponse {
            stream: Box::pin(stream),
            output: future,
        }
    }

    /// Executes the action with a raw JSON value, handling parsing.
    ///
    /// This is useful for invoking an action from a generic source, like an HTTP request.
    pub async fn run_http(
        &self,
        input: Value,
        options: Option<ActionRunOptions<S>>,
    ) -> Result<ActionResult<O>> {
        let parsed_input: I = match self.meta.input_schema.clone() {
            Some(schema_def) => parse_schema(input, ProvidedSchema::FromType(schema_def))?,
            None => serde_json::from_value(input).map_err(|e| {
                Error::new_user_facing(
                    StatusCode::InvalidArgument,
                    format!("Failed to parse input: {}", e),
                    None,
                )
            })?,
        };
        self.run(parsed_input, options).await
    }
}

/// Builder for creating a new `Action`.
pub struct ActionBuilder<I, O, S, F> {
    action_type: ActionType,
    name: String,
    description: Option<String>,
    metadata: Option<HashMap<String, Value>>,
    func: F,
    _marker: std::marker::PhantomData<(I, O, S)>,
}

impl<I, O, S, F, Fut> ActionBuilder<I, O, S, F>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + 'static,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
{
    pub fn new(action_type: ActionType, name: impl Into<String>, func: F) -> Self {
        Self {
            action_type,
            name: name.into(),
            description: None,
            metadata: None,
            func,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Finalizes the build and creates the `Action` struct.
    pub fn build(self) -> Action<I, O, S> {
        let subtype = match self.action_type {
            ActionType::Flow => Some("flow".to_string()),
            ActionType::Tool => Some("tool".to_string()),
            _ => None,
        };
        let meta = Arc::new(ActionMetadata {
            action_type: self.action_type,
            name: self.name,
            description: self.description,
            input_schema: Some(schema_for::<I>()),
            output_schema: Some(schema_for::<O>()),
            stream_schema: Some(schema_for::<S>()),
            metadata: self.metadata.unwrap_or_default(),
            subtype,
        });

        Action {
            meta,
            func: Arc::new(self.func),
        }
    }
}

/// Creates a detached action.
///
/// A detached action is not registered with the framework automatically and
/// can be used directly or registered manually later. This is the Rust
/// equivalent of `detachedAction` in Genkit TS.
pub fn detached_action<I, O, S, F, Fut>(
    action_type: ActionType,
    name: impl Into<String>,
    func: F,
) -> Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + 'static,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
{
    ActionBuilder::new(action_type, name, func).build()
}

/// Defines an action and registers it with the given registry.
///
/// This is the primary way to create and register actions that will be visible
/// to the Genkit framework.
pub fn define_action<I, O, S, F, Fut>(
    registry: &mut Registry,
    action_type: ActionType,
    name: impl Into<String>,
    func: F,
) -> Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
    Action<I, O, S>: crate::registry::ErasedAction + 'static,
{
    let name_str = name.into();
    let action = ActionBuilder::new(action_type, name_str.clone(), func).build();
    registry
        .register_action(name_str.as_str(), action.clone())
        .expect("Failed to register action"); // Or handle error more gracefully
    action
}
