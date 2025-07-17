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
use crate::registry::{ActionType, ErasedAction, Registry};
use crate::schema::{parse_schema, schema_for, ProvidedSchema};
use crate::status::StatusCode;
use crate::tracing::{self, TraceContext};
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::{Future, Stream, StreamExt};
use schemars::{JsonSchema, Schema};
use serde::Deserialize;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// A simple, cloneable error representation for use in shared futures.
#[derive(Clone, Debug)]
pub struct CloneableError {
    message: String,
}

impl fmt::Display for CloneableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl From<Error> for CloneableError {
    fn from(err: Error) -> Self {
        Self {
            message: err.to_string(),
        }
    }
}

// --- ActionName Enum ---

/// Represents the name of an action, which can be simple or namespaced.
#[derive(Debug, Clone)]
pub enum ActionName {
    /// A simple, global action name.
    Simple(String),
    /// A name scoped to a specific plugin.
    Namespaced {
        plugin_id: String,
        action_id: String,
    },
}

impl From<String> for ActionName {
    fn from(s: String) -> Self {
        ActionName::Simple(s)
    }
}

impl From<&str> for ActionName {
    fn from(s: &str) -> Self {
        ActionName::Simple(s.to_string())
    }
}

impl Display for ActionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionName::Simple(s) => write!(f, "{}", s),
            ActionName::Namespaced {
                plugin_id,
                action_id,
            } => write!(f, "{}/{}", plugin_id, action_id),
        }
    }
}

// --- Middleware Structs ---

/// A function that represents the next step in a middleware chain.
/// The future returned by a middleware `next` function.
pub type ActionMiddlewareNextFuture<O> =
    Pin<Box<dyn Future<Output = Result<ActionResult<O>>> + Send>>;

/// The trait object for a middleware `next` function.
pub type ActionMiddlewareNextFn<I, O, S> =
    dyn Fn(I, Option<ActionRunOptions<S>>) -> ActionMiddlewareNextFuture<O> + Send + Sync;

/// A function that represents the next step in a middleware chain.
pub struct ActionMiddlewareNext<I, O, S: Send + 'static> {
    pub f: Arc<ActionMiddlewareNextFn<I, O, S>>,
    pub _phantom: std::marker::PhantomData<(I, O, S)>,
}

impl<I, O, S: Send + 'static> Clone for ActionMiddlewareNext<I, O, S> {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// A middleware function for an action.
pub type ActionMiddlewareFn<I, O, S> = dyn Fn(
        I,
        Option<ActionRunOptions<S>>,
        ActionMiddlewareNext<I, O, S>,
    ) -> ActionMiddlewareNextFuture<O>
    + Send
    + Sync;

pub struct ActionMiddleware<I, O, S: Send + 'static> {
    pub f: Arc<ActionMiddlewareFn<I, O, S>>,
    pub _phantom: std::marker::PhantomData<(I, O, S)>,
}

impl<I, O, S: Send + 'static> Clone for ActionMiddleware<I, O, S> {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

// --- Core Action Structs ---

/// Metadata describing a Genkit `Action`.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Clone)]
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
#[derive(Default, Clone)]
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

// --- Action Logic ---

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
pub struct Action<I, O, S: Send + 'static> {
    pub meta: Arc<ActionMetadata>,
    pub func: Arc<dyn ActionFn<I, O, S>>,
    pub middleware: Vec<ActionMiddleware<I, O, S>>,
}

impl<I, O, S: Send + 'static> Clone for Action<I, O, S> {
    fn clone(&self) -> Self {
        Self {
            meta: self.meta.clone(),
            func: self.func.clone(),
            middleware: self.middleware.clone(),
        }
    }
}

impl<I, O, S: Send + 'static> Debug for Action<I, O, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Action")
            .field("meta", &self.meta)
            .field("func", &"Arc<dyn ActionFn>")
            .field("middleware", &self.middleware.len())
            .finish()
    }
}

impl<I, O, S> Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + Serialize + 'static,
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

    /// Executes the action, running any configured middleware first.
    pub async fn run(
        &self,
        input: I,
        options: Option<ActionRunOptions<S>>,
    ) -> Result<ActionResult<O>> {
        if self.middleware.is_empty() {
            return self.run_internal(input, options).await;
        }

        let self_clone = self.clone();
        // The final link in the chain calls the internal run method.
        let mut next: ActionMiddlewareNext<I, O, S> = ActionMiddlewareNext {
            f: Arc::new(move |i, o| {
                let self_clone_for_internal = self_clone.clone();
                Box::pin(async move { self_clone_for_internal.run_internal(i, o).await })
            }),
            _phantom: std::marker::PhantomData,
        };

        // Wrap the `next` function with each middleware, in reverse order.
        for mw in self.middleware.iter().rev() {
            let current_mw = mw.clone();
            let next_in_chain = next.clone();
            next = ActionMiddlewareNext {
                f: Arc::new(move |i, o| (current_mw.f)(i, o, next_in_chain.clone())),
                _phantom: std::marker::PhantomData,
            };
        }

        // Execute the full chain.
        (next.f)(input, options).await
    }

    /// Executes the action with the given input and returns the final result.
    ///
    /// This method can handle streaming via the `on_chunk` callback in `ActionRunOptions`.
    async fn run_internal(
        &self,
        input: I,
        options: Option<ActionRunOptions<S>>,
    ) -> Result<ActionResult<O>> {
        let mut opts = options.unwrap_or_default();
        // If no context is provided in options, try to get it from the task local.
        if opts.context.is_none() {
            opts.context = context::get_context();
        }

        // In TS, all high-level actions are of type 'action', with the real type in the subtype.
        let telemetry_type = "action";
        let mut telemetry_attrs = HashMap::new();
        telemetry_attrs.insert(
            "genkit:type".to_string(),
            Value::String(telemetry_type.to_string()),
        );
        telemetry_attrs.insert(
            "genkit:name".to_string(),
            Value::String(self.meta.name.clone()),
        );
        if let Some(subtype) = &self.meta.subtype {
            telemetry_attrs.insert(
                "genkit:metadata:subtype".to_string(),
                Value::String(subtype.clone()),
            );
        }

        if let Ok(input_str) = serde_json::to_string(&input) {
            telemetry_attrs.insert("genkit:input".to_string(), Value::String(input_str));
        }

        if let Some(labels) = opts.telemetry_labels.take() {
            for (k, v) in labels {
                telemetry_attrs.insert(k, Value::String(v));
            }
        }

        // Per TS implementation, always add context, even if empty.
        let context_for_telemetry = opts.context.clone().unwrap_or_default();
        if let Ok(context_str) = serde_json::to_string(&context_for_telemetry) {
            telemetry_attrs.insert(
                "genkit:metadata:context".to_string(),
                Value::String(context_str),
            );
        }

        let (result, telemetry) = tracing::in_new_span(
            self.meta.name.clone(),
            Some(telemetry_attrs),
            |trace_context| async move {
                let (chunk_tx, mut chunk_rx) = channel();

                let on_chunk_task = if let Some(on_chunk) = opts.on_chunk {
                    let task = tokio::spawn(async move {
                        while let Some(chunk_result) = chunk_rx.next().await {
                            on_chunk(chunk_result.map_err(|e| Error::new_internal(e.to_string())));
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

                // This is the key change to prevent recursion.
                // We check if a context is already set. If so, we are in a nested call
                // and should not try to set it again with `run_with_context`.
                let is_context_already_set = context::get_context().is_some();

                let run_result = if !is_context_already_set {
                    if let Some(ctx) = opts.context {
                        context::run_with_context(ctx, fut).await
                    } else {
                        fut.await
                    }
                } else {
                    fut.await
                };

                chunk_tx.close();
                if let Some(task) = on_chunk_task {
                    task.await.unwrap();
                }

                run_result
            },
        )
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

// --- Action Builder ---

/// Builder for creating a new `Action`.
pub struct ActionBuilder<I, O, S: Send + 'static, F> {
    action_type: ActionType,
    name: ActionName,
    description: Option<String>,
    metadata: Option<HashMap<String, Value>>,
    func: F,
    middleware: Vec<ActionMiddleware<I, O, S>>,
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
    pub fn new(action_type: ActionType, name: impl Into<ActionName>, func: F) -> Self {
        Self {
            action_type,
            name: name.into(),
            description: None,
            metadata: None,
            func,
            middleware: Vec::new(),
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

    pub fn with_middleware(mut self, middleware: Vec<ActionMiddleware<I, O, S>>) -> Self {
        self.middleware = middleware;
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
            name: self.name.to_string(),
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
            middleware: self.middleware,
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
    name: impl Into<ActionName>,
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
    registry: &Registry,
    action_type: ActionType,
    name: impl Into<ActionName>,
    func: F,
) -> Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Serialize + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
    Action<I, O, S>: crate::registry::ErasedAction + 'static,
{
    let action_name = name.into();
    let action = ActionBuilder::new(action_type, action_name, func).build();
    registry
        .register_action(action_type, action.clone())
        .expect("Failed to register action"); // Or handle error more gracefully
    action
}

/// Defines an action from a future-resolved configuration and registers it asynchronously.
pub fn define_action_async<I, O, S, C, F, Fut, FutConfig>(
    registry: &Registry,
    action_type: ActionType,
    name: impl Into<String>,
    config_future: FutConfig,
) -> BoxFuture<'static, Result<Action<I, O, S>>>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
    C: Into<ActionBuilder<I, O, S, F>>,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
    FutConfig: Future<Output = Result<C>> + Send + 'static,
    Action<I, O, S>: ErasedAction + 'static,
{
    let name = name.into();
    let registry_for_registration = registry.clone();
    let registry_for_lookup = registry.clone();

    let action_future = async move {
        let config = config_future.await?;
        let builder: ActionBuilder<I, O, S, F> = config.into();
        let result: Result<Action<I, O, S>> = Ok(builder.build());
        result
    };

    registry_for_registration.register_action_async(action_type, name.clone(), async move {
        let action = action_future.await?;
        let erased_action: Arc<dyn ErasedAction> = Arc::new(action);
        Ok(erased_action)
    });

    Box::pin(async move {
        registry_for_lookup // Use the 'static clone here.
            .lookup_action(&format!("/{}/{}", action_type, name))
            .await
            .and_then(|action| action.as_any().downcast_ref::<Action<I, O, S>>().cloned())
            .ok_or_else(|| Error::new_internal("Failed to resolve or downcast async action"))
    })
}
