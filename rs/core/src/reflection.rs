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

//! # Genkit Reflection API Server
//!
//! This module provides an API server for inspecting and interacting with a
//! Genkit application in a development environment. It is the Rust equivalent
// of `reflection.ts`.

use crate::registry::Registry;
use crate::status::StatusCode;
use crate::tracing::types::TraceData;
use anyhow::{Context, Result};
use futures::stream::StreamExt;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Body, Bytes, Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response};
use hyper_util::rt::tokio::TokioIo;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

// In-memory stores for reflection API. A persistent implementation would replace these.
static TRACE_STORE: Lazy<Mutex<HashMap<String, TraceData>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static FLOW_STATE_STORE: Lazy<Mutex<HashMap<String, serde_json::Value>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunActionRequest {
    key: String,
    input: Option<serde_json::Value>,
}

#[allow(unused)]
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NotifyRequest {
    telemetry_server_url: Option<String>,
    reflection_api_spec_version: Option<i32>,
}

type BoxBody = Pin<Box<dyn Body<Data = Bytes, Error = Infallible> + Send + 'static>>;

fn full_body(body: impl Into<Bytes>) -> BoxBody {
    Box::pin(Full::new(body.into()).map_err(|never| match never {}))
}

/// Options for configuring the reflection server.
#[derive(Debug, Clone)]
pub struct ReflectionServerOptions {
    /// Port to run the server on.
    pub port: u16,
    /// Body size limit for the server.
    pub body_limit: usize,
    /// Configured environments.
    pub configured_envs: Vec<String>,
}

impl Default for ReflectionServerOptions {
    fn default() -> Self {
        Self {
            port: 3100,
            body_limit: 30 * 1024 * 1024, // 30mb
            configured_envs: vec!["dev".to_string()],
        }
    }
}

/// A handle to a running reflection server.
pub struct ReflectionServerHandle {
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    pub port: u16,
}

impl ReflectionServerHandle {
    /// Stops the running server.
    pub async fn stop(self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.lock().unwrap().take() {
            tx.send(())
                .map_err(|_| anyhow::anyhow!("Failed to send shutdown signal"))
        } else {
            // Already shut down, which is not an error.
            Ok(())
        }
    }
}

/// Starts the reflection server.
pub async fn start(
    registry: Arc<Registry>,
    options: Option<ReflectionServerOptions>,
) -> Result<ReflectionServerHandle> {
    let opts = options.unwrap_or_default();
    let addr = SocketAddr::from(([127, 0, 0, 1], opts.port));
    let listener = TcpListener::bind(addr)
        .await
        .context("Failed to bind TCP listener")?;
    let actual_port = listener.local_addr()?.port();

    let opts_arc = Arc::new(opts);
    let (tx, mut shutdown_rx) = oneshot::channel::<()>();
    let shutdown_tx = Arc::new(Mutex::new(Some(tx)));
    let shutdown_tx_for_handle = shutdown_tx.clone();

    let server_task = tokio::spawn(async move {
        let registry = registry.clone();
        loop {
            tokio::select! {
                res = listener.accept() => {
                    let (stream, _) = match res {
                        Ok(res) => res,
                        Err(e) => {
                            eprintln!("Error accepting connection: {}", e);
                            continue;
                        }
                    };

                    let registry_clone = registry.clone();
                    let opts_clone = opts_arc.clone();
                    let shutdown_tx_clone = shutdown_tx.clone();
                    tokio::task::spawn(async move {
                        let service = service_fn(move |req| {
                            handle_request(
                                req,
                                registry_clone.clone(),
                                opts_clone.clone(),
                                shutdown_tx_clone.clone(),
                            )
                        });

                        if let Err(err) = http1::Builder::new()
                            .serve_connection(TokioIo::new(stream), service)
                            .await
                        {
                            eprintln!("Error serving connection: {:?}", err);
                        }
                    });
                },
                _ = &mut shutdown_rx => {
                    println!("Reflection server shutting down.");
                    break;
                },
            }
        }
    });

    tokio::spawn(async move {
        let _ = server_task.await;
    });

    println!(
        "Genkit reflection server listening on http://127.0.0.1:{}",
        actual_port
    );

    Ok(ReflectionServerHandle {
        shutdown_tx: shutdown_tx_for_handle,
        port: actual_port,
    })
}

/// Main router for all incoming reflection API requests.
async fn handle_request(
    req: Request<Incoming>,
    registry: Arc<Registry>,
    options: Arc<ReflectionServerOptions>,
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
) -> Result<Response<BoxBody>, Infallible> {
    let response = match (req.method(), req.uri().path()) {
        (&Method::GET, "/api/__health") => ok_response("OK"),
        (&Method::POST, "/api/__quitquitquit") => {
            if let Some(tx) = shutdown_tx.lock().unwrap().take() {
                let _ = tx.send(());
            }
            ok_response("OK")
        }
        (&Method::GET, "/api/envs") => json_response(&options.configured_envs),
        (&Method::POST, "/api/notify") => handle_notify(req).await,
        (&Method::GET, "/api/actions") => handle_list_actions(registry).await,
        (&Method::POST, "/api/runAction") => handle_run_action(req, registry).await,
        (method, path) if path.starts_with("/api/envs/") => {
            handle_store_request(method, path).await
        }
        _ => not_found_response(),
    };
    Ok(response)
}

// Specific handler implementations

async fn handle_notify(req: Request<Incoming>) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::InvalidArgument,
                format!("Failed to read body: {}", e),
            )
        }
    };
    let notify_req: NotifyRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::InvalidArgument,
                format!("Invalid request body: {}", e),
            )
        }
    };

    if let Some(url) = notify_req.telemetry_server_url {
        crate::tracing::exporter::set_telemetry_server_url(url);
    }
    ok_response("OK")
}

async fn handle_list_actions(registry: Arc<Registry>) -> Response<BoxBody> {
    let actions = registry.list_actions().await;
    let api_actions: std::collections::HashMap<_, _> = actions
        .iter()
        .map(|(key, action)| {
            let meta = action.metadata();
            (
                key,
                json!({ "key": key, "name": action.name(), "description": meta.description, "inputSchema": meta.input_schema, "outputSchema": meta.output_schema, "metadata": meta }),
            )
        })
        .collect();
    json_response(&api_actions)
}

async fn handle_run_action(req: Request<Incoming>, registry: Arc<Registry>) -> Response<BoxBody> {
    let is_streaming = req.uri().query().unwrap_or("").contains("stream=true");
    let body = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return error_response(
                StatusCode::InvalidArgument,
                format!("Failed to read body: {}", e),
            )
        }
    };
    let run_req: RunActionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::InvalidArgument,
                format!("Invalid request body: {}", e),
            )
        }
    };
    let action = match registry.lookup_action(&run_req.key).await {
        Some(a) => a,
        None => {
            return error_response(
                StatusCode::NotFound,
                format!("Action '{}' not found", run_req.key),
            )
        }
    };
    let input = run_req.input.unwrap_or(serde_json::Value::Null);

    if is_streaming {
        let stream_resp = match action.stream_http_json(input, None) {
            Ok(resp) => resp,
            Err(e) => return error_response(e.as_status().code, e.to_string()),
        };
        let body_stream = stream_resp.stream.map(|result| {
            let frame_str = match result {
                Ok(value) => serde_json::to_string(&json!({ "result": value })).unwrap(),
                Err(e) => serde_json::to_string(&json!({ "error": e.as_status() })).unwrap(),
            };
            Ok(Frame::data(Bytes::from(format!("{}\n", frame_str))))
        });
        let body: BoxBody = Box::pin(StreamBody::new(body_stream));
        Response::builder()
            .header("Content-Type", "application/x-json-stream")
            .body(body)
            .unwrap()
    } else {
        match action.run_http_json(input, None).await {
            Ok(result_value) => {
                if let Some(trace_id) = result_value
                    .get("telemetry")
                    .and_then(|t| t.get("traceId"))
                    .and_then(|id| id.as_str())
                {
                    let trace_id = trace_id.to_string();
                    let trace_data = TraceData {
                        trace_id: trace_id.clone(),
                        display_name: Some(action.name().to_string()),
                        ..Default::default()
                    };
                    TRACE_STORE
                        .lock()
                        .unwrap()
                        .insert(trace_id.clone(), trace_data);

                    if action.metadata().action_type == crate::registry::ActionType::Flow {
                        let flow_state = json!({
                            "flowId": trace_id,
                            "name": action.name(),
                            "status": "done",
                        });
                        FLOW_STATE_STORE
                            .lock()
                            .unwrap()
                            .insert(trace_id, flow_state);
                    }
                }
                json_response(&result_value)
            }
            Err(e) => error_response(e.as_status().code, e.to_string()),
        }
    }
}

async fn handle_store_request(method: &Method, path: &str) -> Response<BoxBody> {
    if method != Method::GET {
        return not_found_response();
    }
    let path_parts: Vec<&str> = path.split('/').collect();
    // Expected path: /api/envs/{env}/traces/{traceId} or /api/envs/{env}/flowStates/{flowId}
    if path_parts.len() < 5 {
        return not_found_response();
    }
    let store_type = path_parts[4];
    let id = path_parts.get(5);

    match store_type {
        "traces" => {
            let store = TRACE_STORE.lock().unwrap();
            handle_get_request(id, &store)
        }
        "flowStates" => {
            let store = FLOW_STATE_STORE.lock().unwrap();
            handle_get_request(id, &store)
        }
        _ => not_found_response(),
    }
}

fn handle_get_request<T: Serialize>(
    id: Option<&&str>,
    store: &HashMap<String, T>,
) -> Response<BoxBody> {
    match id {
        None => {
            // List all items
            let items: Vec<&T> = store.values().collect();
            json_response(&items)
        }
        Some(id_str) => match store.get(*id_str) {
            Some(item) => json_response(item),
            None => error_response(StatusCode::NotFound, "Item not found".to_string()),
        },
    }
}

// Helper functions for building responses

fn json_response<T: Serialize>(data: &T) -> Response<BoxBody> {
    match serde_json::to_string(data) {
        Ok(json) => Response::builder()
            .status(hyper::StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(full_body(json))
            .unwrap(),
        Err(e) => error_response(StatusCode::Internal, format!("Serialization error: {}", e)),
    }
}

fn ok_response(body: &'static str) -> Response<BoxBody> {
    Response::new(full_body(body))
}

fn error_response(status_code: StatusCode, message: String) -> Response<BoxBody> {
    let status_struct = crate::status::Status {
        code: status_code,
        message,
        is_blocked: false,
        details: None,
    };
    Response::builder()
        .status(status_code.to_http_status())
        .header("Content-Type", "application/json")
        .body(full_body(
            serde_json::to_string(&json!({ "error": status_struct })).unwrap(),
        ))
        .unwrap()
}

fn not_found_response() -> Response<BoxBody> {
    error_response(StatusCode::NotFound, "Not Found".to_string())
}
