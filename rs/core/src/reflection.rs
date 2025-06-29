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

use crate::action::StreamingResponse;
use crate::registry::Registry;
use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Body, Bytes, Frame};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use serde::Deserialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunActionRequest {
    key: String,
    input: Option<serde_json::Value>,
}

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

fn full_body(body: impl Into<Bytes>) -> BoxBody {
    use http_body_util::BodyExt;
    Full::new(body.into())
        .map_err(|never| match never {})
        .boxed()
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
    shutdown_tx: oneshot::Sender<()>,
    pub port: u16,
}

impl ReflectionServerHandle {
    /// Stops the running server.
    pub async fn stop(self) -> Result<()> {
        self.shutdown_tx
            .send(())
            .map_err(|_| anyhow::anyhow!("Failed to send shutdown signal"))
    }
}

/// Starts the reflection server.
///
/// This server exposes an API for development tools to inspect and interact
/// with the running Genkit application.
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

    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

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
                    tokio::task::spawn(async move {
                        let service = service_fn(move |req| {
                            handle_request(req, registry_clone.clone())
                        });

                        if let Err(err) = http1::Builder::new()
                            .serve_connection(stream, service)
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
    // This is a failsafe in case the handle is dropped without `stop` being called.
    tokio::spawn(async move {
        let _ = server_task.await;
    });

    println!(
        "Genkit reflection server listening on http://127.0.0.1:{}",
        actual_port
    );

    Ok(ReflectionServerHandle {
        shutdown_tx,
        port: actual_port,
    })
}

/// Main request handler for the reflection server.
async fn handle_request(
    req: Request<Incoming>,
    registry: Arc<Registry>,
) -> Result<Response<BoxBody>, Infallible> {
    let response = match (req.method(), req.uri().path()) {
        (&Method::GET, "/api/actions") => {
            match registry.list_resolvable_actions().await {
                Ok(actions) => {
                    // Convert to the format expected by the reflection API client.
                    let api_actions: std::collections::HashMap<_, _> = actions
                        .iter()
                        .map(|(key, meta)| {
                            (
                                key,
                                serde_json::json!({
                                    "key": key,
                                    "name": &meta.name,
                                    "description": meta.description,
                                    "inputSchema": meta.input_schema,
                                    "outputSchema": meta.output_schema,
                                    "metadata": &meta.metadata
                                }),
                            )
                        })
                        .collect();

                    let json_body = match serde_json::to_string(&api_actions) {
                        Ok(json) => json,
                        Err(e) => {
                            return Ok(Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(full_body(format!(
                                    "{{\"error\":\"Failed to serialize actions: {}\"}}",
                                    e
                                )))
                                .unwrap());
                        }
                    };
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "application/json")
                        .body(full_body(json_body))
                        .unwrap()
                }
                Err(e) => Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(full_body(format!(
                        "{{\"error\":\"Failed to list actions: {}\"}}",
                        e
                    )))
                    .unwrap(),
            }
        }
        (&Method::POST, "/api/runAction") => {
            let query = req.uri().query().unwrap_or("");
            let is_streaming = query.split('&').any(|p| p == "stream=true");

            // Read and parse the request body
            let body = match req.into_body().collect().await {
                Ok(body) => body.to_bytes(),
                Err(e) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Full::new(Bytes::from(format!(
                            "{{\"error\": \"Failed to read request body: {}\"}}",
                            e
                        ))))
                        .unwrap());
                }
            };

            let run_req: RunActionRequest = match serde_json::from_slice(&body) {
                Ok(req) => req,
                Err(e) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Full::new(Bytes::from(format!(
                            "{{\"error\": \"Invalid request body: {}\"}}",
                            e
                        ))))
                        .unwrap());
                }
            };

            // Look up the action
            let action = match registry.lookup_action(&run_req.key).await {
                Some(action) => action,
                None => {
                    return Ok(Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .body(Full::new(Bytes::from(format!(
                            "{{\"error\": \"Action '{}' not found\"}}",
                            run_req.key
                        ))))
                        .unwrap());
                }
            };

            let input = run_req.input.unwrap_or(serde_json::Value::Null);

            if is_streaming {
                // NOTE: The `ErasedAction` trait currently does not support
                // streaming invocation. This is a placeholder response.
                // A full implementation would require a `stream_http_json` method
                // on the trait to be called here.
                let response_body = "{\"error\":{\"message\":\"Streaming not yet implemented for Rust reflection server\",\"status\":\"UNIMPLEMENTED\"}}\n";
                Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/x-json-stream")
                    .body(Full::new(Bytes::from(response_body)))
                    .unwrap()
            } else {
                // Run the action for a non-streaming request
                match action.run_http_json(input, None).await {
                    Ok(result) => {
                        let response_body = serde_json::json!({ "result": result });
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::to_string(&response_body).unwrap(),
                            )))
                            .unwrap()
                    }
                    Err(e) => {
                        let status = e.as_status();
                        let error_body = serde_json::to_string(&status).unwrap();
                        Response::builder()
                            .status(status.code.to_http_status())
                            .header("Content-Type", "application/json")
                            .body(Full::new(Bytes::from(error_body)))
                            .unwrap()
                    }
                }
            }
        }
        (&Method::GET, "/api/__health") => Response::builder()
            .status(StatusCode::OK)
            .body(full_body("OK"))
            .unwrap(),
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(full_body("Not Found"))
            .unwrap(),
    };
    Ok(response)
}
