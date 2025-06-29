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

//! # Test Helpers
//!
//! Shared utilities for integration tests.

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::sync::oneshot;

// Helper to create and run a mock server for a single test.
pub async fn with_mock_server<F>(handler: F) -> String
where
    F: Fn(Request<Body>) -> Result<Response<Body>, Infallible> + Send + Sync + 'static + Copy,
{
    let addr = SocketAddr::from(([127, 0, 0, 1], 0)); // Port 0 asks OS for a free port
    let make_svc =
        make_service_fn(move |_conn| async move { Ok::<_, Infallible>(service_fn(handler)) });
    let server = Server::bind(&addr).serve(make_svc);
    let url = format!("http://{}", server.local_addr());

    let (tx, rx) = oneshot::channel();
    let graceful = server.with_graceful_shutdown(async {
        rx.await.ok();
    });

    tokio::spawn(async move {
        if let Err(e) = graceful.await {
            eprintln!("server error: {}", e);
        }
    });

    // For simplicity in testing, we'll let the server run for the duration of the test.
    // In a more complex setup, we would return the `tx` sender to shut it down.
    url
}

// Common structs for testing `generate` flows

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct GenerateInput {
    pub prompt: String,
    pub config: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct GenerateOutput {
    pub text: String,
}

// Common handlers

/// A mock handler that simulates an "echo" model.
/// It takes the prompt from the input and constructs a response.
pub async fn echo_handler(req: Request<Body>) -> Result<Response<Body>, Infallible> {
    let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
    let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
    let data: GenerateInput = serde_json::from_value(input["data"].clone()).unwrap();

    let config_str = data
        .config
        .map(|v| serde_json::to_string(&v).unwrap())
        .unwrap_or_else(|| "{}".to_string());

    let response_text = format!("Echo: {}; config: {}", data.prompt, config_str);

    let response_data = serde_json::json!({
        "result": {
            "text": response_text
        }
    });
    Ok(Response::new(Body::from(response_data.to_string())))
}

/// A handler that returns a 500 Internal Server Error.
pub async fn internal_error_handler(_: Request<Body>) -> Result<Response<Body>, Infallible> {
    Ok(Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .body("Internal Error".into())
        .unwrap())
}

/// A handler that returns a static streaming response.
pub async fn simple_streaming_handler(_: Request<Body>) -> Result<Response<Body>, Infallible> {
    let body = "data: {\"message\":\"one\"}\n\ndata: {\"message\":\"two\"}\n\ndata: {\"result\":\"done\"}\n\n";
    let response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .body(Body::from(body))
        .unwrap();
    Ok(response)
}
