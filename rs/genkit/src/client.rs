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

//! # Genkit Flow Client
//!
//! This module provides a simple, browser-safe client library for remotely
//! running and streaming deployed Genkit flows. It is the Rust equivalent of
//! `beta/client`.

use crate::error::{Error, Result};
use bytes::Bytes;
use futures_util::Stream;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::str::FromStr;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

const FLOW_STREAM_DELIMITER: &[u8] = b"\n\n";

//
// Types for request/response serialization
//

#[derive(Serialize)]
struct FlowRequest<I> {
    data: I,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FlowResponse<O> {
    Success { result: O },
    Failure { error: Value },
}

#[derive(Deserialize)]
struct StreamChunk<S> {
    message: Option<S>,
    result: Option<Value>,
    error: Option<StreamError>,
}

#[derive(Deserialize, Debug)]
struct StreamError {
    status: String,
    message: String,
    details: Option<String>,
}

//
// Public Structs for Client Interaction
//

/// Parameters for `run_flow`.
pub struct RunFlowParams<I> {
    /// URL of the deployed flow.
    pub url: String,
    /// Flow input.
    pub input: Option<I>,
    /// A map of HTTP headers to be added to the HTTP call.
    pub headers: Option<HashMap<String, String>>,
}

/// Parameters for `stream_flow`.
pub struct StreamFlowParams<I> {
    /// URL of the deployed flow.
    pub url: String,
    /// Flow input.
    pub input: Option<I>,
    /// A map of HTTP headers to be added to the HTTP call.
    pub headers: Option<HashMap<String, String>>,
}

/// The response from a `stream_flow` call.
pub struct StreamFlowResponse<O, S> {
    /// A stream of response chunks of type `S`.
    pub stream: Pin<Box<dyn Stream<Item = Result<S>> + Send>>,
    /// A future that resolves to the final output of type `O`.
    pub output: tokio::task::JoinHandle<Result<O>>,
}

/// Invoke a deployed flow over HTTP(s).
///
/// ## Example
///
/// ```rust,no_run
/// use genkit::client::{run_flow, RunFlowParams};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize)]
/// struct MyInput {
///     name: String,
/// }
///
/// #[derive(Deserialize, Debug)]
/// struct MyOutput {
///     message: String,
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), genkit::error::Error> {
///     let response = run_flow::<_, MyOutput>(RunFlowParams {
///         url: "https://my-flow-deployed-url".to_string(),
///         input: Some(MyInput { name: "Genkit".to_string() }),
///         headers: None,
///     }).await?;
///     println!("Response: {:?}", response);
///     Ok(())
/// }
/// ```
pub async fn run_flow<I, O>(params: RunFlowParams<I>) -> Result<O>
where
    I: Serialize,
    O: DeserializeOwned,
{
    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(user_headers) = params.headers {
        for (key, value) in user_headers {
            let header_name = HeaderName::from_str(&key).map_err(|e| {
                Error::new_internal(format!("Invalid header name '{}': {}", key, e))
            })?;
            let header_value = HeaderValue::from_str(&value).map_err(|e| {
                Error::new_internal(format!("Invalid header value for '{}': {}", key, e))
            })?;
            headers.insert(header_name, header_value);
        }
    }

    let request_body = FlowRequest { data: params.input };
    let body_bytes = serde_json::to_vec(&request_body)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request body: {}", e)))?;

    let response = client
        .post(params.url)
        .headers(headers)
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| Error::new_internal(format!("HTTP request failed: {}", e)))?;

    if response.status().as_u16() != 200 {
        return Err(Error::new_internal(format!(
            "Server returned: {}: {}",
            response.status(),
            response.text().await.unwrap_or_default()
        )));
    }

    let wrapped_result: FlowResponse<O> = response
        .json()
        .await
        .map_err(|e| Error::new_internal(format!("Failed to deserialize response body: {}", e)))?;

    match wrapped_result {
        FlowResponse::Success { result } => Ok(result),
        FlowResponse::Failure { error } => Err(Error::new_internal(format!(
            "Flow returned error: {}",
            error
        ))),
    }
}

/// Invoke and stream the response from a deployed flow.
pub fn stream_flow<I, O, S>(params: StreamFlowParams<I>) -> StreamFlowResponse<O, S>
where
    I: Serialize + Send + 'static,
    O: DeserializeOwned + Send + 'static,
    S: DeserializeOwned + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Result<S>>(100);

    let handle = tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));

        if let Some(user_headers) = params.headers {
            for (key, value) in user_headers {
                let header_name = HeaderName::from_str(&key).map_err(|e| {
                    Error::new_internal(format!("Invalid header name '{}': {}", key, e))
                })?;
                let header_value = HeaderValue::from_str(&value).map_err(|e| {
                    Error::new_internal(format!("Invalid header value for '{}': {}", key, e))
                })?;
                headers.insert(header_name, header_value);
            }
        }

        let request_body = FlowRequest { data: params.input };
        let body_bytes = serde_json::to_vec(&request_body)
            .map_err(|e| Error::new_internal(format!("Failed to serialize request body: {}", e)))?;

        let response = client
            .post(params.url)
            .headers(headers)
            .body(body_bytes)
            .send()
            .await
            .map_err(|e| Error::new_internal(format!("HTTP request failed: {}", e)))?;

        if response.status() != 200 {
            return Err(Error::new_internal(format!(
                "Server returned: {}: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();

        while let Some(item) = stream.next().await {
            let chunk: Bytes =
                item.map_err(|e| Error::new_internal(format!("chunk error: {}", e)))?;
            buffer.extend_from_slice(&chunk);

            while let Some(delimiter_pos) = buffer
                .windows(FLOW_STREAM_DELIMITER.len())
                .position(|window| window == FLOW_STREAM_DELIMITER)
            {
                let message_bytes = buffer.drain(..delimiter_pos).collect::<Vec<u8>>();
                // Also remove the delimiter itself
                buffer.drain(..FLOW_STREAM_DELIMITER.len());

                let message_str = String::from_utf8(message_bytes)
                    .map_err(|e| Error::new_internal(format!("message_bytes error: {}", e)))?;
                if let Some(data_str) = message_str.strip_prefix("data: ") {
                    let stream_chunk: StreamChunk<S> = serde_json::from_str(data_str)
                        .map_err(|e| Error::new_internal(format!("stream_chunk error: {}", e)))?;

                    if let Some(msg) = stream_chunk.message {
                        if tx.send(Ok(msg)).await.is_err() {
                            // Receiver dropped, stop processing.
                            return Err(Error::new_internal("Stream receiver closed."));
                        }
                    } else if let Some(result_value) = stream_chunk.result {
                        let final_output: O =
                            serde_json::from_value(result_value).map_err(|e| {
                                Error::new_internal(format!("final_output error: {}", e))
                            })?;
                        return Ok(final_output);
                    } else if let Some(err_detail) = stream_chunk.error {
                        return Err(Error::new_internal(format!(
                            "{}: {}\n{}",
                            err_detail.status,
                            err_detail.message,
                            err_detail.details.unwrap_or_default()
                        )));
                    } else {
                        return Err(Error::new_internal(format!(
                            "Unknown chunk format: {}",
                            data_str
                        )));
                    }
                }
            }
        }

        Err(Error::new_internal(
            "Stream did not terminate with a result.".to_string(),
        ))
    });

    StreamFlowResponse {
        stream: Box::pin(ReceiverStream::new(rx)),
        output: handle,
    }
}
