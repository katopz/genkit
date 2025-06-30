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

/// A simple client library for remotely running/streaming deployed Genkit flows.
///
/// This module provides functions to call Genkit flows that are exposed over HTTP.
use crate::error::{Error, Result};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::str::FromStr;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const FLOW_STREAM_DELIMITER: &str = "\n\n";

/// Parameters for `run_flow`.
pub struct RunFlowParams<'a, T: Serialize> {
    /// URL of the deployed flow.
    pub url: String,
    /// Flow input, must be serializable.
    pub input: &'a T,
    /// A map of HTTP headers to be added to the HTTP call.
    pub headers: Option<HashMap<String, String>>,
}

/// Invoke a deployed flow over HTTP(s).
///
/// ## Example
/// ```rust,no_run
/// use genkit::beta::client::{run_flow, RunFlowParams};
/// use serde::{Deserialize, Serialize};
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
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let params = RunFlowParams {
///         url: "http://localhost:3400/myFlow".to_string(),
///         input: &MyInput { name: "World".to_string() },
///         headers: None,
///     };
///
///     let response = run_flow::<MyOutput>(params).await?;
///     println!("Response: {:?}", response);
///
///     Ok(())
/// }
/// ```
pub async fn run_flow<O>(params: RunFlowParams<'_, impl Serialize>) -> Result<O>
where
    O: DeserializeOwned,
{
    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(custom_headers) = params.headers {
        for (key, value) in custom_headers {
            let header_name =
                HeaderName::from_str(&key).map_err(|e| Error::new_internal(e.to_string()))?;
            let header_value =
                HeaderValue::from_str(&value).map_err(|e| Error::new_internal(e.to_string()))?;
            headers.insert(header_name, header_value);
        }
    }

    let request_body = json!({ "data": params.input });

    let response = client
        .post(params.url)
        .headers(headers)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| Error::new_internal(e.to_string()))?;

    if response.status() != 200 {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(Error::new_internal(format!(
            "Server returned: {}: {}",
            status, text
        )));
    }

    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum FlowResponse<T> {
        Result { result: T },
        Error { error: Value },
    }

    let wrapped_result: FlowResponse<O> = response
        .json()
        .await
        .map_err(|e| Error::new_internal(e.to_string()))?;

    match wrapped_result {
        FlowResponse::Result { result } => Ok(result),
        FlowResponse::Error { error } => Err(Error::new_internal(format!(
            "Flow execution failed: {}",
            error
        ))),
    }
}

/// Represents the response from a `stream_flow` call.
pub struct StreamFlowResponse<O, S> {
    /// A future that resolves to the final output of the flow.
    pub output: tokio::task::JoinHandle<Result<O>>,
    /// A stream of intermediate chunks from the flow.
    pub stream: ReceiverStream<S>,
}

/// Parameters for `stream_flow`.
pub struct StreamFlowParams<'a, T: Serialize> {
    /// URL of the deployed flow.
    pub url: String,
    /// Flow input, must be serializable.
    pub input: &'a T,
    /// A map of HTTP headers to be added to the HTTP call.
    pub headers: Option<HashMap<String, String>>,
}

/// Invoke and stream response from a deployed flow.
///
/// ## Example
/// ```rust,no_run
/// use genkit::beta::client::{stream_flow, StreamFlowParams};
/// use serde::{Deserialize, Serialize};
/// use futures_util::StreamExt;
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
/// #[derive(Deserialize, Debug)]
/// struct MyChunk {
///    part: String,
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let params = StreamFlowParams {
///         url: "http://localhost:3400/myStreamingFlow".to_string(),
///         input: &MyInput { name: "World".to_string() },
///         headers: None,
///     };
///
///     let mut response = stream_flow::<MyOutput, MyChunk>(params).await?;
///
///     while let Some(chunk) = response.stream.next().await {
///         println!("Chunk: {:?}", chunk);
///     }
///
///     let output = response.output.await??;
///     println!("Final output: {:?}", output);
///
///     Ok(())
/// }
/// ```
pub async fn stream_flow<O, S>(
    params: StreamFlowParams<'_, impl Serialize>,
) -> Result<StreamFlowResponse<O, S>>
where
    O: DeserializeOwned + Send + 'static,
    S: DeserializeOwned + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<S>(100);
    let stream = ReceiverStream::new(rx);

    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(custom_headers) = params.headers {
        for (key, value) in custom_headers {
            let header_name =
                HeaderName::from_str(&key).map_err(|e| Error::new_internal(e.to_string()))?;
            let header_value =
                HeaderValue::from_str(&value).map_err(|e| Error::new_internal(e.to_string()))?;
            headers.insert(header_name, header_value);
        }
    }

    let request_body = json!({ "data": params.input });
    let url = params.url;

    let output = tokio::spawn(async move {
        let response = client
            .post(url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::new_internal(e.to_string()))?;

        if response.status() != 200 {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::new_internal(format!(
                "Server returned: {}: {}",
                status, text
            )));
        }

        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StreamChunk<S, O> {
            Message { message: S },
            Result { result: O },
            Error { error: Value },
        }

        while let Some(item) = byte_stream.next().await {
            let chunk = item.map_err(|e| Error::new_internal(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(delimiter_pos) = buffer.find(FLOW_STREAM_DELIMITER) {
                let chunk_str = buffer[..delimiter_pos].to_string();
                buffer.replace_range(..delimiter_pos + FLOW_STREAM_DELIMITER.len(), "");

                if let Some(json_str) = chunk_str.strip_prefix("data: ") {
                    match serde_json::from_str::<StreamChunk<S, O>>(json_str) {
                        Ok(StreamChunk::Message { message }) => {
                            if tx.send(message).await.is_err() {
                                // Receiver was dropped, stop processing.
                                return Err(Error::new_internal(
                                    "Stream processing was cancelled.",
                                ));
                            }
                        }
                        Ok(StreamChunk::Result { result }) => {
                            return Ok(result);
                        }
                        Ok(StreamChunk::Error { error }) => {
                            return Err(Error::new_internal(format!("Flow error: {}", error)));
                        }
                        Err(e) => {
                            return Err(Error::new_internal(format!(
                                "Unknown chunk format: {}, error: {}",
                                json_str, e
                            )));
                        }
                    }
                }
            }
        }

        Err(Error::new_internal("Stream did not terminate correctly"))
    });

    Ok(StreamFlowResponse { output, stream })
}
