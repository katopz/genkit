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

//! # Common Utilities for Vertex AI
//!
//! This module provides shared structs and functions for the Vertex AI plugin,
//! such as configuration options and client authentication helpers.

use gcp_auth::{provider, CustomServiceAccount, TokenProvider};
use genkit_core::error::{Error, Result};
use serde::Deserialize;
use std::env;
use std::sync::Arc;

/// Options for configuring the Vertex AI plugin.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexAIPluginOptions {
    pub project_id: Option<String>,
    pub location: Option<String>,
    pub service_account: Option<String>,
}

/// The resolved parameters needed to interact with Vertex AI APIs.
pub struct DerivedParams {
    pub project_id: String,
    pub location: String,
    pub token_provider: Arc<dyn TokenProvider>,
}

/// Gets the GCP project ID from options or the environment.
async fn get_project_id(
    options: &VertexAIPluginOptions,
    token_provider: &Arc<dyn TokenProvider>,
) -> Result<String> {
    if let Some(id) = &options.project_id {
        return Ok(id.clone());
    }
    if let Ok(id) = env::var("GCLOUD_PROJECT") {
        if !id.is_empty() {
            return Ok(id);
        }
    }
    if let Ok(id) = env::var("GOOGLE_CLOUD_PROJECT") {
        if !id.is_empty() {
            return Ok(id);
        }
    }
    // Let gcp_auth try to discover it from credentials.
    token_provider
        .project_id()
        .await
        .map(|id| id.to_string())
        .map_err(|e| Error::new_internal(format!("Failed to determine GCP project ID: {}", e)))
}

/// Gets the GCP location from options or the environment.
fn get_location(options: &VertexAIPluginOptions) -> Result<String> {
    if let Some(loc) = &options.location {
        return Ok(loc.clone());
    }
    if let Ok(loc) = env::var("GCLOUD_LOCATION") {
        if !loc.is_empty() {
            return Ok(loc);
        }
    }
    // Default location as a fallback.
    Ok("us-central1".to_string())
}

/// Resolves the necessary parameters for making Vertex AI calls.
pub async fn get_derived_params(options: &VertexAIPluginOptions) -> Result<DerivedParams> {
    let token_provider: Arc<dyn TokenProvider> = if let Some(key_path) = &options.service_account {
        let sa = CustomServiceAccount::from_file(key_path)
            .map_err(|e| Error::new_internal(format!("Failed to load service account: {}", e)))?;
        Arc::new(sa)
    } else {
        provider()
            .await
            .map_err(|e| Error::new_internal(format!("Failed to create auth provider: {}", e)))?
    };

    let project_id = get_project_id(options, &token_provider).await?;
    let location = get_location(options)?;

    Ok(DerivedParams {
        project_id,
        location,
        token_provider,
    })
}
