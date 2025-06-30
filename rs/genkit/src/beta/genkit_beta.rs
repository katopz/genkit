/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
use crate::Genkit;
use genkit_core::Registry;
use std::ops::{Deref, DerefMut};

// TODO: This should be defined in the parent Genkit module (`lib.rs`).
#[derive(Default)]
pub struct GenkitOptions {}

// The Genkit::new method doesn't exist in Rust as it does in TypeScript.
// The `init` method is async and returns a static ref, which is a different pattern.
// This implementation detail needs to be reconciled between the two languages.
// For now, a private `new` is added to Genkit via an extension trait pattern
// to allow `genkit_beta` to compile. This is a temporary workaround.
trait GenkitExt {
    fn new(options: Option<GenkitOptions>) -> Self;
}

impl GenkitExt for Genkit {
    fn new(_options: Option<GenkitOptions>) -> Self {
        Self {
            registry: Registry::new(),
        }
    }
}

pub type GenkitBetaOptions = GenkitOptions;

/// WARNING: these APIs are considered unstable and subject to frequent breaking changes that may not honor semver.
///
/// Initializes Genkit BETA APIs with a set of options.
///
/// This will create a new Genkit registry, register the provided plugins, stores, and other configuration. This
/// should be called before any flows are registered.
///
/// @beta
pub fn genkit(options: GenkitOptions) -> GenkitBeta {
    GenkitBeta::new(Some(options))
}

/// Genkit BETA APIs.
///
/// @beta
pub struct GenkitBeta {
    inner: Genkit,
}

impl GenkitBeta {
    pub fn new(options: Option<GenkitOptions>) -> Self {
        let genkit = Genkit::new(options);
        // This is a placeholder for how to set the stability.
        // The actual implementation might differ depending on the Genkit struct.
        // genkit.registry.api_stability = Some("beta".to_string());
        Self { inner: genkit }
    }
}

impl Deref for GenkitBeta {
    type Target = Genkit;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for GenkitBeta {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
