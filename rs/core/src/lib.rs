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

//! # Genkit Core
//!
//! This crate provides the core functionalities for the Genkit AI framework in Rust.
//! It includes definitions for actions, flows, contexts, and other essential components.

// Declare modules corresponding to the files we will create.
pub mod action;
pub mod async_utils;
pub mod background_action;
pub mod context;
pub mod error;
pub mod flow;
pub mod registry;
pub mod schema;
pub mod status;
pub mod telemetry;
pub mod tracing;
pub mod utils;

// Re-export key components for easier access.
pub use action::{Action, ActionBuilder, ActionFn, ActionFnArg, ActionResult};
pub use context::ActionContext as Context;
pub use error::{Error, Result};
pub use flow::{define_flow, Flow};
pub use registry::Registry;
