# Genkit for Rust

This directory contains the Rust implementation of the Genkit AI Framework. It provides a robust, idiomatic Rust foundation for building production-ready, instrumented AI applications.

The workspace is divided into several crates:

-   `genkit`: The primary, public-facing crate that developers should use. It re-exports the core APIs from the other crates.
-   `genkit-core`: The core library providing fundamental abstractions like Actions, Flows, and the Registry.
-   `genkit-ai`: The AI library containing primitives for generative models, document management, embedding, retrieval, and more.
-   `plugins`: Contains plugins for integrating with various services.
    -   `vertexai`: Provides support for Google Cloud Vertex AI models (Gemini, Imagen, etc.).
    -   `dev-local-vectorstore`: A simple, file-based vector store for local development.
-   `testapps`: Example applications demonstrating how to use the framework.

## ⚙️ Setup & Installation

To get started with Genkit for Rust, you need a standard Rust development environment.

1.  **Install Rust**: If you don't have Rust installed, the recommended way is to use `rustup`. You can find instructions at [rustup.rs](https://rustup.rs/).

    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2.  **Add `genkit` to your dependencies**:
    ```toml
    # In your Cargo.toml
    [dependencies]
    genkit = { path = "path/to/genkit/rs/genkit" }
    # Add any plugins you need
    genkit-vertexai = { path = "path/to/genkit/rs/plugins/vertexai" }
    ```

## 🛠️ Development & Building

This project is structured as a Cargo workspace, containing multiple related crates.

-   **Build all crates**: To build the entire workspace, run the following command from this (`/rs`) directory:
    ```sh
    cargo build
    ```

-   **Run an example**: To run the `basic-gemini` test application:
    ```sh
    cd testapps/basic-gemini
    RUST_LOG=debug GOOGLE_APPLICATION_CREDENTIALS="../../service-account.json" cargo run
    ```
    (See the `README.md` in the `basic-gemini` directory for setup instructions.)

## ✨ Features

This Rust implementation provides a solid foundation with many of the core features found in the Genkit JavaScript library.

-   **Core Framework**:
    -   **Actions**: Self-describing, callable units of work.
    -   **Flows**: Orchestration primitive for creating multi-step AI logic with built-in tracing.
    -   **Registry**: Centralized management for all Genkit components.
    -   **Context Management**: Securely pass authentication and other data through your flows.
    -   **Error Handling**: A standardized error system for robust applications.

-   **AI Primitives**:
    -   **`generate`**: The primary function for interacting with generative models, including support for streaming (`generate_stream`) and long-running background jobs (`generate_operation`).
    -   **`chat`**: High-level API for creating and managing stateful, multi-turn conversations.
    -   **`embed`**: Abstractions for converting documents into vector embeddings.
    -   **`retrieve`**: Standardized interfaces for document indexers and retrievers.
    -   **`rerank`**: Support for re-ranking retrieved documents for improved relevance.
    -   **`tool`**: Define tools that models can use to interact with external systems, including support for resumable executions (`define_interrupt`) and runtime tool definition (`dynamic_tool`).
    -   **`prompt`**: A `define_prompt` function for creating executable, templated prompts.

-   **Plugins**:
    -   **Vertex AI**: Comprehensive support for Google Cloud Vertex AI, including Gemini and Imagen models.
    -   **Local Development Vector Store**: A simple, file-based vector store for getting started quickly.

-   **Output Formatting**:
    -   Built-in formatters to constrain model output to specific formats, including JSON, JSONL, Enum, Text, and Array.
    -   Utilities like `extract_json` for extracting structured data from model output.

-   **Observability & Development**:
    -   **Telemetry**: Integration with OpenTelemetry for tracing flow and action execution.
    -   **Reflection Server**: A local server for inspecting actions, running flows, and viewing telemetry data during development.
