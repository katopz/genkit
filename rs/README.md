# Genkit for Rust

This directory contains the Rust implementation of the Genkit AI Framework. It provides a robust, idiomatic Rust foundation for building production-ready, instrumented AI applications.

The workspace is divided into two main crates:
- `genkit-core`: The core library providing fundamental abstractions like Actions, Flows, and the Registry.
- `genkit-ai`: The AI library containing primitives for generative models, document management, embedding, retrieval, and more.

## ⚙️ Setup & Installation

To get started with Genkit for Rust, you need a standard Rust development environment.

1.  **Install Rust**: If you don't have Rust installed, the recommended way is to use `rustup`. You can find instructions at [rustup.rs](https://rustup.rs/).

    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2.  **Verify Installation**: Ensure `cargo`, the Rust package manager, is available in your path.

    ```sh
    cargo --version
    ```

## 🛠️ Development & Building

This project is structured as a Cargo workspace, containing multiple related crates.

-   **Build all crates**: To build the entire workspace, run the following command from this (`/rs`) directory:
    ```sh
    cargo build
    ```

-   **Build a specific crate**: To build only a single crate (e.g., `genkit-core`), use the `-p` flag:
    ```sh
    cargo build -p genkit-core
    ```

-   **Create a release build**: For an optimized build, use the `--release` flag:
    ```sh
    cargo build --release
    ```

## 🧪 Testing

You can run all tests for the workspace from the root of the `/rs` directory.

-   **Run all tests**:
    ```sh
    cargo test
    ```

-   **Run tests for a specific crate**:
    ```sh
    cargo test -p genkit-ai
    ```

> **Note**: Some tests related to global state (like initializing loggers) may require being run in a single thread to avoid interference. If you encounter issues, you can run tests with: `cargo test -- --test-threads=1`.

## ✨ Features

This Rust implementation provides a solid foundation with many of the core features found in the Genkit JavaScript library.

-   **Core Framework**:
    -   **Actions**: Self-describing, callable units of work, with support for detached definitions.
    -   **Flows**: Orchestration primitive for creating multi-step AI logic with built-in tracing.
    -   **Registry**: Centralized management for all Genkit components.
    -   **Context Management**: Securely pass authentication and other data through your flows.
    -   **Error Handling**: A standardized error system for robust applications.

-   **AI Primitives**:
    -   **`generate`**: The primary function for interacting with generative models, including support for long-running background jobs (`generateOperation`).
    -   **`chat`**: High-level API for creating and managing stateful, multi-turn conversations.
    -   **`embed`**: Abstractions for converting documents into vector embeddings.
    -   **`retrieve`/`index`**: Standardized interfaces for document indexers and retrievers.
    -   **`rerank`**: Support for re-ranking retrieved documents for improved relevance.
    -   **`tool`**: Define tools that models can use to interact with external systems, including support for resumable executions (`defineInterrupt`) and runtime tool definition (`dynamicTool`).
    -   **`prompt`**: A basic `define_prompt` function for creating executable prompts.

-   **Model Middleware**:
    -   Built-in middleware like `downloadRequestMedia` to automatically download and inline media from URLs.

-   **Output Formatting**:
    -   Built-in formatters to constrain model output to specific formats, including JSON, JSONL, Enum, Text, and Array.
    -   Utilities like `parsePartialJson` for extracting structured data from incomplete model output.

-   **Observability & Development**:
    -   **Telemetry**: Integration with OpenTelemetry for tracing flow and action execution, with support for custom and multi-span processors.
    -   **Reflection Server**: A local server for inspecting actions, running flows, and viewing telemetry data during development.

## 📝 TODO (Missing Features)

This section tracks features from the JavaScript implementation that are not yet available in the Rust version. Contributions are welcome!

-   **AI**
    -   [ ] **Dotprompt Integration**:
        -   [ ] Load prompts from the filesystem (`loadPromptFolder`).
        -   [ ] Define partials (`definePartial`) and helpers (`defineHelper`).
        > **Note**: This is blocked by the completion of the [Rust implementation in `dotprompt` (Issue #310)](https://github.com/google/dotprompt/issues/310).