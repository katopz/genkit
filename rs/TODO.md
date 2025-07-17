## TODO
Here is a list of the incomplete items for `rs/core`:

### 1. `reflection.rs` (The Reflection API Server)

The reflection server, which is critical for local development and UI tools, has several significant gaps.

- [ ]   **In-Memory Stores**: The server uses simple, in-memory `HashMap`s for storing traces and flow states. The TypeScript architecture is designed to use pluggable stores (e.g., a `TraceStore` provider), making the current Rust implementation a simplified placeholder.
- [ ]   **Runtime File Creation**: The server does not create a runtime file in the `.genkit/runtimes/` directory. This JSON file is how the Genkit developer UI discovers and connects to the running local server. Its absence means the UI cannot function with the Rust server.

### 2. Tracing (`tracing.rs` and submodules)

- [ ]   **Setting Span Attributes**: The ability to set custom attributes on the *current* active span from anywhere within its execution context is missing. In the TypeScript version, `setCustomMetadataAttributes` allows developers to add context to a span after it has started. The Rust implementation only allows attributes to be set at the moment a span is created with `in_new_span`.
- [ ]   **Firebase Auto-initialization**: The `checkFirebaseMonitoringAutoInit` function, which automatically enables Firebase telemetry based on an environment variable, has not been ported.
