## TODO
Here is a list of the incomplete items for `rs/core`:

### 1. `reflection.rs` (The Reflection API Server)

The reflection server, which is critical for local development and UI tools, has several significant gaps.

- [ ]   **/api/actions Endpoint**: The implementation of this endpoint calls `registry.list_actions()`, which only returns actions that have already been explicitly loaded into memory. The correct implementation should use a method like `listResolvableActions()` (which is also missing, see below) to include actions that can be dynamically loaded by plugins. This is essential for tools to see all available models, flows, etc.
- [ ]   **/api/notify Endpoint**: The logic to check for API version mismatches between the CLI and the runtime library is missing. The TypeScript version includes warnings to the user if versions are incompatible.
- [ ]   **In-Memory Stores**: The server uses simple, in-memory `HashMap`s for storing traces and flow states. The TypeScript architecture is designed to use pluggable stores (e.g., a `TraceStore` provider), making the current Rust implementation a simplified placeholder.
- [ ]   **Runtime File Creation**: The server does not create a runtime file in the `.genkit/runtimes/` directory. This JSON file is how the Genkit developer UI discovers and connects to the running local server. Its absence means the UI cannot function with the Rust server.

### 2. `registry.rs`

The registry is missing key features required for dynamic action loading and schema management.

- [ ]   **Schema Registration**: The registry lacks the functionality to register and look up schemas. The corresponding helper functions `define_schema` and `define_json_schema` in `schema.rs` are stubbed out with `unimplemented!()`.

### 3. Tracing (`tracing.rs` and submodules)

- [ ]   **Setting Span Attributes**: The ability to set custom attributes on the *current* active span from anywhere within its execution context is missing. In the TypeScript version, `setCustomMetadataAttributes` allows developers to add context to a span after it has started. The Rust implementation only allows attributes to be set at the moment a span is created with `in_new_span`.
- [ ]   **Firebase Auto-initialization**: The `checkFirebaseMonitoringAutoInit` function, which automatically enables Firebase telemetry based on an environment variable, has not been ported.
