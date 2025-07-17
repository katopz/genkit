## TODO
Here is a list of the incomplete items for `rs/core`:

### 1. `background_action.rs`

The implementation for background actions is a skeleton. The core logic of registering the component actions with the registry is missing.

- [x]   **`defineBackgroundAction`**: The builder for a background action (`BackgroundActionBuilder`) creates the `start`, `check`, and `cancel` actions, but it does not register them with the Genkit registry. The code contains a comment `// TODO: Register the actions with the registry`, confirming this is unfinished.
- [x]   **`lookupBackgroundAction`**: This function is missing entirely. Without it, there is no way to retrieve and interact with a defined background action from the registry.

### 2. `reflection.rs` (The Reflection API Server)

The reflection server, which is critical for local development and UI tools, has several significant gaps.

*   **/api/actions Endpoint**: The implementation of this endpoint calls `registry.list_actions()`, which only returns actions that have already been explicitly loaded into memory. The correct implementation should use a method like `listResolvableActions()` (which is also missing, see below) to include actions that can be dynamically loaded by plugins. This is essential for tools to see all available models, flows, etc.
*   **/api/notify Endpoint**: The logic to check for API version mismatches between the CLI and the runtime library is missing. The TypeScript version includes warnings to the user if versions are incompatible.
*   **In-Memory Stores**: The server uses simple, in-memory `HashMap`s for storing traces and flow states. The TypeScript architecture is designed to use pluggable stores (e.g., a `TraceStore` provider), making the current Rust implementation a simplified placeholder.
*   **Runtime File Creation**: The server does not create a runtime file in the `.genkit/runtimes/` directory. This JSON file is how the Genkit developer UI discovers and connects to the running local server. Its absence means the UI cannot function with the Rust server.

### 3. `registry.rs`

The registry is missing key features required for dynamic action loading and schema management.

*   **`listResolvableActions` Method**: This entire method, which is used to list all possible actions a plugin can provide (even if not yet loaded), is missing. This is a key part of the developer experience provided by the reflection API.
*   **Schema Registration**: The registry lacks the functionality to register and look up schemas. The corresponding helper functions `define_schema` and `define_json_schema` in `schema.rs` are stubbed out with `unimplemented!()`.

### 4. `action.rs`

*   **`defineActionAsync` Function**: The TypeScript core library includes a `defineActionAsync` function, which allows for defining actions that are resolved lazily via a `Promise`. This asynchronous registration pattern has not been implemented in Rust.

### 5. `flow.rs`

*   **`run` Function**: The `run` utility for creating a step within a flow is incomplete. The Rust version is missing the overload `run(name, input, func)`, which allows passing an explicit input to the step that is then automatically recorded in the trace data. The current version only supports `run(name, func)`.

### 6. Tracing (`tracing.rs` and submodules)

*   **Setting Span Attributes**: The ability to set custom attributes on the *current* active span from anywhere within its execution context is missing. In the TypeScript version, `setCustomMetadataAttributes` allows developers to add context to a span after it has started. The Rust implementation only allows attributes to be set at the moment a span is created with `in_new_span`.
*   **Firebase Auto-initialization**: The `checkFirebaseMonitoringAutoInit` function, which automatically enables Firebase telemetry based on an environment variable, has not been ported.
