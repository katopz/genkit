# TODO

## rs/core:

### 1. `reflection.rs` (The Reflection API Server)

The reflection server, which is critical for local development and UI tools, has several significant gaps.

- [ ]   **In-Memory Stores**: The server uses simple, in-memory `HashMap`s for storing traces and flow states. The TypeScript architecture is designed to use pluggable stores (e.g., a `TraceStore` provider), making the current Rust implementation a simplified placeholder. // Block by `genkit/genkit-tools`
- [ ]   **Runtime File Creation**: The server does not create a runtime file in the `.genkit/runtimes/` directory. This JSON file is how the Genkit developer UI discovers and connects to the running local server. Its absence means the UI cannot function with the Rust server. // Block by `genkit/genkit-tools`

### 2. Tracing (`tracing.rs` and submodules)

- [ ]   **Firebase Auto-initialization**: The `checkFirebaseMonitoringAutoInit` function, which automatically enables Firebase telemetry based on an environment variable, has not been ported. // Block by `genkit/js/plugins/firebase`

## rs/ai:

### File-Based Prompt Management**
- Block by `dotprompt` (Issue #310)](https://github.com/google/dotprompt/issues/310)
-   **Location**: `genkit/js/ai/src/prompt.ts`
-   **Missing Functions**:
    -   `loadPromptFolder`
    -   `loadPromptFolderRecursively`
    -   `definePartial`
    -   `defineHelper`
-   **Description**: The Rust implementation of prompts in `genkit/rs/ai/src/prompt.rs` is missing the utilities for loading prompt templates and Handlebars partials/helpers from the filesystem. The TypeScript version allows developers to organize their prompts into `.prompt` files and load them dynamically, which is a very useful feature for managing larger projects. This capability is currently not available in the Rust version.

## rs/genkit:

1. Dotprompt and Prompt Templating
This is the most significant difference. The JavaScript version has a sophisticated system for loading prompts from `.prompt` files, which can include Handlebars templating.

*   **Missing In**: `genkit/rs/genkit/src/prompt.rs` and `genkit/rs/genkit/src/lib.rs`
*   **Description**: The JS version can load prompts from a `promptDir`. This allows for separating prompt engineering from application code. The Rust version currently requires prompts to be defined entirely in code. Helper functions for Handlebars templating like `defineHelper` and `definePartial` are also missing.
*   **Impact**: High. This affects how prompts are managed and makes it harder to share prompts between JS and Rust projects. The test files `prompts_as_tool_test.rs` and `prompts_define_prompt_default_model_ref_test.rs` are explicitly blocked by this.

#### 2. Using a Prompt as a Tool
As a direct result of the incomplete dotprompt implementation, prompts cannot be used as tools.

*   **Missing In**: `genkit/rs/genkit/src/prompt.rs`
*   **Description**: The JS version allows a prompt to be converted into a tool and used in `generate` calls.
*   **Impact**: Medium. This limits the composability of prompts. The test file `rs/genkit/tests/prompts_as_tool_test.rs` is empty and marked as blocked pending the dotprompt implementation.
