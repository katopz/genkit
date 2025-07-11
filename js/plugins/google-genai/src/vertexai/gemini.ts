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

import {
  ActionMetadata,
  Genkit,
  GenkitError,
  modelActionMetadata,
  z,
} from 'genkit';
import {
  GenerationCommonConfigDescriptions,
  GenerationCommonConfigSchema,
  ModelAction,
  ModelInfo,
  ModelMiddleware,
  ModelReference,
  getBasicUsageStats,
  modelRef,
} from 'genkit/model';
import { downloadRequestMedia } from 'genkit/model/middleware';
import { runInNewSpan } from 'genkit/tracing';
import {
  fromGeminiCandidate,
  toGeminiFunctionModeEnum,
  toGeminiMessage,
  toGeminiSystemInstruction,
  toGeminiTool,
} from '../common/converters';
import { checkModelName, cleanSchema, modelName } from '../common/utils';
import {
  generateContent,
  generateContentStream,
  getVertexAIUrl,
} from './client';
import { toGeminiSafetySettings } from './converters';
import {
  ClientOptions,
  Content,
  GenerateContentRequest,
  GenerateContentResponse,
  GoogleSearchRetrieval,
  GoogleSearchRetrievalTool,
  Model,
  Tool,
  ToolConfig,
  VertexPluginOptions,
} from './types';

export const SafetySettingsSchema = z.object({
  category: z.enum([
    /** The harm category is unspecified. */
    'HARM_CATEGORY_UNSPECIFIED',
    /** The harm category is hate speech. */
    'HARM_CATEGORY_HATE_SPEECH',
    /** The harm category is dangerous content. */
    'HARM_CATEGORY_DANGEROUS_CONTENT',
    /** The harm category is harassment. */
    'HARM_CATEGORY_HARASSMENT',
    /** The harm category is sexually explicit content. */
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
  ]),
  threshold: z.enum([
    'BLOCK_LOW_AND_ABOVE',
    'BLOCK_MEDIUM_AND_ABOVE',
    'BLOCK_ONLY_HIGH',
    'BLOCK_NONE',
  ]),
});

const VertexRetrievalSchema = z.object({
  datastore: z
    .object({
      projectId: z.string().describe('Google Cloud Project ID.').optional(),
      location: z
        .string()
        .describe('Google Cloud region e.g. us-central1.')
        .optional(),
      dataStoreId: z
        .string()
        .describe(
          'The data store id, when project id and location are provided as ' +
            'separate options. Alternatively, the full path to the data ' +
            'store should be provided in the form: "projects/{project}/' +
            'locations/{location}/collections/default_collection/dataStores/{data_store}".'
        ),
    })
    .describe('Vertex AI Search data store details'),
  disableAttribution: z
    .boolean()
    .describe(
      'Disable using the search data in detecting grounding attribution. This ' +
        'does not affect how the result is given to the model for generation.'
    )
    .optional(),
});

const GoogleSearchRetrievalSchema = z.object({
  disableAttribution: z
    .boolean()
    .describe(
      'Disable using the search data in detecting grounding attribution. This ' +
        'does not affect how the result is given to the model for generation.'
    )
    .optional(),
});

/**
 * Zod schema of Gemini model options.
 * Please refer to: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig, for further information.
 */
export const GeminiConfigSchema = GenerationCommonConfigSchema.extend({
  temperature: z
    .number()
    .min(0.0)
    .max(2.0)
    .describe(
      GenerationCommonConfigDescriptions.temperature +
        ' The default value is 1.0.'
    )
    .optional(),
  topP: z
    .number()
    .min(0)
    .max(1.0)
    .describe(
      GenerationCommonConfigDescriptions.topP + ' The default value is 0.95.'
    )
    .optional(),
  location: z
    .string()
    .describe('Google Cloud region e.g. us-central1.')
    .optional(),

  /**
   * Safety filter settings. See: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters#configurable-filters
   *
   * E.g.
   *
   * ```js
   * config: {
   *   safetySettings: [
   *     {
   *       category: 'HARM_CATEGORY_HATE_SPEECH',
   *       threshold: 'BLOCK_LOW_AND_ABOVE',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
   *       threshold: 'BLOCK_MEDIUM_AND_ABOVE',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_HARASSMENT',
   *       threshold: 'BLOCK_ONLY_HIGH',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
   *       threshold: 'BLOCK_NONE',
   *     },
   *   ],
   * }
   * ```
   */
  safetySettings: z
    .array(SafetySettingsSchema)
    .describe(
      'Adjust how likely you are to see responses that could be harmful. ' +
        'Content is blocked based on the probability that it is harmful.'
    )
    .optional(),

  /**
   * Vertex retrieval options.
   *
   * E.g.
   *
   * ```js
   *   config: {
   *     vertexRetrieval: {
   *       datastore: {
   *         projectId: 'your-cloud-project',
   *         location: 'us-central1',
   *         collection: 'your-collection',
   *       },
   *       disableAttribution: true,
   *     }
   *   }
   * ```
   */
  vertexRetrieval: VertexRetrievalSchema.describe(
    'Retrieve from Vertex AI Search data store for grounding ' +
      'generative responses.'
  ).optional(),

  /**
   * Google Search retrieval options.
   *
   * ```js
   *   config: {
   *     googleSearchRetrieval: {
   *       disableAttribution: true,
   *     }
   *   }
   * ```
   */
  googleSearchRetrieval: GoogleSearchRetrievalSchema.describe(
    'Retrieve public web data for grounding, powered by Google Search.'
  ).optional(),

  /**
   * Function calling options.
   *
   * E.g. forced tool call:
   *
   * ```js
   *   config: {
   *     functionCallingConfig: {
   *       mode: 'ANY',
   *     }
   *   }
   * ```
   */
  functionCallingConfig: z
    .object({
      mode: z.enum(['MODE_UNSPECIFIED', 'AUTO', 'ANY', 'NONE']).optional(),
      allowedFunctionNames: z.array(z.string()).optional(),
    })
    .describe(
      'Controls how the model uses the provided tools (function declarations). ' +
        'With AUTO (Default) mode, the model decides whether to generate a ' +
        'natural language response or suggest a function call based on the ' +
        'prompt and context. With ANY, the model is constrained to always ' +
        'predict a function call and guarantee function schema adherence. ' +
        'With NONE, the model is prohibited from making function calls.'
    )
    .optional(),
}).passthrough();
export type GeminiConfigSchemaType = typeof GeminiConfigSchema;
/**
 * Gemini model configuration options.
 *
 * E.g.
 * ```js
 *   config: {
 *     temperature: 0.9,
 *     maxOutputTokens: 300,
 *     safetySettings: [
 *       {
 *         category: 'HARM_CATEGORY_HATE_SPEECH',
 *         threshold: 'BLOCK_LOW_AND_ABOVE',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
 *         threshold: 'BLOCK_MEDIUM_AND_ABOVE',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_HARASSMENT',
 *         threshold: 'BLOCK_ONLY_HIGH',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
 *         threshold: 'BLOCK_NONE',
 *       },
 *     ],
 *     functionCallingConfig: {
 *       mode: 'ANY',
 *     }
 *   }
 * ```
 */
export type GeminiConfig = z.infer<GeminiConfigSchemaType>;

// This contains all the Gemini config schema types
type ConfigSchemaType = GeminiConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = GeminiConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertexai/${name}`,
    version: name,
    configSchema,
    info: info ?? {
      supports: {
        multiturn: true,
        media: true,
        tools: true,
        toolChoice: true,
        systemRole: true,
        constrained: 'no-tools',
      },
    },
  });
}

export const GENERIC_MODEL = commonRef('gemini');

export const KNOWN_MODELS = {
  'gemini-2.0-flash': commonRef('gemini-2.0-flash'),
  'gemini-2.0-flash-001': commonRef('gemini-2.0-flash-001'),
  'gemini-2.0-flash-lite': commonRef('gemini-2.0-flash-lite'),
  'gemini-2.0-flash-lite-preview-02-05': commonRef(
    'gemini-2.0-flash-lite-preview-02-05'
  ),
  'gemini-2.0-pro-exp-02-05': commonRef('gemini-2.0-pro-exp-02-05'),
  'gemini-2.5-pro-exp-03-25': commonRef('gemini-2.5-pro-exp-03-25'),
  'gemini-2.5-pro-preview-03-25': commonRef('gemini-2.5-pro-preview-03-25'),
  'gemini-2.5-flash-preview-04-17': commonRef('gemini-2.5-flash-preview-04-17'),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS;
export type GeminiModelName = `gemini-${string}`;
export function isGeminiModelName(value?: string): value is GeminiModelName {
  return !!value?.startsWith('gemini-') && !value.includes('embedding');
}

export function model(
  version: string,
  options: GeminiConfig = {}
): ModelReference<typeof GeminiConfigSchema> {
  const name = checkModelName(version);

  return modelRef({
    name: `vertexai/${name}`,
    version: name,
    config: options,
    configSchema: GeminiConfigSchema,
    info: {
      ...GENERIC_MODEL.info,
    },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  const KNOWN_DECOMISSIONED_MODELS = [
    'gemini-pro-vision',
    'gemini-pro',
    'gemini-ultra',
    'gemini-ultra-vision',
  ];

  return models
    .filter(
      (m) =>
        isGeminiModelName(modelName(m.name)) &&
        !KNOWN_DECOMISSIONED_MODELS.includes(modelName(m.name) || '')
    )
    .map((m) => {
      const ref = model(m.name);
      return modelActionMetadata({
        name: ref.name,
        info: ref.info,
        configSchema: ref.configSchema,
      });
    });
}

export function defineKnownModels(
  ai: Genkit,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
) {
  for (const name of Object.keys(KNOWN_MODELS)) {
    defineModel(ai, name, clientOptions, pluginOptions);
  }
}

/**
 * Define a Vertex AI Gemini model.
 */
export function defineModel(
  ai: Genkit,
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): ModelAction {
  const ref = model(name);
  const middlewares: ModelMiddleware[] = [];
  if (ref.info?.supports?.media) {
    // the gemini api doesn't support downloading media from http(s)
    middlewares.push(
      downloadRequestMedia({
        maxBytes: 1024 * 1024 * 20,
        filter: (part) => {
          try {
            const url = new URL(part.media.url);
            if (
              // Gemini can handle these URLs
              ['www.youtube.com', 'youtube.com', 'youtu.be'].includes(
                url.hostname
              )
            )
              return false;
          } catch {}
          return true;
        },
      })
    );
  }

  return ai.defineModel(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
      use: middlewares,
    },
    async (request, sendChunk) => {
      // Make a copy of messages to avoid side-effects
      const messages = structuredClone(request.messages);
      if (messages.length === 0) throw new Error('No messages provided.');

      // Handle system instructions separately
      let systemInstruction: Content | undefined = undefined;
      const systemMessage = messages.find((m) => m.role === 'system');
      if (systemMessage) {
        messages.splice(messages.indexOf(systemMessage), 1);
        systemInstruction = toGeminiSystemInstruction(systemMessage);
      }

      const requestConfig = request.config as GeminiConfig;

      const {
        functionCallingConfig,
        version: versionFromConfig,
        googleSearchRetrieval,
        tools: toolsFromConfig,
        vertexRetrieval,
        location, // location can be overridden via config, take it out.
        safetySettings,
        ...restOfConfig
      } = requestConfig;

      const tools: Tool[] = [];
      if (request.tools?.length) {
        tools.push({
          functionDeclarations: request.tools.map(toGeminiTool),
        });
      }

      let toolConfig: ToolConfig | undefined;
      if (functionCallingConfig) {
        toolConfig = {
          functionCallingConfig: {
            allowedFunctionNames: functionCallingConfig.allowedFunctionNames,
            mode: toGeminiFunctionModeEnum(functionCallingConfig.mode),
          },
        };
      } else if (request.toolChoice) {
        toolConfig = {
          functionCallingConfig: {
            mode: toGeminiFunctionModeEnum(request.toolChoice),
          },
        };
      }

      // Cannot use tools and function calling at the same time
      const jsonMode =
        (request.output?.format === 'json' || !!request.output?.schema) &&
        tools.length === 0;

      if (toolsFromConfig) {
        tools.push(...(toolsFromConfig as any[]));
      }

      if (googleSearchRetrieval) {
        // Gemini 1.5 models use googleSearchRetrieval, newer models use googleSearch.
        if (ref.name.startsWith('vertexai/gemini-1.5')) {
          tools.push({
            googleSearchRetrieval:
              googleSearchRetrieval as GoogleSearchRetrieval,
          } as GoogleSearchRetrievalTool);
        } else {
          tools.push({
            googleSearch: googleSearchRetrieval as GoogleSearchRetrieval,
          } as GoogleSearchRetrievalTool);
        }
      }

      if (vertexRetrieval) {
        const _projectId =
          vertexRetrieval.datastore.projectId || clientOptions.projectId;
        const _location =
          vertexRetrieval.datastore.location || clientOptions.location;
        const _dataStoreId = vertexRetrieval.datastore.dataStoreId;
        const datastore = `projects/${_projectId}/locations/${_location}/collections/default_collection/dataStores/${_dataStoreId}`;
        tools.push({
          retrieval: {
            vertexAiSearch: {
              datastore,
            },
            disableAttribution: vertexRetrieval.disableAttribution,
          },
        });
      }

      const generateContentRequest: GenerateContentRequest = {
        systemInstruction,
        generationConfig: {
          ...restOfConfig,
          candidateCount: request.candidates || undefined,
          responseMimeType: jsonMode ? 'application/json' : undefined,
        },
        tools,
        toolConfig,
        safetySettings: toGeminiSafetySettings(safetySettings),
        contents: messages.map((message) => toGeminiMessage(message, ref)),
      };

      const modelVersion = versionFromConfig || (ref.version as string);

      if (jsonMode && request.output?.constrained) {
        generateContentRequest.generationConfig!.responseSchema = cleanSchema(
          request.output.schema
        );
      }

      const callGemini = async () => {
        let response: GenerateContentResponse;

        // Handle streaming and non-streaming responses
        if (sendChunk) {
          const result = await generateContentStream(
            modelVersion,
            generateContentRequest,
            clientOptions
          );

          for await (const item of result.stream) {
            (item as GenerateContentResponse).candidates?.forEach(
              (candidate) => {
                const c = fromGeminiCandidate(candidate);
                sendChunk({
                  index: c.index,
                  content: c.message.content,
                });
              }
            );
          }
          response = await result.response;
        } else {
          response = await generateContent(
            modelVersion,
            generateContentRequest,
            clientOptions
          );
        }

        if (!response.candidates?.length) {
          throw new GenkitError({
            status: 'FAILED_PRECONDITION',
            message: 'No valid candidates returned.',
          });
        }

        const candidateData = response.candidates.map((c) =>
          fromGeminiCandidate(c)
        );

        return {
          candidates: candidateData,
          custom: response,
          usage: {
            ...getBasicUsageStats(request.messages, candidateData),
            inputTokens: response.usageMetadata?.promptTokenCount,
            outputTokens: response.usageMetadata?.candidatesTokenCount,
            totalTokens: response.usageMetadata?.totalTokenCount,
            cachedContentTokens:
              response.usageMetadata?.cachedContentTokenCount,
          },
        };
      };

      // If debugTraces is enabled, we wrap the actual model call with a span,
      // add raw API params as for input.
      const msg = toGeminiMessage(messages[messages.length - 1], ref);
      return pluginOptions?.experimental_debugTraces
        ? await runInNewSpan(
            ai.registry,
            {
              metadata: {
                name: sendChunk ? 'sendMessageStream' : 'sendMessage',
              },
            },
            async (metadata) => {
              metadata.input = {
                apiEndpoint: getVertexAIUrl({
                  includeProjectAndLocation: false,
                  resourcePath: '',
                  clientOptions,
                }),
                cache: {},
                model: modelVersion,
                generateContentOptions: generateContentRequest,
                parts: msg.parts,
                clientOptions,
              };
              const response = await callGemini();
              metadata.output = response.custom;
              return response;
            }
          )
        : await callGemini();
    }
  );
}

export const TEST_ONLY = { KNOWN_MODELS };
