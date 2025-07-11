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

import { GoogleAuth, GoogleAuthOptions } from 'google-auth-library';
import {
  CitationMetadata,
  Content,
  FunctionCallingMode,
  GenerateContentCandidate,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerateContentStreamResult,
  GoogleSearchRetrieval,
  GoogleSearchRetrievalTool,
  GroundingMetadata,
  HarmBlockThreshold,
  HarmCategory,
  ImagenInstance,
  ImagenParameters,
  ImagenPredictRequest,
  ImagenPredictResponse,
  ImagenPrediction,
  TaskType,
  TaskTypeSchema,
  Tool,
  ToolConfig,
} from '../common/types';

// This makes it easier to import all types from one place
export {
  FunctionCallingMode,
  HarmBlockThreshold,
  HarmCategory,
  TaskTypeSchema,
  type CitationMetadata,
  type Content,
  type GenerateContentCandidate,
  type GenerateContentRequest,
  type GenerateContentResponse,
  type GenerateContentStreamResult,
  type GoogleSearchRetrieval,
  type GoogleSearchRetrievalTool,
  type GroundingMetadata,
  type ImagenInstance,
  type ImagenParameters,
  type ImagenPredictRequest,
  type ImagenPredictResponse,
  type ImagenPrediction,
  type Tool,
  type ToolConfig,
};

/** Options for Vertex AI plugin configuration */
export interface VertexPluginOptions {
  /** The Google Cloud project id to call. */
  projectId?: string;
  /** The Google Cloud region to call. */
  location: string;
  /** Provide custom authentication configuration for connecting to Vertex AI. */
  googleAuth?: GoogleAuthOptions;
  /** Enables additional debug traces (e.g. raw model API call details). */
  experimental_debugTraces?: boolean;
}

/** Resolved options for use with the client */
export interface ClientOptions {
  location: string;
  projectId: string;
  authClient: GoogleAuth;
}

/**
 * Request options params.
 */
export interface RequestOptions {
  /** timeout in milli seconds. time out value needs to be non negative. */
  timeout?: number;
  /**
   * Version of API endpoint to call (e.g. "v1" or "v1beta"). If not specified,
   * defaults to 'v1beta'.
   */
  apiVersion?: string;
  /**
   * Value for x-goog-api-client header to set on the API request. This is
   * intended for wrapper SDKs to set additional SDK identifiers for the
   * backend.
   */
  apiClient?: string;
  /**
   * Value for custom HTTP headers to set on the HTTP request.
   */
  customHeaders?: Headers;
}

// Vertex AI  model definition
export interface Model {
  name: string;
  launchStage: string;
}

// Vertex AI list models response
export interface ListModelsResponse {
  publisherModels: Model[];
}

// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#request_body
interface TextEmbeddingInstance {
  task_type?: TaskType;
  content: string;
  title?: string;
}

// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api#request_body
interface MultimodalEmbeddingInstance {
  text?: string;
  image?: {
    // Union field can only be one of the following:
    bytesBase64Encoded?: string;
    gcsUri?: string;
    // End of list of possible types for union field.
    mimeType?: string;
  };
  video?: {
    // Union field can only be one of the following:
    bytesBase64Encoded?: string;
    gcsUri?: string;
    // End of list of possible types for union field.
    videoSegmentConfig?: {
      startOffsetSec: number;
      endOffsetSec: number;
      intervalSec: number;
    };
  };
  parameters?: {
    dimension: number;
  };
}

export declare type EmbeddingInstance =
  | TextEmbeddingInstance
  | MultimodalEmbeddingInstance;

export declare interface TextEmbeddingPrediction {
  embeddings: {
    statistics: {
      truncated: boolean;
      token_count: number;
    };
    values: number[];
  };
}

export declare interface VideoEmbedding {
  startOffsetSec: number;
  endOffsetSec: number;
  embedding: number[];
}

export declare interface MultimodalEmbeddingPrediction {
  textEmbedding?: number[];
  imageEmbedding?: number[];
  videoEmbeddings?: VideoEmbedding[];
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function isMultimodalEmbeddingPrediction(
  value: unknown
): value is MultimodalEmbeddingPrediction {
  if (!isObject(value)) {
    return false;
  }
  if (!value.textEmbedding && !value.imageEmbedding && !value.videoEmbeddings) {
    return false;
  }
  if (value.textEmbedding && !Array.isArray(value.textEmbedding)) {
    return false;
  }
  if (value.imageEmbedding && !Array.isArray(value.imageEmbedding)) {
    return false;
  }
  if (value.videoEmbeddings && !Array.isArray(value.videoEmbeddings)) {
    return false;
  }
  if (value.videoEmbeddings) {
    for (const emb of value.videoEmbeddings as Array<unknown>) {
      if (!isObject(emb)) {
        return false;
      }
      if (!emb.embedding || !Array.isArray(emb.embedding)) {
        return false;
      }
    }
  }

  return true;
}

export declare type EmbeddingPrediction =
  | TextEmbeddingPrediction
  | MultimodalEmbeddingPrediction;

export declare interface EmbedContentRequest {
  instances: EmbeddingInstance[];
  parameters: EmbedContentConfig;
}

export declare interface EmbedContentResponse {
  predictions: EmbeddingPrediction[];
}

/** Optional parameters for the embed content method. */
export declare interface EmbedContentConfig {
  /** Type of task for which the embedding will be used. */
  taskType?: string;
  /** Title for the text. Only applicable when TaskType is
      `RETRIEVAL_DOCUMENT`.
       */
  title?: string;
  /** Reduced dimension for the output embedding. If set,
      excessive values in the output embedding are truncated from the end.
      Supported by newer models since 2024 only. You cannot set this value if
      using the earlier model (`models/embedding-001`).
       */
  outputDimensionality?: number;
  /** The MIME type of the input. */
  mimeType?: string;
  /** Vertex API only. Whether to silently truncate inputs longer than
      the max sequence length. If this option is set to false, oversized inputs
      will lead to an INVALID_ARGUMENT error, similar to other text APIs.
       */
  autoTruncate?: boolean;
}

export declare type EmbeddingResult = {
  embedding: number[];
  metadata?: Record<string, unknown>;
};
