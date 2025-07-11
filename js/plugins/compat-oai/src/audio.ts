/**
 * Copyright 2024 The Fire Company
 * Copyright 2024 Google LLC
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
import type {
  GenerateRequest,
  GenerateResponseData,
  Genkit,
  ModelReference,
} from 'genkit';
import { Message, z } from 'genkit';
import type { ModelAction } from 'genkit/model';
import type OpenAI from 'openai';
import { Response } from 'openai/core.mjs';
import type {
  SpeechCreateParams,
  Transcription,
  TranscriptionCreateParams,
} from 'openai/resources/audio/index.mjs';

/**
 * Supported media formats for Audio generation
 */
export const RESPONSE_FORMAT_MEDIA_TYPES = {
  mp3: 'audio/mpeg',
  opus: 'audio/opus',
  aac: 'audio/aac',
  flac: 'audio/flac',
  wav: 'audio/wav',
  pcm: 'audio/L16',
};

function toTTSRequest(
  modelName: string,
  request: GenerateRequest
): SpeechCreateParams {
  const {
    voice,
    version: modelVersion,
    temperature,
    maxOutputTokens,
    stopSequences,
    topK,
    topP,
    ...restOfConfig
  } = request.config ?? {};

  const options: SpeechCreateParams = {
    model: modelVersion ?? modelName,
    input: new Message(request.messages[0]).text,
    voice: voice ?? 'alloy',
    ...restOfConfig, // passthorugh rest of the config
  };
  for (const k in options) {
    if (options[k] === undefined) {
      delete options[k];
    }
  }
  return options;
}

async function toGenerateResponse(
  response: Response,
  responseFormat: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm' = 'mp3'
): Promise<GenerateResponseData> {
  const resultArrayBuffer = await response.arrayBuffer();
  const resultBuffer = Buffer.from(new Uint8Array(resultArrayBuffer));
  const mediaType = RESPONSE_FORMAT_MEDIA_TYPES[responseFormat];
  return {
    message: {
      role: 'model',
      content: [
        {
          media: {
            contentType: mediaType,
            url: `data:${mediaType};base64,${resultBuffer.toString('base64')}`,
          },
        },
      ],
    },
    finishReason: 'stop',
    raw: response,
  };
}

/**
 * Method to define a new Genkit Model that is compatible with the Open AI Audio
 * API. 
 *
 * These models are to be used to create audio speech from a given request.
 * @param params An object containing parameters for defining the OpenAI speech
 * model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.

 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAISpeechModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  ai: Genkit;
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<CustomOptions>;
}): ModelAction {
  const { ai, name, client, modelRef } = params;

  const model = name.split('/').pop();
  return ai.defineModel(
    {
      name,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request) => {
      const ttsRequest = toTTSRequest(model!, request);
      const result = await client.audio.speech.create(ttsRequest);
      return await toGenerateResponse(result, ttsRequest.response_format);
    }
  );
}

function toSttRequest(
  modelName: string,
  request: GenerateRequest
): TranscriptionCreateParams {
  const message = new Message(request.messages[0]);
  const media = message.media;
  if (!media?.url) {
    throw new Error('No media found in the request');
  }
  const mediaBuffer = Buffer.from(
    media.url.slice(media.url.indexOf(',') + 1),
    'base64'
  );
  const mediaFile = new File([mediaBuffer], 'input', {
    type:
      media.contentType ??
      media.url.slice('data:'.length, media.url.indexOf(';')),
  });
  const {
    temperature,
    version: modelVersion,
    maxOutputTokens,
    stopSequences,
    topK,
    topP,
    ...restOfConfig
  } = request.config ?? {};

  const options: TranscriptionCreateParams = {
    model: modelVersion ?? modelName,
    file: mediaFile,
    prompt: message.text,
    temperature,
    ...restOfConfig, // passthrough rest of the config
  };
  const outputFormat = request.output?.format as 'json' | 'text' | 'media';
  const customFormat = request.config?.response_format;
  if (outputFormat && customFormat) {
    if (
      outputFormat === 'json' &&
      customFormat !== 'json' &&
      customFormat !== 'verbose_json'
    ) {
      throw new Error(
        `Custom response format ${customFormat} is not compatible with output format ${outputFormat}`
      );
    }
  }
  if (outputFormat === 'media') {
    throw new Error(`Output format ${outputFormat} is not supported.`);
  }
  options.response_format = customFormat || outputFormat || 'text';
  for (const k in options) {
    if (options[k] === undefined) {
      delete options[k];
    }
  }
  return options;
}

function transcriptionToGenerateResponse(
  result: Transcription | string
): GenerateResponseData {
  return {
    message: {
      role: 'model',
      content: [
        {
          text: typeof result === 'string' ? result : result.text,
        },
      ],
    },
    finishReason: 'stop',
    raw: result,
  };
}

/**
 * Method to define a new Genkit Model that is compatible with Open AI
 * Transcriptions API. 
 *
 * These models are to be used to transcribe audio to text.
 *
 * @param params An object containing parameters for defining the OpenAI
 * transcription model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.

 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAITranscriptionModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  ai: Genkit;
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<CustomOptions>;
}): ModelAction {
  const { ai, name, client, modelRef } = params;

  return ai.defineModel(
    {
      name,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request) => {
      const modelName = name.split('/').pop();
      const params = toSttRequest(modelName!, request);
      // Explicitly setting stream to false ensures we use the non-streaming overload
      const result = await client.audio.transcriptions.create({
        ...params,
        stream: false,
      });
      return transcriptionToGenerateResponse(result);
    }
  );
}
