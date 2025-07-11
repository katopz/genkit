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

import { deepSeek } from '@genkit-ai/compat-oai/deepseek';
import { openAI } from '@genkit-ai/compat-oai/openai';
import { xAI } from '@genkit-ai/compat-oai/xai';
import { startFlowServer } from '@genkit-ai/express';
import dotenv from 'dotenv';
import { genkit, z } from 'genkit';

dotenv.config();

const ai = genkit({
  plugins: [
    openAI(),
    deepSeek({ apiKey: process.env['DEEPSEEK_API_KEY'] }),
    xAI({ apiKey: process.env['XAI_API_KEY'] }),
  ],
});

export const jokeFlow = ai.defineFlow(
  {
    name: 'jokeFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    const llmResponse = await ai.generate({
      prompt: `tell me a joke about ${subject}`,
      model: openAI.model('gpt-4.1'),
    });
    return llmResponse.text;
  }
);

export const modelWrapFlow = ai.defineFlow(
  {
    name: 'modelWrapFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
    streamSchema: z.string(),
  },
  async (subject, { sendChunk }) => {
    const { stream, response } = ai.generateStream({
      model: deepSeek.model('deepseek-chat'),
      prompt: `tell me a fun fact about ${subject}`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.text);
    }

    const { text } = await response;

    return text;
  }
);

export const webSearchFlow = ai.defineFlow(
  {
    name: 'webSearchFlow',
    outputSchema: z.string(),
  },
  async () => {
    const llmResponse = await ai.generate({
      prompt: `What was a positive news story from today?`,
      model: openAI.model('gpt-4o-search-preview'),
      config: {
        web_search_options: {},
      },
    });
    return llmResponse.text;
  }
);

//  genkit flow:run embedFlow \"hello world\"

export const embedFlow = ai.defineFlow(
  {
    name: 'embedFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (text) => {
    const embedding = await ai.embed({
      embedder: openAI.embedder('text-embedding-ada-002'),
      content: text,
    });

    return JSON.stringify(embedding);
  }
);

startFlowServer({
  flows: [jokeFlow, embedFlow],
});
