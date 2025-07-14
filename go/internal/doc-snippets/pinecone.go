// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package snippets

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/pinecone"
)

func pineconeEx(ctx context.Context) error {
	g, err := genkit.Init(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// [START init]
	if err := (&pinecone.Pinecone{}).Init(ctx, g); err != nil {
		return err
	}
	// [END init]

	var pineconeAPIKey string
	// [START initkey]
	if err := (&pinecone.Pinecone{APIKey: pineconeAPIKey}).Init(ctx, g); err != nil {
		return err
	}
	// [END initkey]

	var docChunks []*ai.Document

	// [START defineretriever]
	retOpts := &ai.RetrieverOptions{
		ConfigSchema: pinecone.PineconeRetrieverOptions{},
		Info: &ai.RetrieverInfo{
			Label: "Pinecone",
			Supports: &ai.RetrieverSupports{
				Media: false,
			},
		},
	}
	ds, menuRetriever, err := pinecone.DefineRetriever(ctx, g, pinecone.Config{
		IndexID:  "menu_data",                                           // Your Pinecone index
		Embedder: googlegenai.GoogleAIEmbedder(g, "text-embedding-004"), // Embedding model of your choice
	}, retOpts)
	if err != nil {
		return err
	}
	// [END defineretriever]

	// [START index]
	err = pinecone.Index(ctx, docChunks, ds, "")
	if err != nil {
		return err
	}
	// [END index]

	var userInput string

	// [START retrieve]
	resp, err := menuRetriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query:   ai.DocumentFromText(userInput, nil),
		Options: nil,
	})
	if err != nil {
		return err
	}
	menuInfo := resp.Documents
	// [END retrieve]

	_ = menuInfo

	return nil
}
