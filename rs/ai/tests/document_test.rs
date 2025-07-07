// Copyright 2024 Google LLC
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

//! # Document Tests

use genkit_ai::document::{check_unique_documents, Document, Part};
use genkit_ai::embedder::Embedding;
use serde_json::{from_value, json, Value};
use std::collections::HashMap;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_document_constructor_makes_copy() {
        let original_content: Vec<Part> = from_value(json!([
            { "media": { "url": "data:foo" } },
            { "media": { "url": "data:bar" } },
        ]))
        .unwrap();
        let original_metadata: HashMap<String, Value> =
            from_value(json!({ "bar": "baz", "embedMetadata": { "bar": "qux" } })).unwrap();

        let doc = Document::new(original_content.clone(), Some(original_metadata.clone()));

        // Modify the original content and metadata after creating the document
        let mut modified_content = original_content;
        modified_content[0].media.as_mut().unwrap().url = "data: bam".to_string();

        let mut modified_metadata = original_metadata;
        modified_metadata
            .get_mut("embedMetadata")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("bar".to_string(), json!("boom"));

        // Assert that the document's content and metadata are unchanged
        assert_eq!(doc.content[0].media.as_ref().unwrap().url, "data:foo");
        assert_eq!(
            doc.metadata.as_ref().unwrap()["embedMetadata"]["bar"],
            "qux"
        );
    }

    #[test]
    fn test_text_concatenation() {
        let doc1 = Document::from_text("foo", None);
        assert_eq!(doc1.text(), "foo");

        let doc2 = Document::new(
            from_value(json!([ { "text": "foo" }, { "text": "bar" }])).unwrap(),
            None,
        );
        assert_eq!(doc2.text(), "foobar");
    }

    #[test]
    fn test_media_retrieval() {
        let doc = Document::new(
            from_value(json!([
                { "media": { "url": "data:foo" } },
                { "media": { "url": "data:bar" } },
            ]))
            .unwrap(),
            None,
        );
        let media = doc.media();
        assert_eq!(media.len(), 2);
        assert_eq!(media[0].url, "data:foo");
        assert_eq!(media[1].url, "data:bar");
    }

    #[test]
    fn test_data_retrieval() {
        let text_doc = Document::from_text("foo", None);
        assert_eq!(text_doc.data(), "foo");

        let image_url = "gs://somebucket/someimage.png";
        let image_doc = Document::from_media(image_url, Some("image/png".to_string()), None);
        assert_eq!(image_doc.data(), image_url);

        let video_url = "gs://somebucket/somevideo.mp4";
        let video_doc = Document::from_media(video_url, Some("video/mp4".to_string()), None);
        assert_eq!(video_doc.data(), video_url);
    }

    #[test]
    fn test_data_type_retrieval() {
        let text_doc = Document::from_text("foo", None);
        assert_eq!(text_doc.data_type(), "text");

        let image_content_type = "image/png";
        let image_doc = Document::from_media(
            "gs://somebucket/someimage.png",
            Some(image_content_type.to_string()),
            None,
        );
        assert_eq!(image_doc.data_type(), image_content_type);

        let video_content_type = "video/mp4";
        let video_doc = Document::from_media(
            "gs://somebucket/somevideo.mp4",
            Some(video_content_type.to_string()),
            None,
        );
        assert_eq!(video_doc.data_type(), video_content_type);
    }

    #[test]
    fn test_to_json_value_and_copy() {
        let original_content: Vec<Part> = from_value(json!([
            { "media": { "url": "data:foo" } },
            { "media": { "url": "data:bar" } },
        ]))
        .unwrap();
        let original_metadata: HashMap<String, Value> =
            from_value(json!({ "bar": "baz", "embedMetadata": { "bar": "qux" } })).unwrap();

        let mut doc = Document::new(original_content.clone(), Some(original_metadata.clone()));
        let json_doc = doc.to_json_value();

        assert_eq!(json_doc["content"], json!(original_content));
        assert_eq!(json_doc["metadata"], json!(original_metadata));

        // Change the deep parts of the content in the doc
        doc.content[0].media.as_mut().unwrap().url = "data: bam".to_string();
        assert_eq!(json_doc["content"][0]["media"]["url"], "data:foo");

        // Change the deep parts of the metadata in the doc
        doc.metadata
            .as_mut()
            .unwrap()
            .get_mut("embedMetadata")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("bar".to_string(), json!("boom"));
        assert_eq!(json_doc["metadata"]["embedMetadata"]["bar"], "qux");
    }

    #[test]
    fn test_from_text() {
        let doc = Document::from_text("foo", Some(from_value(json!({ "bar": "baz" })).unwrap()));
        assert_eq!(
            doc.to_json_value(),
            json!({
                "content": [{ "text": "foo" }],
                "metadata": { "bar": "baz" },
            })
        );
    }

    #[test]
    fn test_from_media() {
        let url = "gs://somebucket/someimage.jpg";
        let content_type = "image/jpeg";
        let metadata =
            from_value(json!({ "embedMetadata": { "embeddingType": "image" } })).unwrap();
        let doc = Document::from_media(url, Some(content_type.to_string()), Some(metadata));
        assert_eq!(
            doc.to_json_value(),
            json!({
                "content": [
                    {
                        "media": {
                            "contentType": content_type,
                            "url": url,
                        },
                    },
                ],
                "metadata": { "embedMetadata": { "embeddingType": "image" } },
            })
        );
    }

    #[test]
    fn test_from_data() {
        let text_data = "foo";
        let text_doc = Document::from_data(
            text_data.to_string(),
            "text",
            Some(from_value(json!({ "embedMetadata": { "embeddingType": "text" } })).unwrap()),
        );
        assert_eq!(
            text_doc.to_json_value(),
            json!({
                "content": [{ "text": text_data }],
                "metadata": { "embedMetadata": { "embeddingType": "text" } },
            })
        );

        let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAjCB0C8AAAAASUVORK5CYII=";
        let image_doc = Document::from_data(
            image_data.to_string(),
            "image/png",
            Some(from_value(json!({ "embedMetadata": { "embeddingType": "image" } })).unwrap()),
        );
        assert_eq!(
            image_doc.to_json_value(),
            json!({
                "content": [
                    {
                        "media": {
                            "contentType": "image/png",
                            "url": image_data,
                        },
                    },
                ],
                "metadata": { "embedMetadata": { "embeddingType": "image" } },
            })
        );
    }

    #[test]
    fn test_get_embedding_documents_single() {
        let doc = Document::from_text("foo", None);
        let embeddings: Vec<Embedding> =
            from_value(json!([{"embedding": [0.1, 0.2, 0.3]}])).unwrap();
        let docs = doc.get_embedding_documents(&embeddings);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0], doc);
    }

    #[test]
    fn test_get_embedding_documents_multiple() {
        let url = "gs://somebucket/somevideo.mp4";
        let doc = Document::from_media(
            url,
            Some("video/mp4".to_string()),
            Some(from_value(json!({ "start": 0, "end": 60 })).unwrap()),
        );

        let embeddings: Vec<Embedding> = (0..4)
            .map(|i| {
                from_value(json!({
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {
                        "embeddingType": "video",
                        "start": i * 15,
                        "end": (i + 1) * 15,
                    },
                }))
                .unwrap()
            })
            .collect();

        let docs = doc.get_embedding_documents(&embeddings);
        assert_eq!(docs.len(), embeddings.len());

        for (i, d) in docs.iter().enumerate() {
            assert_eq!(d.content, doc.content);
            assert_eq!(
                d.metadata.as_ref().unwrap()["embedMetadata"],
                json!(embeddings[i].metadata.as_ref().unwrap())
            );
            let mut original_metadata = d.metadata.clone().unwrap();
            original_metadata.remove("embedMetadata");
            assert_eq!(original_metadata, doc.metadata.clone().unwrap());
        }
    }

    #[test]
    fn test_check_unique_documents() {
        let url = "gs://somebucket/somevideo.mp4";
        let doc = Document::from_media(
            url,
            Some("video/mp4".to_string()),
            Some(from_value(json!({ "start": 0, "end": 60 })).unwrap()),
        );

        let embeddings: Vec<Embedding> = (0..4)
            .map(|i| {
                from_value(json!({
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {
                        "embeddingType": "video",
                        "start": i * 15,
                        "end": (i + 1) * 15,
                    },
                }))
                .unwrap()
            })
            .collect();
        let docs = doc.get_embedding_documents(&embeddings);
        assert!(check_unique_documents(&docs));

        // Test with duplicate documents
        let duplicate_docs = vec![doc.clone(), doc];
        assert!(!check_unique_documents(&duplicate_docs));
    }
}
