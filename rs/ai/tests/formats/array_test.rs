use serde_json::json;

// Assuming the following structs and functions exist in your crate.
// You might need to adjust the paths and definitions according to your project structure.
use genkit::formats::array::array_formatter;
use genkit::generate::{GenerateResponseChunk, GenerateResponseChunkData};
use genkit::message::{Message, MessageData};
use genkit::model::{Content, GenerateRequest, Output, Schema, SchemaType};

#[cfg(test)]
mod test {
    use super::*;
    fn test_streaming_emits_complete_array_items_as_they_arrive() {
        let chunks_data = vec![
            ("{\"id\": 1,", json!([])),
            ("\"name\": \"first\"}", json!([{"id": 1, "name": "first"}])),
            (
                ", {\"id\": 2, \"name\": \"second\"}]",
                json!([{"id": 2, "name": "second"}]),
            ),
        ];

        let parser = array_formatter().handler(None).unwrap();
        let mut previous_chunks: Vec<GenerateResponseChunk> = Vec::new();

        for (text, want) in chunks_data {
            let new_chunk_data = GenerateResponseChunkData {
                index: 0,
                role: "model".to_string(),
                content: vec![Content::from_text(text)],
                ..Default::default()
            };

            let result = parser.parse_chunk(&new_chunk_data).unwrap();

            previous_chunks.push(GenerateResponseChunk::from(new_chunk_data));
            assert_eq!(result, want);
        }
    }

    #[test]
    fn test_streaming_handles_single_item_arrays() {
        let chunks_data = vec![(
            "[{\"id\": 1, \"name\": \"single\"}]",
            json!([{"id": 1, "name": "single"}]),
        )];

        let parser = array_formatter().handler(None).unwrap();
        let mut previous_chunks: Vec<GenerateResponseChunk> = Vec::new();

        for (text, want) in chunks_data {
            let new_chunk_data = GenerateResponseChunkData {
                index: 0,
                role: "model".to_string(),
                content: vec![Content::from_text(text)],
                ..Default::default()
            };

            let result = parser.parse_chunk(&new_chunk_data).unwrap();

            previous_chunks.push(GenerateResponseChunk::from(new_chunk_data));
            assert_eq!(result, want);
        }
    }

    #[test]
    fn test_streaming_handles_preamble_with_code_fence() {
        let chunks_data = vec![
            ("Here is the array you requested:\n\n```json\n[", json!([])),
            (
                "{\"id\": 1, \"name\": \"item\"}]\n```",
                json!([{"id": 1, "name": "item"}]),
            ),
        ];

        let parser = array_formatter().handler(None).unwrap();
        let mut previous_chunks: Vec<GenerateResponseChunk> = Vec::new();

        for (text, want) in chunks_data {
            let new_chunk_data = GenerateResponseChunkData {
                index: 0,
                role: "model".to_string(),
                content: vec![Content::from_text(text)],
                ..Default::default()
            };

            let result = parser.parse_chunk(&new_chunk_data).unwrap();

            previous_chunks.push(GenerateResponseChunk::from(new_chunk_data));
            assert_eq!(result, want);
        }
    }

    #[test]
    fn test_message_parses_complete_array_response() {
        let message_data = MessageData {
            role: "model".to_string(),
            content: vec![Content::from_text("[{\"id\": 1, \"name\": \"test\"}]")],
        };
        let want = json!([{"id": 1, "name": "test"}]);

        let parser = array_formatter().handler(None).unwrap();
        let result = parser.parse_message(&Message::from(message_data)).unwrap();
        assert_eq!(result, want);
    }

    #[test]
    fn test_message_parses_empty_array() {
        let message_data = MessageData {
            role: "model".to_string(),
            content: vec![Content::from_text("[]")],
        };
        let want = json!([]);

        let parser = array_formatter().handler(None).unwrap();
        let result = parser.parse_message(&Message::from(message_data)).unwrap();
        assert_eq!(result, want);
    }

    #[test]
    fn test_message_parses_array_with_preamble_and_code_fence() {
        let message_data = MessageData {
            role: "model".to_string(),
            content: vec![Content::from_text(
                "Here is the array:\n\n```json\n[{\"id\": 1}]\n```",
            )],
        };
        let want = json!([{"id": 1}]);

        let parser = array_formatter().handler(None).unwrap();
        let result = parser.parse_message(&Message::from(message_data)).unwrap();
        assert_eq!(result, want);
    }

    #[test]
    #[should_panic(expected = "Must supply an 'array' schema type")]
    fn test_error_throws_for_non_array_schema_type() {
        let request = GenerateRequest {
            messages: Vec::new(),
            output: Some(Output {
                schema: Some(Schema {
                    r#type: SchemaType::String,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        array_formatter().handler(Some(&request)).unwrap();
    }

    #[test]
    #[should_panic(expected = "Must supply an 'array' schema type")]
    fn test_error_throws_for_object_schema_type() {
        let request = GenerateRequest {
            messages: Vec::new(),
            output: Some(Output {
                schema: Some(Schema {
                    r#type: SchemaType::Object,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        array_formatter().handler(Some(&request)).unwrap();
    }
}
