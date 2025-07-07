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

use genkit_ai::{
    resource::{
        define_resource, find_matching_resource, ResourceInput, ResourceOptions, ResourceOutput,
    },
    Part,
};
use genkit_core::registry::{ErasedAction, Registry};
use rstest::{fixture, rstest};
use serde_json::{from_value, json};
use std::collections::HashMap;

#[fixture]
fn registry() -> Registry {
    Registry::new()
}

#[rstest]
#[tokio::test]
async fn test_defines_and_matches_static_resource_uri(mut registry: Registry) {
    let options: ResourceOptions = from_value(json!({
        "name": "testResource",
        "uri": "foo://bar",
        "description": "does foo things",
        "metadata": { "foo": "bar" }
    }))
    .unwrap();

    let test_resource = define_resource(&mut registry, options, |_, _| async {
        Ok(ResourceOutput {
            content: vec![Part::text("foo stuff")],
        })
    })
    .unwrap();

    assert_eq!(test_resource.name(), "testResource");
    assert_eq!(
        test_resource.metadata().description,
        Some("does foo things".to_string())
    );
    let metadata = test_resource.metadata().metadata.clone();
    assert_eq!(metadata["foo"], "bar");
    assert_eq!(metadata["resource"]["uri"], "foo://bar");

    // Test matching logic
    let matched = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "foo://bar".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(matched.is_some());

    let not_matched = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "foo://baz".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(not_matched.is_none());

    // Test runner execution and metadata injection
    let output = test_resource
        .run(
            ResourceInput {
                uri: "foo://bar".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    let part = &output.result.content[0];
    assert_eq!(part.text.as_deref(), Some("foo stuff"));
    let resource_meta = part.metadata.as_ref().unwrap()["resource"]
        .as_object()
        .unwrap();
    let parent_meta = resource_meta["parent"].as_object().unwrap();
    assert_eq!(parent_meta["uri"], "foo://bar");
    assert!(parent_meta.get("template").is_none());

    assert!(registry
        .lookup_action("/resource/testResource")
        .await
        .is_some());
}

#[rstest]
#[tokio::test]
async fn test_defines_and_matches_template_resource_uri(mut registry: Registry) {
    let test_resource = define_resource(
        &mut registry,
        ResourceOptions {
            template: Some("foo://bar/{baz}".to_string()),
            description: Some("does foo things".to_string()),
            ..Default::default()
        },
        |input, _| async move {
            Ok(ResourceOutput {
                content: vec![Part::text(format!("foo stuff {}", input.uri))],
            })
        },
    )
    .unwrap();

    assert_eq!(test_resource.name(), "foo://bar/{baz}");
    assert_eq!(
        test_resource.metadata().description,
        Some("does foo things".to_string())
    );

    // Test matching logic
    let matched_resource = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "foo://bar/something".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(matched_resource.is_some());

    let not_matched_resource = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "foo://baz/something".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(not_matched_resource.is_none());

    // Test runner execution
    let output = test_resource
        .run(
            ResourceInput {
                uri: "foo://bar/something".to_string(),
            },
            None,
        )
        .await
        .unwrap();
    let part = &output.result.content[0];
    assert_eq!(part.text.as_ref().unwrap(), "foo stuff foo://bar/something");
    let resource_meta = part.metadata.as_ref().unwrap()["resource"]
        .as_object()
        .unwrap();
    let parent_meta = resource_meta["parent"].as_object().unwrap();
    assert_eq!(parent_meta["template"], "foo://bar/{baz}");
    assert_eq!(parent_meta["uri"], "foo://bar/something");

    assert!(registry
        .lookup_action("/resource/foo://bar/{baz}")
        .await
        .is_some());
}

#[rstest]
#[tokio::test]
async fn test_handles_parent_resources(mut registry: Registry) {
    let test_resource = define_resource(
        &mut registry,
        ResourceOptions {
            name: Some("testResource".to_string()),
            template: Some("file://{id*}".to_string()),
            ..Default::default()
        },
        |file, _| async move {
            let mut metadata1 = HashMap::new();
            metadata1.insert(
                "resource".to_string(),
                json!({ "uri": format!("{}/sub1.txt", file.uri) }),
            );

            let mut metadata2 = HashMap::new();
            metadata2.insert(
                "resource".to_string(),
                json!({ "uri": format!("{}/sub2.txt", file.uri) }),
            );

            Ok(ResourceOutput {
                content: vec![
                    Part {
                        text: Some("sub1".to_string()),
                        metadata: Some(metadata1),
                        ..Default::default()
                    },
                    Part {
                        text: Some("sub2".to_string()),
                        metadata: Some(metadata2),
                        ..Default::default()
                    },
                ],
            })
        },
    )
    .unwrap();

    let output = test_resource
        .run(
            ResourceInput {
                uri: "file:///some/directory".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    // Check first part for parent injection
    let part1_meta = &output.result.content[0].metadata.as_ref().unwrap()["resource"];
    assert_eq!(part1_meta["uri"], "file:///some/directory/sub1.txt");
    assert_eq!(part1_meta["parent"]["template"], "file://{id*}");
    assert_eq!(part1_meta["parent"]["uri"], "file:///some/directory");

    // Check second part for parent injection
    let part2_meta = &output.result.content[1].metadata.as_ref().unwrap()["resource"];
    assert_eq!(part2_meta["uri"], "file:///some/directory/sub2.txt");
    assert_eq!(part2_meta["parent"]["template"], "file://{id*}");
    assert_eq!(part2_meta["parent"]["uri"], "file:///some/directory");
}

#[rstest]
#[tokio::test]
async fn test_finds_matching_resource(mut registry: Registry) {
    define_resource(
        &mut registry,
        ResourceOptions {
            name: Some("testTemplateResource".to_string()),
            template: Some("foo://bar/{baz}".to_string()),
            ..Default::default()
        },
        |input, _| async move {
            Ok(ResourceOutput {
                content: vec![Part::text(input.uri)],
            })
        },
    )
    .unwrap();
    define_resource(
        &mut registry,
        ResourceOptions {
            name: Some("testResource".to_string()),
            uri: Some("bar://baz".to_string()),
            ..Default::default()
        },
        |_, _| async {
            Ok(ResourceOutput {
                content: vec![Part::text("bar")],
            })
        },
    )
    .unwrap();

    // Find static resource
    let got_bar = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "bar://baz".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(got_bar.is_some());
    assert_eq!(got_bar.unwrap().name(), "testResource");

    // Find template resource
    let got_foo = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "foo://bar/something".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(got_foo.is_some());
    assert_eq!(got_foo.unwrap().name(), "testTemplateResource");

    // Find nothing
    let got_unmatched = find_matching_resource(
        &registry,
        &ResourceInput {
            uri: "unknown://bar/something".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(got_unmatched.is_none());
}
