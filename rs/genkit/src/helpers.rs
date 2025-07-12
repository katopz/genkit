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

use serde_json::Value;

/// Helper function to merge two serde_json::Value objects.
/// Merges `b` into `a`. If both are objects, their fields are merged.
/// Otherwise, `b` overwrites `a` if `b` is not null.
pub fn merge_configs(a: &mut Value, b: Value) {
    if let Value::Object(a_map) = a {
        if let Value::Object(b_map) = b {
            for (k, v) in b_map {
                if v.is_null() {
                    a_map.remove(&k);
                } else {
                    merge_configs(a_map.entry(k).or_insert(Value::Null), v);
                }
            }
            return;
        }
    }
    if !b.is_null() {
        *a = b;
    }
}
