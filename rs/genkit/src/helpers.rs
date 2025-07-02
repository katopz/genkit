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
