# basic-gemini

## Setup

1. Get `service-account.json` from GCP.
2. Put `service-account.json` in the `rs` folder.
3. You may hit `Vertex AI error: API request to list models failed with status 403 Forbidden`, do add a permission for `client_email` via cli.

```bash
gcloud auth login
gcloud config set project your-project-id

gcloud projects add-iam-policy-binding your-project-id \
  --member serviceAccount:firebase-adminsdk-xxx@yyy.iam.gserviceaccount.com \
  --role roles/serviceusage.serviceUsageConsumer
```

## Dev
> In `testapps/basic-gemini` path
```sh
RUST_LOG=debug GOOGLE_APPLICATION_CREDENTIALS="./service-account.json" cargo run
```

## Expect

```
Flow action_result: ActionResult {
    result: "A banana.\n",
    telemetry: TelemetryInfo {
        trace_id: "00000000000000000000000000000000",
        span_id: "0000000000000000",
    },
}
```
