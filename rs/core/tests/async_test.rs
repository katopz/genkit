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

//! # Async Utilities Tests
//!
//! Integration tests for the async utilities, ported from `async_test.ts`.

use futures::stream::TryStreamExt;
use genkit_core::async_utils::{channel, lazy_fut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

mod test {
    use crate::*;

    #[tokio::test]
    async fn test_lazy_future() {
        let has_run = Arc::new(AtomicBool::new(false));
        let has_run_clone = has_run.clone();

        let fut = lazy_fut(move || {
            // This closure should only run when the future is awaited.
            has_run_clone.store(true, Ordering::SeqCst);
            async { 42 }
        });

        // The closure should not have run yet.
        assert!(!has_run.load(Ordering::SeqCst));

        // Now, await the future.
        let result = fut.await;

        // The closure should have run.
        assert!(has_run.load(Ordering::SeqCst));
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_channel_simple_send_and_close() {
        let (tx, mut rx) = channel::<i32>();

        tx.send(1);
        tx.send(2);
        tx.close();

        assert_eq!(rx.try_next().await.unwrap(), Some(1));
        assert_eq!(rx.try_next().await.unwrap(), Some(2));
        assert_eq!(rx.try_next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_channel_stream_waits_for_send() {
        let (tx, mut rx) = channel::<i32>();

        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            tx.send(100);
            tx.close();
        });

        assert_eq!(rx.try_next().await.unwrap(), Some(100));
        assert_eq!(rx.try_next().await.unwrap(), None);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_channel_error() {
        let (tx, mut rx) = channel::<i32>();

        tx.send(1);
        tx.error("something went wrong");

        assert_eq!(rx.try_next().await.unwrap(), Some(1));
        let err = rx.try_next().await.unwrap_err();
        assert_eq!(err.to_string(), "ChannelError: something went wrong");
        assert_eq!(rx.try_next().await.unwrap(), None); // Stream should be closed after error
    }

    #[tokio::test]
    async fn test_channel_close_before_read() {
        let (tx, mut rx) = channel::<i32>();

        tx.close();

        assert_eq!(rx.try_next().await.unwrap(), None);
    }
}
