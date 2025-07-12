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

//! # Asynchronous Utilities
//!
//! This module provides utilities for asynchronous programming, including a `Channel`
//! for converting callbacks to streams and a `lazy` future utility. This is a port
// of `async.ts`.

use futures::stream::Stream;
use pin_project_lite::pin_project;
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use thiserror::Error;

// SECTION: Channel

/// An error that can be sent through a `Channel`.
#[derive(Debug, Clone, Error, PartialEq)]
pub struct ChannelError {
    message: String,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelError: {}", self.message)
    }
}

/// The state shared between a `Channel` and its `ChannelStream`.
#[derive(Default)]
struct ChannelState<T> {
    buffer: VecDeque<T>,
    closed: bool,
    error: Option<ChannelError>,
    waker: Option<Waker>,
}

/// A handle to send data into a channel, which can be consumed by a `ChannelStream`.
///
/// This is useful for bridging callback-based or imperative code with async streams.
#[derive(Default)]
pub struct Channel<T: Send> {
    state: Arc<Mutex<ChannelState<T>>>,
}

impl<T: Send> Clone for Channel<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

/// An asynchronous stream that consumes data from a `Channel`.
pub struct ChannelStream<T: Send> {
    state: Arc<Mutex<ChannelState<T>>>,
}

/// Creates a new channel, returning the sender and the stream.
pub fn channel<T: Send>() -> (Channel<T>, ChannelStream<T>) {
    let state = Arc::new(Mutex::new(ChannelState {
        buffer: VecDeque::new(),
        closed: false,
        error: None,
        waker: None,
    }));
    let sender = Channel {
        state: state.clone(),
    };
    let stream = ChannelStream { state };
    (sender, stream)
}

impl<T: Send> Channel<T> {
    /// Sends a value into the channel.
    ///
    /// This will wake up the stream if it is waiting for data.
    /// Returns an `Err` if the channel is closed.
    pub fn send(&self, value: T) -> Result<(), ChannelError> {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(ChannelError {
                message: "Tried to send on a closed channel.".to_string(),
            });
        }
        state.buffer.push_back(value);
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
        Ok(())
    }

    /// Closes the channel.
    ///
    /// The stream will end once all items in the buffer have been consumed.
    /// No more items can be sent after closing.
    pub fn close(&self) {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return;
        }
        state.closed = true;
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }

    /// Sends an error through the channel.
    ///
    /// The stream will immediately terminate with this error.
    /// The channel will also be closed.
    pub fn error(&self, message: impl Into<String>) {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return;
        }
        state.error = Some(ChannelError {
            message: message.into(),
        });
        state.closed = true;
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }
}

impl<T: Send> Stream for ChannelStream<T> {
    type Item = Result<T, ChannelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut state = self.state.lock().unwrap();

        // Prioritize draining the buffer before propagating an error.
        if let Some(value) = state.buffer.pop_front() {
            return Poll::Ready(Some(Ok(value)));
        }

        // If the buffer is empty, check for an error.
        if let Some(err) = state.error.take() {
            return Poll::Ready(Some(Err(err)));
        }

        // If the buffer and error are empty, check if the stream is closed.
        if state.closed {
            return Poll::Ready(None);
        }

        // Otherwise, the stream is open but awaiting data.
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

// SECTION: Lazy Future

pin_project! {
    /// A future that creates and runs another future only when it is first polled.
    ///
    /// This is created by the `lazy_fut` function.
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct LazyFut<F, Fut> {
        #[pin]
        state: LazyState<F, Fut>,
    }
}

pin_project! {
    #[project = LazyStateProj]
    enum LazyState<F, Fut> {
        // Un-polled, holding the closure that creates the future.
        New { f: Option<F> },
        // Polled, running the actual future.
        Running { #[pin] fut: Fut },
        // Finished.
        Done,
    }
}

/// Creates a future that defers creation of another future until it is first polled.
///
/// This is the Rust equivalent of a "lazy promise". Since futures in Rust are
/// already lazy (they do nothing until polled), this utility is for cases where
/// the *creation* of the future itself is an expensive operation that should
/// be deferred.
///
/// NOTE: This requires the `pin-project-lite` crate.
pub fn lazy_fut<F, Fut>(f: F) -> LazyFut<F, Fut>
where
    F: FnOnce() -> Fut,
    Fut: Future,
{
    LazyFut {
        state: LazyState::New { f: Some(f) },
    }
}

impl<F, Fut> Future for LazyFut<F, Fut>
where
    F: FnOnce() -> Fut,
    Fut: Future,
{
    type Output = Fut::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();

        // If this is the first poll, transition from `New` to `Running`.
        if let LazyStateProj::New { f } = this.state.as_mut().project() {
            let func = f.take().expect("LazyFut polled after being completed");
            let fut = func();
            this.state.set(LazyState::Running { fut });
        }

        // Poll the running future.
        match this.state.as_mut().project() {
            LazyStateProj::Running { fut } => {
                let res = fut.poll(cx);
                // If the inner future is done, we transition our state to `Done`.
                if res.is_ready() {
                    this.state.set(LazyState::Done);
                }
                res
            }
            // Should not be polled after completion.
            LazyStateProj::Done => panic!("Lazy future polled after completion"),
            // This case was handled above.
            LazyStateProj::New { .. } => {
                unreachable!("Polled a New state that should have been transitioned")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::TryStreamExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tokio::time::sleep;

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

        tx.send(1).unwrap();
        tx.send(2).unwrap();
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
            tx.send(100).unwrap();
            tx.close();
        });

        assert_eq!(rx.try_next().await.unwrap(), Some(100));
        assert_eq!(rx.try_next().await.unwrap(), None);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_channel_error() {
        let (tx, mut rx) = channel::<i32>();

        tx.send(1).unwrap();
        tx.error("something went wrong");

        assert_eq!(rx.try_next().await.unwrap(), Some(1));
        let err: ChannelError = rx.try_next().await.unwrap_err();
        assert_eq!(err.to_string(), "ChannelError: something went wrong");
        assert_eq!(rx.try_next().await.unwrap(), None); // Stream should be closed after error
    }

    #[tokio::test]
    async fn test_channel_close_before_read() {
        let (tx, mut rx) = channel::<i32>();

        tx.close();

        assert_eq!(rx.try_next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_send_on_closed_channel() {
        let (tx, mut rx) = channel::<i32>();
        tx.close();
        let result = tx.send(42);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err.to_string(),
            "ChannelError: Tried to send on a closed channel."
        );
        // Make sure nothing was actually sent.
        assert_eq!(rx.try_next().await.unwrap(), None);
    }
}
