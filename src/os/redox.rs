//! Redox OS Scheme Protocol Simulator and Multimodal Daemon.
//!
//! Implements a userspace simulation of the Redox microkernel C packet protocol (`RedoxPacket`)
//! and provides testable command, voice, and vision scheme channels (`sail:`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::InferenceBackend;
use crate::engine::Engine;
use crate::generate::SamplingConfig;

pub const SYS_OPEN: usize = 5;
pub const SYS_READ: usize = 3;
pub const SYS_WRITE: usize = 4;
pub const SYS_CLOSE: usize = 6;

/// Redox microkernel request packet structure.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RedoxPacket {
    pub id: usize,
    pub op: usize,
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}

/// The active multimodal stream channel opened by a client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SailChannel {
    /// Text command & terminal interaction channel.
    Command,
    /// Continuous voice stream channel (raw PCM / Mel frames).
    Voice,
    /// Visual context & framebuffer capture channel.
    Vision,
}

/// An open file descriptor handle inside the `sail:` scheme.
pub struct SchemeHandle {
    pub channel: SailChannel,
    pub read_buffer: Vec<u8>,
    pub write_buffer: Vec<u8>,
    pub cursor: usize,
}

/// Redox Scheme Handler managing clients and translating packets into SAIL inferences.
pub struct RedoxSchemeHandler {
    engine: Option<Engine<InferenceBackend>>,
    next_handle_id: AtomicUsize,
    handles: Arc<Mutex<HashMap<usize, SchemeHandle>>>,
    sampling: SamplingConfig,
}

impl RedoxSchemeHandler {
    pub fn new(engine: Option<Engine<InferenceBackend>>) -> Self {
        Self {
            engine,
            next_handle_id: AtomicUsize::new(1),
            handles: Arc::new(Mutex::new(HashMap::new())),
            sampling: SamplingConfig {
                temperature: 0.7,
                top_k: 4,
                seed: 42,
                ..Default::default()
            },
        }
    }

    /// Handles an incoming microkernel packet, returning the result code.
    pub fn handle_packet(&self, packet: &RedoxPacket) -> usize {
        match packet.op {
            SYS_OPEN => {
                // b: path pointer in user memory (simulated as slice or string id)
                // In simulation or test: a=channel type (1=command, 2=voice, 3=vision)
                let channel = match packet.a {
                    1 => SailChannel::Command,
                    2 => SailChannel::Voice,
                    3 => SailChannel::Vision,
                    _ => return usize::MAX, // Error: unknown channel
                };

                let handle_id = self.next_handle_id.fetch_add(1, Ordering::Relaxed);
                let handle = SchemeHandle {
                    channel,
                    read_buffer: Vec::new(),
                    write_buffer: Vec::new(),
                    cursor: 0,
                };
                self.handles.write_lock().insert(handle_id, handle);
                handle_id
            }

            SYS_WRITE => {
                let handle_id = packet.a;
                let data_ptr = packet.b as *const u8;
                let len = packet.c;

                let mut handles = self.handles.write_lock();
                let handle = match handles.get_mut(&handle_id) {
                    Some(h) => h,
                    None => return usize::MAX,
                };

                if len > 0 && !data_ptr.is_null() {
                    // Safe copy of userspace buffer
                    let slice = unsafe { std::slice::from_raw_parts(data_ptr, len) };
                    handle.write_buffer.extend_from_slice(slice);

                    // If a complete line/command is received, process it
                    if handle.channel == SailChannel::Command && slice.contains(&b'\n') {
                        let text = String::from_utf8_lossy(&handle.write_buffer).to_string();
                        handle.write_buffer.clear();

                        let response = if let Some(eng) = &self.engine {
                            match eng.generate(&text, 50, &self.sampling) {
                                Ok((res, _)) => res,
                                Err(e) => format!("SAIL Error: {e}\n"),
                            }
                        } else {
                            format!("[SAIL-RedoxOS]: Received '{}'\n", text.trim())
                        };

                        handle.read_buffer = response.into_bytes();
                        handle.cursor = 0;
                    }
                }
                len
            }

            SYS_READ => {
                let handle_id = packet.a;
                let buf_ptr = packet.b as *mut u8;
                let max_len = packet.c;

                let mut handles = self.handles.write_lock();
                let handle = match handles.get_mut(&handle_id) {
                    Some(h) => h,
                    None => return usize::MAX,
                };

                let available = handle.read_buffer.len().saturating_sub(handle.cursor);
                let to_read = available.min(max_len);

                if to_read > 0 && !buf_ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            handle.read_buffer.as_ptr().add(handle.cursor),
                            buf_ptr,
                            to_read,
                        );
                    }
                    handle.cursor += to_read;
                }
                to_read
            }

            SYS_CLOSE => {
                let handle_id = packet.a;
                self.handles.write_lock().remove(&handle_id);
                0
            }

            _ => usize::MAX, // Unsupported operation
        }
    }
}

/// Helper for Mutex lock convenience
trait MutexHelper<T> {
    fn write_lock(&self) -> std::sync::MutexGuard<'_, T>;
}

impl<T> MutexHelper<T> for Mutex<T> {
    fn write_lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.lock().unwrap()
    }
}

/// The standalone SAIL daemon (`saild`) managing the Redox microkernel scheme loop.
pub struct SailDaemon {
    pub handler: Arc<RedoxSchemeHandler>,
}

impl SailDaemon {
    pub fn new(engine: Option<Engine<InferenceBackend>>) -> Self {
        Self {
            handler: Arc::new(RedoxSchemeHandler::new(engine)),
        }
    }

    /// Dispatches a high-level command request through the scheme without raw pointers.
    pub fn send_command(&self, prompt: &str) -> Result<String, String> {
        let open_pkt = RedoxPacket {
            id: 1,
            op: SYS_OPEN,
            a: 1, // Command channel
            b: 0,
            c: 0,
            d: 0,
        };
        let handle_id = self.handler.handle_packet(&open_pkt);
        if handle_id == usize::MAX {
            return Err("Failed to open sail:command scheme".into());
        }

        let mut input_bytes = prompt.as_bytes().to_vec();
        if !input_bytes.ends_with(b"\n") {
            input_bytes.push(b'\n');
        }

        let write_pkt = RedoxPacket {
            id: 2,
            op: SYS_WRITE,
            a: handle_id,
            b: input_bytes.as_ptr() as usize,
            c: input_bytes.len(),
            d: 0,
        };
        self.handler.handle_packet(&write_pkt);

        let mut read_buf = vec![0u8; 1024];
        let read_pkt = RedoxPacket {
            id: 3,
            op: SYS_READ,
            a: handle_id,
            b: read_buf.as_mut_ptr() as usize,
            c: read_buf.len(),
            d: 0,
        };
        let bytes_read = self.handler.handle_packet(&read_pkt);

        let close_pkt = RedoxPacket {
            id: 4,
            op: SYS_CLOSE,
            a: handle_id,
            b: 0,
            c: 0,
            d: 0,
        };
        self.handler.handle_packet(&close_pkt);

        if bytes_read > 0 && bytes_read != usize::MAX {
            Ok(String::from_utf8_lossy(&read_buf[..bytes_read]).to_string())
        } else {
            Ok(String::new())
        }
    }
}
