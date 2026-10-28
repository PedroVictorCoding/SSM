//! Redox OS Microkernel Integration Layer for SAIL.
//!
//! Exposes the SAIL Omnimodal Engine to Redox OS userspace programs, daemons,
//! and the shell via the native Redox Scheme architecture (`sail:`).
//!
//! On Redox OS:
//! - Everything is a scheme (`audio:`, `orbital:`, `input:`, `sail:`).
//! - Applications interact through standard `open()`, `read()`, `write()`, and `close()` syscalls.
//! - The `saild` daemon listens on the `:sail` scheme socket and dispatches:
//!   - `sail:command`: Text & Shell command interaction.
//!   - `sail:voice`: Real-time audio streaming from/to the `audio:` sound driver.
//!   - `sail:vision`: Framebuffer / camera capture from the `orbital:` display server.

pub mod redox;

pub use redox::{
    RedoxPacket, RedoxSchemeHandler, SYS_CLOSE, SYS_OPEN, SYS_READ, SYS_WRITE, SailChannel,
    SailDaemon,
};
