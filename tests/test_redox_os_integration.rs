use std::fs;
use std::path::PathBuf;

use sail::backend::{InferenceBackend, default_device};
use sail::data::{CharTokenizer, Tokenizer};
use sail::engine::Engine;
use sail::export::pack_to_artifact;
use sail::model::HybridLmConfig;
use sail::os::{
    RedoxPacket, RedoxSchemeHandler, SYS_CLOSE, SYS_OPEN, SYS_READ, SYS_WRITE, SailDaemon,
};

#[test]
fn test_redox_scheme_packet_lifecycle() {
    let handler = RedoxSchemeHandler::new(None);

    // 1. SYS_OPEN on command channel (a=1)
    let open_pkt = RedoxPacket {
        id: 1,
        op: SYS_OPEN,
        a: 1,
        b: 0,
        c: 0,
        d: 0,
    };
    let handle_id = handler.handle_packet(&open_pkt);
    assert!(handle_id > 0 && handle_id != usize::MAX);

    // 2. SYS_WRITE to send a command
    let input = b"ls -la /scheme\n";
    let write_pkt = RedoxPacket {
        id: 2,
        op: SYS_WRITE,
        a: handle_id,
        b: input.as_ptr() as usize,
        c: input.len(),
        d: 0,
    };
    let bytes_written = handler.handle_packet(&write_pkt);
    assert_eq!(bytes_written, input.len());

    // 3. SYS_READ response from saild
    let mut read_buf = vec![0u8; 128];
    let read_pkt = RedoxPacket {
        id: 3,
        op: SYS_READ,
        a: handle_id,
        b: read_buf.as_mut_ptr() as usize,
        c: read_buf.len(),
        d: 0,
    };
    let bytes_read = handler.handle_packet(&read_pkt);
    assert!(bytes_read > 0);
    let output = String::from_utf8_lossy(&read_buf[..bytes_read]);
    assert!(output.contains("[SAIL-RedoxOS]: Received 'ls -la /scheme'"));

    // 4. SYS_CLOSE
    let close_pkt = RedoxPacket {
        id: 4,
        op: SYS_CLOSE,
        a: handle_id,
        b: 0,
        c: 0,
        d: 0,
    };
    assert_eq!(handler.handle_packet(&close_pkt), 0);
}

#[test]
fn test_redox_daemon_with_live_engine_inference() {
    let test_dir = PathBuf::from("target/test_redox_daemon");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    let model_path = test_dir.join("model.sea");
    let device = default_device();

    let text = "Redox microkernel OS voice and text command assistant. \n";
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(text));
    let cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8)
        .with_d_conv(2);

    let model = cfg.init::<InferenceBackend>(&device);
    pack_to_artifact(&model, &cfg, &tokenizer, &model_path).unwrap();

    let engine = Engine::<InferenceBackend>::open(&model_path, &device)
        .expect("Engine must open valid .sea");

    let daemon = SailDaemon::new(Some(engine));

    // Send high-level text command via Redox scheme daemon
    let response = daemon
        .send_command("Redox")
        .expect("Daemon send_command must succeed");

    assert!(response.starts_with("Redox"));
    assert!(response.len() > "Redox".len());

    let _ = fs::remove_dir_all(&test_dir);
}
