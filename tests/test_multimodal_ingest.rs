use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sail::backend::{InferenceBackend, default_device};
use sail::data::{
    AudioData, CharTokenizer, IncomingMultimodalWatcher, OmnimodalDataset, OmnimodalRecord,
    Tokenizer, VisionData, extract_mel_spectrogram, load_ppm_image, load_wav_as_mel,
};

#[test]
fn test_mel_spectrogram_extraction_and_wav_loading() {
    let test_dir = PathBuf::from("target/test_mel_wav");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    // 1. Synthetic 16kHz sine wave audio (400ms = 6400 samples)
    let sample_rate = 16000;
    let n_samples = 6400;
    let freq = 440.0;
    let mut pcm_samples = Vec::with_capacity(n_samples);
    for n in 0..n_samples {
        let t = n as f32 / sample_rate as f32;
        let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
        pcm_samples.push(s);
    }

    // 2. Compute Mel spectrogram directly
    let n_mels = 80;
    let mel = extract_mel_spectrogram(&pcm_samples, sample_rate, n_mels, 400, 160);
    assert_eq!(mel.channels, n_mels);
    assert!(mel.frames > 30);
    assert_eq!(mel.data.len(), n_mels * mel.frames);

    // 3. Write WAV file via hound
    let wav_path = test_dir.join("test_tone.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
    for &s in &pcm_samples {
        let i = (s * i16::MAX as f32) as i16;
        writer.write_sample(i).unwrap();
    }
    writer.finalize().unwrap();

    // 4. Load WAV as Mel spectrogram
    let loaded_mel = load_wav_as_mel(&wav_path, n_mels).unwrap();
    assert_eq!(loaded_mel.channels, n_mels);
    assert_eq!(loaded_mel.frames, mel.frames);

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_ppm_image_loading() {
    let test_dir = PathBuf::from("target/test_ppm_load");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    // Create a 4x4 binary P6 PPM
    let ppm_path = test_dir.join("test_img.ppm");
    let mut file = File::create(&ppm_path).unwrap();
    file.write_all(b"P6\n# Test Image\n4 4\n255\n").unwrap();
    // 16 pixels * 3 channels = 48 bytes
    let raw_bytes = vec![128u8; 48];
    file.write_all(&raw_bytes).unwrap();

    let vision = load_ppm_image(&ppm_path).unwrap();
    assert_eq!(vision.channels, 3);
    assert_eq!(vision.height, 4);
    assert_eq!(vision.width, 4);
    assert_eq!(vision.data.len(), 48);

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_omnimodal_dataset_sampling_and_watcher_ingest() {
    let test_dir = PathBuf::from("target/test_omni_ingest");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    let incoming = test_dir.join("incoming");
    fs::create_dir_all(&incoming).unwrap();

    let text_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:!?-_\n";
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(text_chars));
    let device = default_device();

    // Initial seed dataset
    let seed_record = OmnimodalRecord {
        text: Some("Initial Prompt".into()),
        target: "Initial Target".into(),
        audio: Some(AudioData {
            channels: 80,
            frames: 10,
            data: vec![0.1; 800],
        }),
        vision: Some(VisionData {
            channels: 3,
            height: 16,
            width: 16,
            data: vec![0.5; 3 * 16 * 16],
        }),
        audio_file: None,
        image_file: None,
        provenance: None,
    };

    let dataset = OmnimodalDataset::new(vec![seed_record], 0.3);
    assert_eq!(dataset.len(), 1);

    // Watcher monitoring incoming
    let mut watcher = IncomingMultimodalWatcher::new(&incoming).unwrap();

    // Drop new chunk: chunk_01.omni.jsonl
    let chunk_file = incoming.join("chunk_01.omni.jsonl");
    let record1 = OmnimodalRecord {
        text: Some("Audio prompt".into()),
        target: "Voice acknowledged".into(),
        audio: Some(AudioData {
            channels: 80,
            frames: 8,
            data: vec![0.2; 640],
        }),
        vision: None,
        audio_file: None,
        image_file: None,
        provenance: None,
    };
    let record2 = OmnimodalRecord {
        text: None,
        target: "A cat sleeping".into(),
        audio: None,
        vision: Some(VisionData {
            channels: 3,
            height: 16,
            width: 16,
            data: vec![0.8; 3 * 16 * 16],
        }),
        audio_file: None,
        image_file: None,
        provenance: None,
    };

    let jsonl = format!(
        "{}\n{}\n",
        serde_json::to_string(&record1).unwrap(),
        serde_json::to_string(&record2).unwrap()
    );
    fs::write(&chunk_file, jsonl).unwrap();

    let ingested = watcher.scan_and_ingest(&dataset, 80);
    assert_eq!(ingested, 2);
    assert_eq!(dataset.len(), 3);

    // Second scan should not re-ingest the same chunk file
    let second_scan = watcher.scan_and_ingest(&dataset, 80);
    assert_eq!(second_scan, 0);
    assert_eq!(dataset.len(), 3);

    // Sample an OmnimodalBatch from dataset
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let batch = dataset
        .sample_batch::<InferenceBackend, _>(2, &tokenizer, &mut rng, &device)
        .expect("Must sample multimodal batch");

    assert!(batch.text_inputs.is_some());
    assert!(batch.audio_frames.is_some());
    assert!(batch.vision_frame.is_some());
    let [tb, _tl] = batch.targets.dims();
    assert_eq!(tb, 2);

    let _ = fs::remove_dir_all(&test_dir);
}
