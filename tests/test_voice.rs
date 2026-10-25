use sail::voice::{
    AudioHardware, AudioRingBuffer, DEFAULT_SAMPLE_RATE, VadDetector, VoiceAsr, VoiceTts,
};

#[test]
fn test_audio_ring_buffer_fifo() {
    let ring = AudioRingBuffer::new(1024);
    assert_eq!(ring.len(), 0);
    assert!(ring.is_empty());

    let samples = vec![0.1f32, 0.2, -0.3, 0.4];
    ring.push_samples(&samples);
    assert_eq!(ring.len(), 4);

    let drained = ring.drain_samples();
    assert_eq!(drained, samples);
    assert_eq!(ring.len(), 0);
}

#[test]
fn test_vad_speech_activity_and_turn_detection() {
    let mut vad = VadDetector::new(0.05, 60); // 3 silence frames needed

    let silence_frame = vec![0.001f32; 320];
    let speech_frame = vec![0.2f32; 320];

    // Silence: not speaking
    let (is_active, turn_ended) = vad.process_frame(&silence_frame);
    assert!(!is_active);
    assert!(!turn_ended);

    // Speech: speaking
    let (is_active, turn_ended) = vad.process_frame(&speech_frame);
    assert!(is_active);
    assert!(!turn_ended);

    // 1st silence frame during speech
    let (is_active, turn_ended) = vad.process_frame(&silence_frame);
    assert!(is_active);
    assert!(!turn_ended);

    // 2nd silence frame
    let (is_active, turn_ended) = vad.process_frame(&silence_frame);
    assert!(is_active);
    assert!(!turn_ended);

    // 3rd silence frame: turn finished!
    let (is_active, turn_ended) = vad.process_frame(&silence_frame);
    assert!(!is_active);
    assert!(turn_ended);
}

#[test]
fn test_voice_tts_synthesis_and_wav_generation() {
    let tmp_wav = std::env::temp_dir().join("ssm_test_speech.wav");
    let _ = std::fs::remove_file(&tmp_wav);

    let tts = VoiceTts::new(DEFAULT_SAMPLE_RATE);
    let speech_text = "The speed of light is verified.";
    let audio = tts.synthesize(speech_text);

    assert!(!audio.is_empty());
    assert!(audio.len() > 16000 / 2); // At least 0.5s of audio

    // Save to WAV file
    tts.save_wav(&audio, &tmp_wav).expect("saving WAV file");
    assert!(tmp_wav.exists());

    // Verify WAV with hound reader
    let reader = hound::WavReader::open(&tmp_wav).expect("reading WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(spec.bits_per_sample, 16);
    assert_eq!(reader.duration() as usize, audio.len());

    let _ = std::fs::remove_file(&tmp_wav);
}

#[test]
fn test_asr_and_hardware_presence() {
    let asr = VoiceAsr::new(DEFAULT_SAMPLE_RATE);
    let sample_audio = vec![0.1f32; 8000];
    let transcription = asr.transcribe(&sample_audio);
    assert!(!transcription.is_empty());

    // Hardware check does not panic even if no mic attached
    let _has_device = AudioHardware::has_audio_device();
}
