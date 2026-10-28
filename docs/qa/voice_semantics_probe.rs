//! Prints observations from synthetic, non-speech audio; never runs tools or records a microphone.
use sail::voice::AudioChunk;
use sail::voice::asr::{AcousticFeatureAsr, AsrIntentGate, StreamingAsrAdapter};
fn main() {
    let sample_rate=16000;
    for frequency in [900.0f32, 1200.0, 1800.0] {
        let samples=(0..19200).map(|i| {
            (std::f32::consts::TAU * frequency * i as f32 / sample_rate as f32).sin()*0.35
        }).collect();
        let asr=AcousticFeatureAsr::new(sample_rate,"en");
        let chunk=AudioChunk::new(samples,sample_rate,1,0,0);
        asr.feed_audio("non-speech-fixture",&chunk).unwrap();
        let segment=asr.finalize_utterance("non-speech-fixture").unwrap().unwrap();
        println!("{frequency}Hz sine, no spoken words: text={:?}, confidence={:.3}, final={}, timestamp span={}ns, sample span={}",
            segment.text,segment.confidence,segment.is_final,
            segment.end_timestamp_ns-segment.start_timestamp_ns,segment.end_offset-segment.start_offset);
        let mut gate=AsrIntentGate::new();
        let first=gate.ingest_segment(segment.clone()).is_some();
        let duplicate=gate.ingest_segment(segment).is_some();
        println!("same final utterance delivered twice: first accepted={first}, duplicate accepted={duplicate}");
    }
}
