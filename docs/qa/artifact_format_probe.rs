use sail::runtime::validate_model_format;
fn main() {
    let root = std::env::temp_dir().join(format!("sail-format-review-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (name,bytes) in [("native-header.sea",b"SEAPACK1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".as_slice()),("zip.bin",b"PKxx".as_slice()),("truncated.gguf",b"GGUF".as_slice())] {
        let path=root.join(name);std::fs::write(&path,bytes).unwrap();
        println!("{name}: {:?}",validate_model_format(&path));
    }
    let model=std::path::Path::new("runs/hf_gpu_20261025_075531/model.sea");
    if model.exists() {println!("existing exported artifact: {:?}",validate_model_format(model));}
    std::fs::remove_dir_all(root).unwrap();
}
