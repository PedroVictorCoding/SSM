//! Publish complete files to concurrent readers; never rename an open writer.
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    atomic_write_before_commit(path, bytes, || Ok(()))
}

fn atomic_write_before_commit(
    path: &Path,
    bytes: &[u8],
    before_commit: impl FnOnce() -> std::io::Result<()>,
) -> std::io::Result<()> {
    let name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("output filename required"))?
        .to_string_lossy();
    let temp = path.with_file_name(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        before_commit()?;
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interrupted_publication_preserves_previous_file_and_cleans_temp() {
        let dir =
            std::env::temp_dir().join(format!("sail-publication-failure-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.sea");
        atomic_write(&path, b"previous committed artifact").unwrap();
        let failure = atomic_write_before_commit(&path, b"uncommitted replacement", || {
            Err(std::io::Error::other("injected interruption before rename"))
        });
        assert!(
            failure
                .unwrap_err()
                .to_string()
                .contains("injected interruption")
        );
        assert_eq!(fs::read(&path).unwrap(), b"previous committed artifact");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        atomic_write(&path, b"next complete artifact").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"next complete artifact");
        fs::remove_dir_all(dir).unwrap();
    }
}
