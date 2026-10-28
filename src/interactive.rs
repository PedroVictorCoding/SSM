//! Human-directed terminal configuration for the `sail` binary.
//!
//! The wizard only constructs arguments for the existing command handlers. It
//! never starts a workflow without showing the equivalent command and receiving
//! explicit confirmation from the terminal user.

use std::fmt::Display;
use std::io::{self, BufRead, Write};
use std::str::FromStr;

pub fn run() -> Result<Option<Vec<String>>, String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_with_io(stdin.lock(), stdout.lock())
}

fn run_with_io<R: BufRead, W: Write>(
    mut input: R,
    mut output: W,
) -> Result<Option<Vec<String>>, String> {
    writeln!(output, "SAIL interactive setup").map_err(io_error)?;
    writeln!(
        output,
        "Choose a workflow. Press Enter to accept a [default].\n"
    )
    .map_err(io_error)?;
    writeln!(output, "  1. Run or chat with a model").map_err(io_error)?;
    writeln!(output, "  2. Train from a text dataset").map_err(io_error)?;
    writeln!(output, "  3. Continual stream training").map_err(io_error)?;
    writeln!(output, "  4. Export a run to model.sea").map_err(io_error)?;
    writeln!(output, "  5. Generate from a training run").map_err(io_error)?;
    writeln!(output, "  6. Show command help").map_err(io_error)?;
    writeln!(output, "  0. Cancel").map_err(io_error)?;

    let choice: usize = prompt_parsed(&mut input, &mut output, "Workflow", 1, |v| *v <= 6)?;
    let args = match choice {
        0 => return cancelled(&mut output),
        1 => configure_run(&mut input, &mut output)?,
        2 => configure_train(&mut input, &mut output)?,
        3 => configure_stream_train(&mut input, &mut output)?,
        4 => configure_export(&mut input, &mut output)?,
        5 => configure_generate(&mut input, &mut output)?,
        6 => return Ok(Some(vec!["help".into()])),
        _ => unreachable!(),
    };

    writeln!(
        output,
        "\nEquivalent reproducible command:\n  {}",
        render_command(&args)
    )
    .map_err(io_error)?;
    if !prompt_bool(&mut input, &mut output, "Run this now", false)? {
        return cancelled(&mut output);
    }
    Ok(Some(args))
}

fn configure_run<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
) -> Result<Vec<String>, String> {
    let mut args = vec!["run".into()];
    option(
        &mut args,
        "model",
        prompt_text(input, output, "Model file", "model.sea")?,
    );
    let interactive = prompt_bool(input, output, "Open conversational chat", true)?;
    if interactive {
        flag(&mut args, "interactive");
    } else {
        option(
            &mut args,
            "prompt",
            prompt_text(input, output, "Prompt", "Hello")?,
        );
    }
    option(
        &mut args,
        "tokens",
        prompt_number(input, output, "Maximum generated tokens", 200usize)?,
    );
    option(
        &mut args,
        "temperature",
        prompt_number(input, output, "Temperature (0 = greedy)", 0.7f64)?,
    );
    option(
        &mut args,
        "top-k",
        prompt_number(input, output, "Top-k (0 = disabled)", 40usize)?,
    );
    let image = prompt_text(input, output, "Optional PPM image path", "")?;
    if !image.is_empty() {
        option(&mut args, "image", image);
    }
    let audio = prompt_text(input, output, "Optional WAV audio path", "")?;
    if !audio.is_empty() {
        option(&mut args, "audio", audio);
    }
    Ok(args)
}

fn configure_train<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
) -> Result<Vec<String>, String> {
    let mut args = vec!["train".into()];
    option(
        &mut args,
        "data",
        prompt_text(
            input,
            output,
            "Training text file",
            "data/tinyshakespeare.txt",
        )?,
    );
    option(
        &mut args,
        "out",
        prompt_text(input, output, "Run directory", "runs/shakespeare")?,
    );
    if prompt_bool(input, output, "Resume this run", false)? {
        flag(&mut args, "resume");
    }
    if prompt_bool(input, output, "Use BPE tokenization", true)? {
        flag(&mut args, "bpe");
    }
    add_training_basics(input, output, &mut args, 2000, 0)?;
    if prompt_bool(
        input,
        output,
        "Configure advanced model and optimizer settings",
        false,
    )? {
        add_advanced_training(input, output, &mut args)?;
    }
    Ok(args)
}

fn configure_stream_train<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
) -> Result<Vec<String>, String> {
    let mut args = vec!["stream-train".into()];
    option(
        &mut args,
        "incoming",
        prompt_text(input, output, "Incoming data directory", "data/incoming")?,
    );
    let initial = prompt_text(input, output, "Optional initial dataset", "")?;
    if !initial.is_empty() {
        option(&mut args, "initial-data", initial);
    }
    option(
        &mut args,
        "out",
        prompt_text(input, output, "Run directory", "runs/live_growing")?,
    );
    if prompt_bool(input, output, "Resume this run", false)? {
        flag(&mut args, "resume");
    }
    if prompt_bool(input, output, "Enable multimodal records", false)? {
        flag(&mut args, "multimodal");
    }
    if prompt_bool(input, output, "Use BPE tokenization", true)? {
        flag(&mut args, "bpe");
    }
    option(
        &mut args,
        "min-tokens",
        prompt_number(input, output, "Tokens required before training", 256usize)?,
    );
    option(
        &mut args,
        "replay-ratio",
        prompt_number(input, output, "Historical replay ratio", 0.30f64)?,
    );
    option(
        &mut args,
        "scan-interval",
        prompt_number(input, output, "Directory scan interval (ms)", 1000u64)?,
    );
    add_training_basics(input, output, &mut args, 2000, 100)?;
    if prompt_bool(
        input,
        output,
        "Configure advanced model and optimizer settings",
        false,
    )? {
        add_advanced_training(input, output, &mut args)?;
    }
    Ok(args)
}

fn add_training_basics<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    args: &mut Vec<String>,
    default_steps: usize,
    default_save_every: usize,
) -> Result<(), String> {
    option(
        args,
        "steps",
        prompt_number(input, output, "Total training steps", default_steps)?,
    );
    option(
        args,
        "batch-size",
        prompt_number(input, output, "Batch size", 32usize)?,
    );
    option(
        args,
        "seq-len",
        prompt_number(input, output, "Sequence length", 128usize)?,
    );
    option(
        args,
        "save-every",
        prompt_number(
            input,
            output,
            "Save every N steps (0 = on evaluation)",
            default_save_every,
        )?,
    );
    option(
        args,
        "eval-every",
        prompt_number(input, output, "Evaluate every N steps", 250usize)?,
    );
    option(
        args,
        "log-every",
        prompt_number(input, output, "Print progress every N steps", 25usize)?,
    );
    option(
        args,
        "viz-every",
        prompt_number(
            input,
            output,
            "Update visualization every N steps (0 = off)",
            25usize,
        )?,
    );
    Ok(())
}

fn add_advanced_training<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    args: &mut Vec<String>,
) -> Result<(), String> {
    option(
        args,
        "lr",
        prompt_number(input, output, "Peak learning rate", 0.002f64)?,
    );
    option(
        args,
        "min-lr",
        prompt_number(input, output, "Minimum learning rate", 0.0001f64)?,
    );
    option(
        args,
        "warmup",
        prompt_number(input, output, "Warmup steps", 100usize)?,
    );
    option(
        args,
        "grad-clip",
        prompt_number(input, output, "Gradient clipping norm", 1.0f64)?,
    );
    option(
        args,
        "d-model",
        prompt_number(input, output, "Model width", 128usize)?,
    );
    option(
        args,
        "n-layers",
        prompt_number(input, output, "Number of layers", 4usize)?,
    );
    option(
        args,
        "d-state",
        prompt_number(input, output, "SSM state size", 16usize)?,
    );
    option(
        args,
        "scan-chunk",
        prompt_number(input, output, "Scan chunk size", 64usize)?,
    );
    option(
        args,
        "moe-experts",
        prompt_number(input, output, "MoE experts (0 = dense)", 0usize)?,
    );
    option(
        args,
        "moe-top-k",
        prompt_number(input, output, "Active experts per token", 1usize)?,
    );
    let attention = prompt_text(
        input,
        output,
        "Attention layer indices (comma list or none)",
        "none",
    )?;
    option(args, "attn-layers", attention);
    if prompt_bool(input, output, "Enable synchronized profiling", false)? {
        flag(args, "profile");
    }
    if prompt_bool(input, output, "Use on-device gradient clipping", false)? {
        flag(args, "device-clip");
    }
    Ok(())
}

fn configure_export<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
) -> Result<Vec<String>, String> {
    let mut args = vec!["export".into()];
    option(
        &mut args,
        "run",
        prompt_text(input, output, "Training run directory", "runs/shakespeare")?,
    );
    option(
        &mut args,
        "out",
        prompt_text(input, output, "Output model file", "model.sea")?,
    );
    Ok(args)
}

fn configure_generate<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
) -> Result<Vec<String>, String> {
    let mut args = vec!["generate".into()];
    option(
        &mut args,
        "run",
        prompt_text(input, output, "Training run directory", "runs/shakespeare")?,
    );
    option(
        &mut args,
        "prompt",
        prompt_text(input, output, "Prompt", "ROMEO:")?,
    );
    option(
        &mut args,
        "tokens",
        prompt_number(input, output, "Maximum generated tokens", 300usize)?,
    );
    option(
        &mut args,
        "temperature",
        prompt_number(input, output, "Temperature (0 = greedy)", 0.8f64)?,
    );
    option(
        &mut args,
        "top-k",
        prompt_number(input, output, "Top-k (0 = disabled)", 0usize)?,
    );
    Ok(args)
}

fn prompt_text<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    label: &str,
    default: &str,
) -> Result<String, String> {
    write!(output, "{label} [{default}]: ").map_err(io_error)?;
    output.flush().map_err(io_error)?;
    let mut line = String::new();
    if input.read_line(&mut line).map_err(io_error)? == 0 {
        return Err("interactive setup cancelled: end of input".into());
    }
    let value = line.trim();
    Ok(if value.is_empty() {
        default.to_string()
    } else {
        value.to_string()
    })
}

fn prompt_parsed<R, W, T, F>(
    input: &mut R,
    output: &mut W,
    label: &str,
    default: T,
    valid: F,
) -> Result<T, String>
where
    R: BufRead,
    W: Write,
    T: FromStr + Display + Copy,
    F: Fn(&T) -> bool,
{
    loop {
        let raw = prompt_text(input, output, label, &default.to_string())?;
        match raw.parse::<T>() {
            Ok(value) if valid(&value) => return Ok(value),
            _ => writeln!(output, "Invalid value; please try again.").map_err(io_error)?,
        }
    }
}

fn prompt_number<R, W, T>(
    input: &mut R,
    output: &mut W,
    label: &str,
    default: T,
) -> Result<T, String>
where
    R: BufRead,
    W: Write,
    T: FromStr + Display + Copy,
{
    prompt_parsed(input, output, label, default, |_| true)
}

fn prompt_bool<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    label: &str,
    default: bool,
) -> Result<bool, String> {
    let suffix = if default { "Y/n" } else { "y/N" };
    loop {
        write!(output, "{label} [{suffix}]: ").map_err(io_error)?;
        output.flush().map_err(io_error)?;
        let mut line = String::new();
        if input.read_line(&mut line).map_err(io_error)? == 0 {
            return Err("interactive setup cancelled: end of input".into());
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => writeln!(output, "Please answer yes or no.").map_err(io_error)?,
        }
    }
}

fn option(args: &mut Vec<String>, name: &str, value: impl Display) {
    args.push(format!("--{name}"));
    args.push(value.to_string());
}

fn flag(args: &mut Vec<String>, name: &str) {
    args.push(format!("--{name}"));
}

fn render_command(args: &[String]) -> String {
    let rendered = args
        .iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ");
    format!("sail {rendered}")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-._/:,".contains(c))
    {
        value.to_string()
    } else {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

fn cancelled<W: Write>(output: &mut W) -> Result<Option<Vec<String>>, String> {
    writeln!(output, "No command was run.").map_err(io_error)?;
    Ok(None)
}

fn io_error(error: io::Error) -> String {
    format!("terminal I/O: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn cancellation_never_constructs_a_command() {
        let mut output = Vec::new();
        let result = run_with_io(Cursor::new(b"0\n"), &mut output).unwrap();
        assert_eq!(result, None);
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("No command was run")
        );
    }

    #[test]
    fn invalid_menu_value_retries_then_returns_help() {
        let mut output = Vec::new();
        let result = run_with_io(Cursor::new(b"not-a-number\n9\n6\n"), &mut output).unwrap();
        assert_eq!(result, Some(vec!["help".into()]));
        assert_eq!(
            String::from_utf8(output)
                .unwrap()
                .matches("Invalid value")
                .count(),
            2
        );
    }

    #[test]
    fn export_wizard_constructs_reproducible_arguments_after_confirmation() {
        let input = b"4\nruns/my model\nbuild/my model.sea\ny\n";
        let mut output = Vec::new();
        let result = run_with_io(Cursor::new(input), &mut output)
            .unwrap()
            .unwrap();
        assert_eq!(
            result,
            vec![
                "export",
                "--run",
                "runs/my model",
                "--out",
                "build/my model.sea"
            ]
        );
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("sail export --run \"runs/my model\"")
        );
    }

    #[test]
    fn run_wizard_uses_safe_visible_defaults() {
        let input = b"1\n\n\n\n\n\n\n\ny\n";
        let result = run_with_io(Cursor::new(input), Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(
            result,
            vec![
                "run",
                "--model",
                "model.sea",
                "--interactive",
                "--tokens",
                "200",
                "--temperature",
                "0.7",
                "--top-k",
                "40"
            ]
        );
    }

    #[test]
    fn train_wizard_constructs_default_training_configuration() {
        let mut answers = vec!["2".to_string()];
        answers.extend(std::iter::repeat_n(String::new(), 12));
        answers.push("y".to_string());
        let input = format!("{}\n", answers.join("\n"));
        let result = run_with_io(Cursor::new(input.as_bytes()), Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(result.first().map(String::as_str), Some("train"));
        assert!(
            result
                .windows(2)
                .any(|v| v == ["--data", "data/tinyshakespeare.txt"])
        );
        assert!(result.iter().any(|v| v == "--bpe"));
        assert!(result.windows(2).any(|v| v == ["--steps", "2000"]));
        assert!(result.windows(2).any(|v| v == ["--viz-every", "25"]));
        assert!(!result.iter().any(|v| v == "--resume"));
    }

    #[test]
    fn eof_cancels_instead_of_starting_defaults() {
        let result = run_with_io(Cursor::new(Vec::<u8>::new()), Vec::new());
        assert!(result.unwrap_err().contains("end of input"));
    }
}
