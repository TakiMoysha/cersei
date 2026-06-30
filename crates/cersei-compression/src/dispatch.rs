//! Route a raw tool output through the right compression stage based on the
//! tool name and its input JSON.
//!
//! Every invocation emits a single `tracing::info!` event on the
//! `cersei_compression` target with before/after bytes, lines, savings
//! percent, strategy, and the matched rule / language. Subscribers can filter
//! with `RUST_LOG=cersei_compression=info`.

use crate::{ansi, code, level::CompressionLevel, toml_rules, truncate};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MAX_LINES_SAFETY: usize = 600;

/// Structured before/after compression metrics for a single tool output.
/// Surfaced on the event stream so consumers don't have to scrape the log line.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompressionStats {
    pub before_bytes: usize,
    pub after_bytes: usize,
    pub before_lines: usize,
    pub after_lines: usize,
    /// Percentage of bytes removed (0.0 when nothing was compressed).
    pub savings_pct: f64,
}

impl CompressionStats {
    fn measure(before: &str, after: &str) -> Self {
        let before_bytes = before.len();
        let after_bytes = after.len();
        let savings_pct = if before_bytes > 0 {
            100.0 * (before_bytes as f64 - after_bytes as f64) / before_bytes as f64
        } else {
            0.0
        };
        Self {
            before_bytes,
            after_bytes,
            before_lines: before.lines().count(),
            after_lines: after.lines().count(),
            savings_pct,
        }
    }
}

/// Infallible entry point. On any internal panic-free error we fall back to
/// the raw content unchanged so the agent loop never breaks.
pub fn compress_tool_output(
    tool_name: &str,
    tool_input: &Value,
    content: &str,
    level: CompressionLevel,
) -> String {
    compress_tool_output_with_stats(tool_name, tool_input, content, level).0
}

/// Like [`compress_tool_output`], but also returns the [`CompressionStats`] so
/// callers can surface the savings (e.g. on an event) instead of scraping the
/// `cersei_compression` log line.
pub fn compress_tool_output_with_stats(
    tool_name: &str,
    tool_input: &Value,
    content: &str,
    level: CompressionLevel,
) -> (String, CompressionStats) {
    if level.is_off() || content.is_empty() {
        return (content.to_string(), CompressionStats::measure(content, content));
    }

    let lowered = tool_name.to_ascii_lowercase();

    // strategy = short tag describing the branch we took
    // detail   = finer identifier (rule name, detected language, empty string, …)
    let (out, strategy, detail): (String, &'static str, String) = match lowered.as_str() {
        // ─── Shell-like tools → TOML rules DSL ───────────────────────────
        "bash" | "exec" | "execshell" | "shell" | "run" | "runshell" => {
            let command = tool_input
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("");
            let (out, rule) = compress_command(command, content, level);
            (out, "shell", rule)
        }

        // ─── File read → code filter ────────────────────────────────────
        "read" | "readfile" | "read_file" | "view" => {
            let path = tool_input
                .get("file_path")
                .or_else(|| tool_input.get("path"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let lang = code::Language::from_path(path);
            let filtered = code::filter(content, lang, level);
            let capped = safety_cap(&filtered, level);
            (capped, "code", format!("{lang:?}"))
        }

        // ─── Structured retrieval tools pass straight through ────────────
        "grep" | "glob" | "list" | "ls" | "find" | "tree" => {
            (content.to_string(), "passthrough", String::new())
        }

        // ─── Web fetch → strip ANSI + generic TOML catch-all ─────────────
        "webfetch" | "web_fetch" | "fetch" | "http" => {
            let stripped = ansi::strip_ansi(content);
            let (out, rule) = compress_command("webfetch", &stripped, level);
            (out, "web", rule)
        }

        // ─── Anything else → minimal safety cap at Aggressive, else noop ─
        _ => {
            if matches!(level, CompressionLevel::Aggressive) {
                (safety_cap(content, level), "unknown-capped", String::new())
            } else {
                (content.to_string(), "unknown", String::new())
            }
        }
    };

    let stats = CompressionStats::measure(content, &out);
    log_compression(tool_name, level, strategy, &detail, &stats);
    (out, stats)
}

/// Returns (filtered_output, matched_rule_name_or_empty).
fn compress_command(command: &str, content: &str, level: CompressionLevel) -> (String, String) {
    let stripped = ansi::strip_ansi(content);
    let (out, rule) = if let Some(filter) = toml_rules::find_matching(command.trim()) {
        (toml_rules::apply(filter, &stripped), filter.name.clone())
    } else {
        (stripped, String::new())
    };
    (safety_cap(&out, level), rule)
}

fn safety_cap(content: &str, level: CompressionLevel) -> String {
    let cap = match level {
        CompressionLevel::Off => return content.to_string(),
        CompressionLevel::Minimal => MAX_LINES_SAFETY,
        CompressionLevel::Aggressive => MAX_LINES_SAFETY / 2,
    };
    if content.lines().count() <= cap {
        content.to_string()
    } else {
        truncate::smart_truncate(content, cap)
    }
}

fn log_compression(
    tool: &str,
    level: CompressionLevel,
    strategy: &str,
    detail: &str,
    stats: &CompressionStats,
) {
    tracing::info!(
        target: "cersei_compression",
        tool,
        level = %level,
        strategy,
        detail,
        before_bytes = stats.before_bytes,
        after_bytes = stats.after_bytes,
        before_lines = stats.before_lines,
        after_lines = stats.after_lines,
        savings_pct = format!("{:.1}", stats.savings_pct),
        "tool-output compressed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn off_is_noop() {
        let raw = "\x1b[31mhello\x1b[0m";
        let out = compress_tool_output("Bash", &json!({}), raw, CompressionLevel::Off);
        assert_eq!(out, raw);
    }

    #[test]
    fn stats_report_byte_savings() {
        let raw = "\x1b[31mfatal: not a git repo\x1b[0m";
        let (out, stats) = compress_tool_output_with_stats(
            "Bash",
            &json!({ "command": "git status" }),
            raw,
            CompressionLevel::Minimal,
        );
        assert_eq!(stats.before_bytes, raw.len());
        assert_eq!(stats.after_bytes, out.len());
        assert!(out.len() < raw.len(), "expected ANSI stripping to shrink output");
        assert!(stats.savings_pct > 0.0, "savings_pct should be positive: {stats:?}");
    }

    #[test]
    fn stats_zero_savings_when_off() {
        let raw = "hello world";
        let (_out, stats) =
            compress_tool_output_with_stats("Bash", &json!({}), raw, CompressionLevel::Off);
        assert_eq!(stats.savings_pct, 0.0);
        assert_eq!(stats.before_bytes, stats.after_bytes);
    }

    #[test]
    fn bash_strips_ansi_at_minimal() {
        let raw = "\x1b[31mfatal: not a git repo\x1b[0m";
        let out = compress_tool_output(
            "Bash",
            &json!({"command": "git status"}),
            raw,
            CompressionLevel::Minimal,
        );
        assert!(!out.contains("\x1b["));
        assert!(out.contains("fatal"));
    }

    #[test]
    fn read_preserves_json_when_data_file() {
        let raw = r#"{"a": 1, "packages": ["x/*"]}"#;
        let out = compress_tool_output(
            "Read",
            &json!({"file_path": "/x/package.json"}),
            raw,
            CompressionLevel::Aggressive,
        );
        assert!(out.contains("packages"));
        assert!(out.contains("x/*"));
    }

    #[test]
    fn read_strips_rust_comments_in_aggressive() {
        let raw = "\
// normal comment
/// doc comment
fn main() {
    let x = 1;
    println!(\"{}\", x);
}
";
        let out = compress_tool_output(
            "Read",
            &json!({"file_path": "src/main.rs"}),
            raw,
            CompressionLevel::Aggressive,
        );
        assert!(!out.contains("// normal comment"));
        assert!(out.contains("fn main"));
    }

    #[test]
    fn grep_passthrough() {
        let raw = "file.rs:1:hit\nfile.rs:2:hit2";
        let out = compress_tool_output(
            "Grep",
            &json!({"pattern": "hit"}),
            raw,
            CompressionLevel::Aggressive,
        );
        assert_eq!(out, raw);
    }
}
