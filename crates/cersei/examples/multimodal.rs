//! # Multimodal Input
//!
//! Attach images, video, audio, or PDFs to a message and send them to any
//! provider. The same provider-agnostic [`ContentBlock`]s work everywhere —
//! each backend takes what it supports (Anthropic: images + PDFs, OpenAI:
//! images + PDF files, Gemini: images + video + audio + PDFs) and drops the
//! rest.
//!
//! ```bash
//! # Pick whichever provider key you have set.
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example multimodal -- path/to/image.png
//! GEMINI_API_KEY=...           cargo run --example multimodal -- clip.mp4 diagram.png
//! ```
//!
//! High-level entry points shown here:
//!   - `ContentBlock::from_path`     — read a file, auto-detect its media type
//!   - `ContentBlock::image_bytes`   — build from raw bytes you already hold
//!   - `ContentBlock::image_url`     — reference a remote image by URL
//!   - `Message::user_with_files`    — text + several local files in one call

use cersei::prelude::*;
use cersei::provider::{CompletionRequest, Provider, ProviderOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: multimodal <file> [<file> ...]   (image / video / audio / pdf)");
        std::process::exit(2);
    }

    // ── Build a multimodal user message in one line ─────────────────────────
    // Each file is read from disk, its MIME type is sniffed from the bytes
    // (with an extension fallback), and it becomes an Image or Document block.
    let message = Message::user_with_files("Describe what you see in detail.", &paths)?;

    // Equivalent lower-level constructors, for reference:
    //   let block = ContentBlock::from_path("diagram.png")?;
    //   let block = ContentBlock::image_bytes("image/png", &std::fs::read("x.png")?);
    //   let block = ContentBlock::image_url("https://example.com/cat.jpg");
    //   let msg   = Message::user_with_media("caption", vec![block]);

    // ── Pick a provider from whatever key is in the environment ─────────────
    let (provider, model): (Box<dyn Provider>, &str) =
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            (Box::new(Anthropic::from_env()?), "claude-sonnet-4-6")
        } else if std::env::var("GEMINI_API_KEY").is_ok() {
            (Box::new(Gemini::from_env()?), "gemini-2.5-flash")
        } else if std::env::var("OPENAI_API_KEY").is_ok() {
            (Box::new(OpenAi::from_env()?), "gpt-5")
        } else {
            anyhow::bail!("set ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY");
        };

    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![message],
        system: Some("You are a careful visual analyst.".into()),
        tools: Vec::new(),
        max_tokens: 1024,
        temperature: None,
        stop_sequences: Vec::new(),
        options: ProviderOptions::default(),
    };

    println!("─── Sending {} file(s) to {model} ───", paths.len());
    let response = provider.complete(request).await?.collect().await?;

    println!("{}", response.message.get_all_text());
    println!("─── Usage ───");
    println!("Input tok:  {}", response.usage.input_tokens);
    println!("Output tok: {}", response.usage.output_tokens);

    Ok(())
}
