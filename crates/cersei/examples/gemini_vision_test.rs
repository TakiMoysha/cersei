//! # Gemini Vision Smoke Test
//!
//! End-to-end check that the multimodal input path actually reaches the model:
//! reads a local PNG via the high-level `ContentBlock::from_path` API, attaches
//! it to a user message, and asks Gemini to (1) describe the image and (2) show
//! how to rebuild the layout in React.
//!
//! ```bash
//! set -a; source .env; set +a
//! cargo run --example gemini_vision_test -- demo.png
//! ```
//!
//! If the printed description matches the actual image contents, vision works.

use cersei::prelude::*;
use cersei::provider::{CompletionRequest, Provider, ProviderOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "demo.png".to_string());

    // High-level API: read file, sniff MIME, base64-encode → Image block.
    let image = ContentBlock::from_path(&path)?;
    if let ContentBlock::Image { source } = &image {
        eprintln!(
            "[loaded] {path} → media_type={:?}, {} base64 bytes",
            source.media_type,
            source.data.as_ref().map(|d| d.len()).unwrap_or(0),
        );
    }

    let message = Message::user_with_media(
        "Two tasks:\n\
         1. Describe exactly what this UI is and every element you can see \
            (header icons, buttons, text, and especially every item in any open \
            dropdown/menu).\n\
         2. Give a concise React + CSS implementation that replicates this layout.",
        vec![image],
    );

    let provider = Gemini::from_env()?;
    let request = CompletionRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: vec![message],
        system: Some("You are a meticulous UI engineer. Be specific and concrete.".into()),
        tools: Vec::new(),
        temperature: Some(0.2),
        // Generous budget: gemini-2.5-flash spends part of maxOutputTokens on
        // hidden "thinking", so a small cap truncates the visible answer.
        max_tokens: 8192,
        stop_sequences: Vec::new(),
        options: {
            // Disable dynamic thinking so the full budget goes to the answer.
            let mut o = ProviderOptions::default();
            o.set("thinking_budget", 0);
            o
        },
    };

    eprintln!("[sending] gemini-2.5-flash …\n");
    let response = provider.complete(request).await?.collect().await?;

    println!("========== GEMINI RESPONSE ==========");
    println!("{}", response.message.get_all_text());
    println!("=====================================");
    eprintln!(
        "\n[usage] input={} output={} tokens  stop_reason={:?}",
        response.usage.input_tokens, response.usage.output_tokens, response.stop_reason
    );

    Ok(())
}
