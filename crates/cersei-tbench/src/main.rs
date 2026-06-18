//! tbench-agent — a purpose-built Terminal-Bench coding agent.
//!
//! A thin, fast binary over the Cersei SDK + AgentRL: a TB-specialized coding
//! agent (Gemini 3.1 Pro by default) with the full IO/coding toolset, a high
//! turn budget, and a verifier that trusts a real in-container test script when
//! one exists (enabling best-of-N / recovery), and otherwise runs a single
//! strong attempt. File changes land in the working directory for grading.

mod prompt;

use cersei_agentrl::{
    CerseiRunner, ChainVerifier, Orchestrator, OrchestratorConfig, ProviderFactory, Solved,
    TestScriptVerifier, ToolRegistry, Verifier,
};
use cersei_provider::Provider;
use clap::Parser;
use std::io::Read;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "tbench-agent",
    version,
    about = "Purpose-built Terminal-Bench coding agent (Cersei SDK + AgentRL)"
)]
struct Cli {
    /// Task instruction (positional). Alternatively use -p, or pipe via stdin.
    task: Option<String>,

    /// Task instruction.
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Provider/model string (e.g. vertex/claude-opus-4-8, google/gemini-3.1-pro-preview).
    #[arg(long, default_value = "vertex/claude-opus-4-8")]
    model: String,

    /// Tool-output compression for token efficiency: off | minimal | aggressive.
    #[arg(long, default_value = "minimal")]
    compress: String,

    /// Working directory (default: current dir).
    #[arg(short = 'C', long)]
    dir: Option<String>,

    /// Best-of-N: retry the attempt up to N times, keeping the first that passes
    /// the verifier. Only adds value when a real test script is present.
    #[arg(long, default_value_t = 1)]
    samples: u32,

    /// Recovery rounds: on verified failure, plan + run sandboxed proposals.
    /// Only meaningful with a real in-container test script (default off).
    #[arg(long, default_value_t = 0)]
    rounds: u32,

    /// Proposals per recovery round.
    #[arg(long, default_value_t = 2)]
    proposals: usize,

    /// Max agent turns per attempt.
    #[arg(long, default_value_t = 80)]
    max_turns: u32,

    /// Emit a machine-readable JSON result line on stdout.
    #[arg(long)]
    json: bool,

    /// Tool-registry directory (default: ephemeral temp dir).
    #[arg(long)]
    registry: Option<String>,
}

fn resolve_task(cli: &Cli) -> anyhow::Result<String> {
    if let Some(t) = cli.prompt.clone().or_else(|| cli.task.clone()) {
        if !t.trim().is_empty() {
            return Ok(t);
        }
    }
    // fall back to stdin
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    if buf.trim().is_empty() {
        anyhow::bail!("no task provided (positional arg, -p, or stdin)");
    }
    Ok(buf)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let instruction = resolve_task(&cli)?;

    // Validate the provider/model and resolve the bare model name.
    let (_probe, resolved_model) = cersei_provider::from_model_string(&cli.model)
        .map_err(|e| anyhow::anyhow!("cannot resolve provider for '{}': {e}", cli.model))?;

    let model = cli.model.clone();
    let provider_factory: ProviderFactory = Arc::new(move || {
        cersei_provider::from_model_string(&model)
            .expect("provider resolution already validated")
            .0 as Box<dyn Provider>
    });

    // Connectivity probe: surface provider/network errors directly (a static
    // musl binary can fail at DNS/TLS before the agent loop ever runs a tool).
    {
        let p = (provider_factory)();
        let mut req = cersei_provider::CompletionRequest::new(&resolved_model);
        req.messages = vec![cersei_types::Message::user("ping")];
        req.max_tokens = 8;
        match p.complete_blocking(req).await {
            Ok(r) => eprintln!(
                "[tbench-agent] probe OK: {:?}",
                r.message.get_text().map(|t| t.chars().take(40).collect::<String>())
            ),
            Err(e) => eprintln!("[tbench-agent] probe ERR: {e}"),
        }
    }

    let workdir = cli
        .dir
        .clone()
        .map(std::path::PathBuf::from)
        .unwrap_or(std::env::current_dir()?);

    // Trust only a real in-container test script; otherwise accept the single
    // attempt (default-pass) — never speculatively "verify" with accept-anything.
    let verifier: Arc<dyn Verifier> = Arc::new(
        ChainVerifier::new(vec![Arc::new(TestScriptVerifier::default_candidates())])
            .with_default(true),
    );
    // Proposals must NEVER win by default — only on a genuine test-script pass.
    // (A recovery proposal that can't be verified must not be promoted/registered.)
    let proposal_verifier: Arc<dyn Verifier> = Arc::new(
        ChainVerifier::new(vec![Arc::new(TestScriptVerifier::default_candidates())])
            .with_default(false),
    );

    let registry_dir = cli
        .registry
        .clone()
        .or_else(|| std::env::var("TBENCH_REGISTRY").ok())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cersei-tbench-registry"));
    let registry = ToolRegistry::open(&registry_dir)
        .map_err(|e| anyhow::anyhow!("registry open failed: {e}"))?;

    // Optional extra hints (failure patterns) injected by the harness.
    let hints = std::env::var("TBENCH_HINTS")
        .ok()
        .or_else(|| std::env::var("ABSTRACT_FAILURE_PATTERNS").ok());
    let task = prompt::build_task(&instruction, hints.as_deref());

    let runner = Arc::new(
        CerseiRunner::new(provider_factory, workdir.clone(), registry.clone(), verifier)
            .with_proposal_verifier(proposal_verifier)
            .with_model(&resolved_model)
            .with_max_turns(cli.max_turns)
            .with_compression(
                cli.compress
                    .parse::<cersei_compression::CompressionLevel>()
                    .unwrap_or(cersei_compression::CompressionLevel::Off),
            )
            .with_system_prompt(prompt::TBENCH_SYSTEM_PROMPT),
    );

    let cfg = OrchestratorConfig {
        max_rl_rounds: cli.rounds,
        num_proposals: cli.proposals,
        registry_search_k: 5,
        session_id: "tbench".to_string(),
        num_samples: cli.samples,
    };

    let orchestrator = Orchestrator::new(runner, registry.clone()).with_config(cfg);

    let outcome = orchestrator.solve(&task).await;

    match outcome {
        Ok(o) => {
            let how = match &o.how {
                Some(Solved::Directly) => "directly",
                Some(Solved::ByCachedTool(_)) => "cached_tool",
                Some(Solved::ByNewTool(_)) => "new_tool",
                None => "unsolved",
            };
            // Diagnostics: how much work actually happened. A near-empty graph
            // means the agent never really ran (e.g. provider/network failure);
            // a rich graph means it worked and the task was just hard.
            let nodes = o.last_graph.nodes.len();
            let tool_calls = o
                .last_graph
                .nodes
                .iter()
                .filter(|n| matches!(n.kind, cersei_agentrl::NodeKind::ToolCall))
                .count();
            eprintln!(
                "[tbench-agent] solved={} via={} graph_nodes={} tool_calls={} answer={:?} model={} dir={}",
                o.solved,
                how,
                nodes,
                tool_calls,
                o.answer.chars().take(200).collect::<String>(),
                resolved_model,
                workdir.display()
            );
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "type": "tbench_result",
                        "solved": o.solved,
                        "how": how,
                    })
                );
            }
        }
        Err(e) => {
            eprintln!("[tbench-agent] error: {e}");
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({ "type": "tbench_result", "solved": false, "error": e.to_string() })
                );
            }
        }
    }

    // Always exit 0 — the grader scores the filesystem, not our exit code.
    Ok(())
}
