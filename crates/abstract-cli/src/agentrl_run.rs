//! `--agentrl` mode: solve a single task with the self-evolving AgentRL
//! orchestrator instead of a one-shot agent.
//!
//! Designed for headless / terminal-bench runs: it resolves a provider from the
//! `--model` string, builds a verifier that prefers a task-provided test script
//! (`run-tests.sh`) and falls back to accepting the agent's result, runs the
//! `Orchestrator` in the working directory, and prints a machine-readable result
//! line compatible with the harbor adapter.

use crate::config::AppConfig;
use cersei_agentrl::{
    CerseiRunner, ChainVerifier, Orchestrator, OrchestratorConfig, ProviderFactory, Solved,
    TestScriptVerifier, ToolRegistry, Verifier,
};
use cersei_provider::Provider;
use std::sync::Arc;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn registry_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ABSTRACT_AGENTRL_REGISTRY") {
        return std::path::PathBuf::from(p);
    }
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".cersei/agentrl")
}

pub async fn run(prompt: &str, config: &AppConfig) -> anyhow::Result<()> {
    let model_string = config.model.clone();

    // Validate the provider/model up front and resolve the bare model name.
    let (_probe, resolved_model) = cersei_provider::from_model_string(&model_string)
        .map_err(|e| anyhow::anyhow!("AgentRL: cannot resolve provider for '{model_string}': {e}"))?;

    // Fresh provider per agent (sub-agents get their own client).
    let ms = model_string.clone();
    let provider_factory: ProviderFactory = Arc::new(move || {
        cersei_provider::from_model_string(&ms)
            .expect("provider resolution already validated")
            .0 as Box<dyn Provider>
    });

    let workdir = config.working_dir.clone();
    let registry = ToolRegistry::open(registry_dir())
        .map_err(|e| anyhow::anyhow!("AgentRL: registry open failed: {e}"))?;

    // Verifier: use a real in-container test script when the task ships one
    // (the only trustworthy success signal). When none exists, `default_passed`
    // accepts the GeneralAgent's single attempt — we must NOT speculatively
    // "verify" with an accept-anything check, or the recovery loop would promote
    // and register garbage that the real (hidden) grader rejects. So the chain
    // has only the test-script verifier; absent a script it returns the default.
    let verifier: Arc<dyn Verifier> = Arc::new(
        ChainVerifier::new(vec![Arc::new(TestScriptVerifier::default_candidates())])
            .with_default(true),
    );

    // Fold learned failure patterns (injected by the bench harness) into the task.
    let task = match std::env::var("ABSTRACT_FAILURE_PATTERNS") {
        Ok(p) if !p.trim().is_empty() => format!(
            "{prompt}\n\n[hard-won guidance from prior runs — heed these]:\n{p}"
        ),
        _ => prompt.to_string(),
    };

    let runner = Arc::new(
        CerseiRunner::new(provider_factory, workdir.clone(), registry.clone(), verifier)
            .with_model(&resolved_model)
            .with_max_turns(config.max_turns),
    );

    // Recovery (plan → sandboxed proposals) is OFF by default for terminal-bench:
    // it only helps when proposals can be verified against a real in-container
    // grader, which TB hides. Enable it (ABSTRACT_AGENTRL_ROUNDS>0) only for
    // tasks/datasets that expose a runnable test script. Best-of-N (samples) and
    // the recovery loop both add value only when TestScriptVerifier applies;
    // otherwise this behaves exactly like the single-shot baseline.
    let cfg = OrchestratorConfig {
        max_rl_rounds: env_u32("ABSTRACT_AGENTRL_ROUNDS", 0),
        num_proposals: env_u32("ABSTRACT_AGENTRL_PROPOSALS", 2) as usize,
        registry_search_k: 5,
        session_id: "agentrl-bench".to_string(),
        num_samples: env_u32("ABSTRACT_AGENTRL_SAMPLES", 1),
    };

    let orchestrator = Orchestrator::new(runner, registry.clone()).with_config(cfg);

    let json_mode = config.output_format == "stream-json";

    let outcome = orchestrator.solve(&task).await;

    match outcome {
        Ok(o) => {
            let how = match &o.how {
                Some(Solved::Directly) => "directly",
                Some(Solved::ByCachedTool(_)) => "cached_tool",
                Some(Solved::ByNewTool(_)) => "new_tool",
                None => "unsolved",
            };
            eprintln!(
                "[agentrl] solved={} via={} registry={} model={}",
                o.solved,
                how,
                registry.len(),
                resolved_model
            );
            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({
                        "type": "agentrl_result",
                        "solved": o.solved,
                        "how": how,
                        "registry_size": registry.len(),
                    })
                );
            }
        }
        Err(e) => {
            // Do not fail the process — terminal-bench grades the filesystem,
            // not our exit code. Surface the error and move on.
            eprintln!("[agentrl] error: {e}");
            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({ "type": "agentrl_result", "solved": false, "error": e.to_string() })
                );
            }
        }
    }

    Ok(())
}
