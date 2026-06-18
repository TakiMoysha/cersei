//! Live end-to-end AgentRL test against the real Gemini provider.
//!
//! Ignored by default (makes network calls). Run explicitly after sourcing the
//! key into the environment (never put the key on the command line):
//!
//!   set -a; . ./.env; set +a
//!   cargo test -p cersei-agentrl --test live_gemini -- --ignored --nocapture
//!
//! It hands AgentRL a real coding task with an INDEPENDENT verifier (the agent
//! cannot cheat the check), runs the full orchestrator, asserts the produced
//! artifact actually works, and prints which solve path AgentRL took.

use cersei_agentrl::orchestrator::{Orchestrator, OrchestratorConfig};
use cersei_agentrl::verify::Verifier;
use cersei_agentrl::{CerseiRunner, CommandVerifier, Solved, ToolRegistry};
use cersei_provider::{Gemini, Provider};
use std::sync::Arc;

fn api_key() -> Option<String> {
    std::env::var("GEMINI_API_KEY")
        .or_else(|_| std::env::var("GOOGLE_API_KEY"))
        .ok()
        .filter(|k| !k.is_empty())
}

#[tokio::test]
#[ignore = "live network test; run with --ignored after sourcing .env"]
async fn agentrl_solves_a_real_coding_task() {
    let Some(key) = api_key() else {
        eprintln!("SKIP: no GEMINI_API_KEY/GOOGLE_API_KEY in env");
        return;
    };

    // Isolated working dir + persisted registry dir.
    let work = tempfile::tempdir().unwrap();
    let reg_dir = tempfile::tempdir().unwrap();
    let registry = ToolRegistry::open(reg_dir.path()).unwrap();

    // The task: a real little CLI program with edge cases.
    let task = "Create a Python script named `gcd.py` in the working directory. It takes two \
        integer command-line arguments and prints (only) their greatest common divisor computed \
        with the Euclidean algorithm. Example: `python3 gcd.py 48 36` prints `12`. Handle the \
        case where one argument is 0 (gcd(0, n) == n).";

    // INDEPENDENT verifier — our own checks, not the agent's. Exercises edge cases.
    let verifier: Arc<dyn Verifier> = Arc::new(CommandVerifier::new(
        "python3 gcd.py 48 36 | grep -qx 12 && \
         python3 gcd.py 1071 462 | grep -qx 21 && \
         python3 gcd.py 0 7 | grep -qx 7 && \
         python3 gcd.py 17 5 | grep -qx 1",
    ));

    // Provider factory: capture the key in-memory only; never persisted.
    let provider_factory: cersei_agentrl::ProviderFactory =
        Arc::new(move || Box::new(Gemini::new(key.clone())) as Box<dyn Provider>);

    // NOTE: the model must be set explicitly — cersei-agent's runner defaults to
    // an Anthropic model when none is given, which a Gemini key cannot serve.
    let runner = Arc::new(
        CerseiRunner::new(provider_factory, work.path(), registry.clone(), verifier.clone())
            .with_model("gemini-3.1-pro-preview")
            .with_max_turns(16),
    );

    let orch = Orchestrator::new(runner, registry.clone()).with_config(OrchestratorConfig {
        max_rl_rounds: 1,
        num_proposals: 2,
        registry_search_k: 5,
        session_id: "live".into(),
            num_samples: 1,
    });

    let outcome = orch.solve(task).await.expect("orchestrator ran");

    // ── Report what AgentRL actually did ──
    eprintln!("\n──────── AgentRL run report ────────");
    eprintln!("solved: {}", outcome.solved);
    eprintln!("path:   {:?}", outcome.how);
    eprintln!(
        "graph:  {} nodes, {} failed",
        outcome.last_graph.nodes.len(),
        outcome
            .last_graph
            .nodes
            .iter()
            .filter(|n| n.status == cersei_agentrl::NodeStatus::Failed)
            .count()
    );
    eprintln!("registry entries: {}", registry.len());
    match &outcome.how {
        Some(Solved::Directly) => eprintln!("→ GeneralAgent solved it on the first pass."),
        Some(Solved::ByNewTool(id)) => {
            eprintln!("→ RL loop fired: GeneralAgent failed, a sandboxed proposal passed and was registered as {id}.")
        }
        Some(Solved::ByCachedTool(id)) => eprintln!("→ Solved via a cached tool {id}."),
        None => eprintln!("→ NOT solved."),
    }
    eprintln!("────────────────────────────────────\n");

    // ── Assert the artifact genuinely works (independent of `solved`) ──
    assert!(outcome.solved, "AgentRL did not solve the task");
    let final_check = verifier.verify(work.path()).await;
    assert!(
        final_check.passed,
        "the produced gcd.py failed the independent verifier: {}",
        final_check.detail
    );
}

/// Forces the RL RECOVERY loop with a live LLM: the GeneralAgent is restricted to
/// a read-only toolset, so it CANNOT create the file and fails the verifier. The
/// orchestrator then traces the failure, the PlannerAgent proposes fixes, the
/// proposals run with full coding tools in isolated sandboxes, a winner is
/// promoted, and the solution is registered as a reusable tool.
#[tokio::test]
#[ignore = "live network test; run with --ignored after sourcing .env"]
async fn agentrl_recovery_loop_registers_a_tool() {
    let Some(key) = api_key() else {
        eprintln!("SKIP: no GEMINI_API_KEY/GOOGLE_API_KEY in env");
        return;
    };
    let work = tempfile::tempdir().unwrap();
    let reg_dir = tempfile::tempdir().unwrap();
    let registry = ToolRegistry::open(reg_dir.path()).unwrap();

    let task = "Create a Python script named `greet.py` in the working directory that prints \
        exactly `hello, world` when run as `python3 greet.py`.";
    let verifier: Arc<dyn Verifier> =
        Arc::new(CommandVerifier::new("python3 greet.py | grep -qx 'hello, world'"));

    let key2 = key.clone();
    let provider_factory: cersei_agentrl::ProviderFactory =
        Arc::new(move || Box::new(Gemini::new(key2.clone())) as Box<dyn Provider>);

    // GeneralAgent gets a READ-ONLY toolset → guaranteed first-pass failure →
    // forces escalation into the planner/proposal/register loop.
    let readonly: cersei_agentrl::ToolsFactory = Arc::new(|| {
        vec![Box::new(cersei_tools::file_read::FileReadTool) as Box<dyn cersei_tools::Tool>]
    });

    let runner = Arc::new(
        CerseiRunner::new(provider_factory, work.path(), registry.clone(), verifier.clone())
            .with_model("gemini-3.1-pro-preview")
            .with_max_turns(12)
            .with_general_tools(readonly),
    );

    let orch = Orchestrator::new(runner, registry.clone()).with_config(OrchestratorConfig {
        max_rl_rounds: 1,
        num_proposals: 2,
        registry_search_k: 5,
        session_id: "recovery".into(),
            num_samples: 1,
    });

    let outcome = orch.solve(task).await.expect("orchestrator ran");

    eprintln!("\n──────── AgentRL recovery report ────────");
    eprintln!("solved: {}  path: {:?}", outcome.solved, outcome.how);
    eprintln!("registry entries after run: {}", registry.len());
    for e in registry.all() {
        eprintln!("  registered tool: {} ({})", e.name, e.tool_id);
    }
    eprintln!("─────────────────────────────────────────\n");

    assert!(outcome.solved, "recovery loop failed to solve the task");
    assert!(
        matches!(outcome.how, Some(Solved::ByNewTool(_))),
        "expected the RL loop to register a NEW tool, got {:?}",
        outcome.how
    );
    assert_eq!(registry.len(), 1, "a reusable tool should be registered");
    // The promoted winner must actually satisfy the independent verifier.
    assert!(verifier.verify(work.path()).await.passed, "promoted artifact does not work");
}
