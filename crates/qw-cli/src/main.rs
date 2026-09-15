//! `qwen38` — command line entry point.

mod bench;
mod check;
mod gen;

use anyhow::Result;
use clap::{Parser, Subcommand};
use qw_model::{ModelConfig, WeightLayout};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "qwen38",
    version,
    about = "Rust + Metal inference engine for Qwen3.8-27B FP4 on Apple Silicon"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Inspect the checkpoint: config, layer plan, weight coverage.
    Info {
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
    },
    /// Serve OpenAI- and Anthropic-compatible endpoints.
    Serve {
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "qwen3.8-27b-fp4")]
        model_id: String,
    },
    /// Benchmark decode throughput (tok/s).
    Bench {
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 3)]
        iters: usize,
        /// verify this many tokens per weight sweep (speculative verification)
        #[arg(long, default_value_t = 1)]
        tokens: usize,
        /// output rows per threadgroup; 0 = round-3 kernel, >=1 = x-hoisted variant
        #[arg(long, default_value_t = 0)]
        rows: usize,
    },
    /// Validate the 4-bit GEMV kernel against a CPU reference on real weights.
    Check {
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
        #[arg(long)]
        tensor: Option<String>,
        #[arg(long, default_value_t = 0)]
        samples: usize,
    },
    /// End-to-end parity gate against the mlx-lm oracle.
    Verify {
        #[arg(long, default_value = "loop/artifacts/oracle.json")]
        oracle: PathBuf,
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
    },
    /// One-shot generation.
    Gen {
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
        #[arg(long, default_value = "Hello!")]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        max_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        max_t: usize,
        /// print the first-token top-N logits (oracle comparison)
        #[arg(long, default_value_t = 0)]
        dump_top: usize,
        /// print per-layer residual-stream statistics
        #[arg(long)]
        dump_hidden: bool,
        /// write every layer's hidden vector (f32, layer-major) to a file
        #[arg(long)]
        dump_vectors: Option<String>,
        /// ignore EOS (throughput measurement)
        #[arg(long)]
        no_stop: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,qw_metal=info".into()),
        )
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { model_dir } => cmd_info(&model_dir),
        Cmd::Serve {
            model_dir,
            port,
            model_id,
        } => cmd_serve(model_dir, port, model_id),
        Cmd::Bench {
            model_dir,
            iters,
            tokens,
            rows,
        } => bench::run(&model_dir, iters, tokens, rows),
        Cmd::Check {
            model_dir,
            tensor,
            samples,
        } => check::run(&model_dir, tensor.as_deref(), samples),
        Cmd::Verify { oracle, model_dir } => cmd_verify(oracle, model_dir),
        Cmd::Gen {
            model_dir,
            prompt,
            max_tokens,
            max_t,
            dump_top,
            dump_hidden,
            dump_vectors,
            no_stop,
        } => gen::run(gen::GenOpts {
            model_dir: &model_dir,
            prompt: &prompt,
            max_tokens,
            max_t,
            dump_top,
            dump_hidden,
            dump_vectors,
            stop_at_eos: !no_stop,
        }),
    }
}

fn cmd_info(model_dir: &std::path::Path) -> Result<()> {
    let cfg = ModelConfig::load(model_dir)?;
    let t = &cfg.text_config;
    println!("== Qwen3.8-27B ==");
    println!("model_type         : {:?}", cfg.model_type);
    println!(
        "hidden / inter     : {} / {}",
        t.hidden_size, t.intermediate_size
    );
    println!(
        "layers             : {} ({} linear + {} full)",
        t.num_hidden_layers,
        t.num_linear_layers(),
        t.num_full_layers()
    );
    println!(
        "heads (q/kv, dim)  : {}/{}, head_dim={}",
        t.num_attention_heads, t.num_key_value_heads, t.head_dim
    );
    println!(
        "linear attn        : {} v-heads x {}, {} k-heads x {}, conv={}",
        t.linear_num_value_heads,
        t.linear_value_head_dim,
        t.linear_num_key_heads,
        t.linear_key_head_dim,
        t.linear_conv_kernel_dim
    );
    println!(
        "vocab / ctx        : {} / {}",
        t.vocab_size, t.max_position_embeddings
    );
    println!(
        "rope               : theta={} partial={} rotary_dim={}",
        t.rope_theta(),
        t.partial_rotary_factor(),
        t.rotary_dim()
    );
    println!("MTP layers         : {}", t.mtp_num_hidden_layers);

    let dev = qw_metal::GpuDevice::new()?;
    let store = qw_weights::WeightStore::load_dir(&dev, model_dir)?;
    let zero_copy = store.shards.iter().filter(|s| s.zero_copy).count();
    println!("\n== weights ==");
    println!(
        "shards             : {} ({} zero-copy mmap aliased)",
        store.shards.len(),
        zero_copy
    );
    println!("tensors            : {}", store.index.len());

    let layout = WeightLayout::default();
    let mut missing = Vec::new();
    let check = |name: String, missing: &mut Vec<String>| {
        if !store.has(&name) {
            missing.push(name);
        }
    };
    check(layout.embed_tokens(), &mut missing);
    check(layout.final_norm(), &mut missing);
    check(layout.lm_head(), &mut missing);
    for i in 0..t.num_hidden_layers {
        check(layout.input_norm(i), &mut missing);
        check(layout.post_attn_norm(i), &mut missing);
        check(layout.mlp_gate(i), &mut missing);
        check(layout.mlp_up(i), &mut missing);
        check(layout.mlp_down(i), &mut missing);
        if t.is_linear_layer(i) {
            check(layout.gdn_in_qkv(i), &mut missing);
            check(layout.gdn_in_z(i), &mut missing);
            check(layout.gdn_in_b(i), &mut missing);
            check(layout.gdn_in_a(i), &mut missing);
            check(layout.gdn_out(i), &mut missing);
            check(layout.gdn_conv(i), &mut missing);
            check(layout.gdn_norm(i), &mut missing);
            check(layout.gdn_a_log(i), &mut missing);
            check(layout.gdn_dt_bias(i), &mut missing);
        } else {
            check(layout.attn_q(i), &mut missing);
            check(layout.attn_k(i), &mut missing);
            check(layout.attn_v(i), &mut missing);
            check(layout.attn_o(i), &mut missing);
            check(layout.attn_q_norm(i), &mut missing);
            check(layout.attn_k_norm(i), &mut missing);
        }
    }
    println!("expected-tensor check: {} missing", missing.len());
    for m in missing.iter().take(10) {
        println!("   MISSING {m}");
    }

    let mtp_present = store.has("mtp.fc.weight");
    println!("MTP weights in shard: {}", mtp_present);
    Ok(())
}

fn cmd_serve(model_dir: PathBuf, port: u16, model_id: String) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let state = qw_server::AppState::new(model_id, model_dir.to_string_lossy().to_string());
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        qw_server::serve(state, addr).await
    })
}

fn cmd_verify(oracle: PathBuf, model_dir: PathBuf) -> Result<()> {
    if !oracle.exists() {
        println!(
            "verify: no oracle at {} — running real-weight kernel check",
            oracle.display()
        );
        return check::run(&model_dir, None, 8);
    }
    // Token-for-token parity against mlx-lm is the real correctness gate.
    match gen::check_against_oracle(&model_dir, &oracle, 2048, None) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("parity failed ({e}); falling back to the kernel-level check");
            check::run(&model_dir, None, 8)?;
            Err(e)
        }
    }
}
