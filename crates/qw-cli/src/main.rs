//! `qwen38` — command line entry point.

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
        #[arg(long, default_value_t = 128)]
        tokens: usize,
    },
    /// One-shot generation.
    Gen {
        #[arg(long, default_value = "models/Qwen3.8-27B-4bit")]
        model_dir: PathBuf,
        #[arg(long, default_value = "Hello!")]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = true)]
        greedy: bool,
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
        Cmd::Bench { model_dir, tokens } => cmd_bench(model_dir, tokens),
        Cmd::Gen {
            model_dir,
            prompt,
            max_tokens,
            greedy,
        } => cmd_gen(model_dir, prompt, max_tokens, greedy),
    }
}

fn cmd_info(model_dir: &std::path::Path) -> Result<()> {
    let cfg = ModelConfig::load(model_dir)?;
    let t = &cfg.text_config;
    println!("== Qwen3.8-27B ==");
    println!("model_type         : {:?}", cfg.model_type);
    println!("hidden / inter     : {} / {}", t.hidden_size, t.intermediate_size);
    println!("layers             : {} ({} linear + {} full)",
        t.num_hidden_layers, t.num_linear_layers(), t.num_full_layers());
    println!("heads (q/kv, dim)  : {}/{}, head_dim={}", t.num_attention_heads, t.num_key_value_heads, t.head_dim);
    println!("linear attn        : {} v-heads x {}, {} k-heads x {}, conv={}",
        t.linear_num_value_heads, t.linear_value_head_dim,
        t.linear_num_key_heads, t.linear_key_head_dim, t.linear_conv_kernel_dim);
    println!("vocab / ctx        : {} / {}", t.vocab_size, t.max_position_embeddings);
    println!("rope               : theta={} partial={} rotary_dim={}",
        t.rope_theta(), t.partial_rotary_factor(), t.rotary_dim());
    println!("MTP layers         : {}", t.mtp_num_hidden_layers);

    let dev = qw_metal::GpuDevice::new()?;
    let store = qw_weights::WeightStore::load_dir(&dev, model_dir)?;
    let zero_copy = store.shards.iter().filter(|s| s.zero_copy).count();
    println!("\n== weights ==");
    println!("shards             : {} ({} zero-copy mmap aliased)", store.shards.len(), zero_copy);
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

fn cmd_bench(_model_dir: PathBuf, _tokens: usize) -> Result<()> {
    anyhow::bail!("bench: kernel work lands in milestone M1/M3 (see loop/LOOP.md)")
}

fn cmd_gen(_model_dir: PathBuf, _prompt: String, _max_tokens: usize, _greedy: bool) -> Result<()> {
    anyhow::bail!("gen: forward pass lands in milestone M1/M2 (see loop/LOOP.md)")
}
