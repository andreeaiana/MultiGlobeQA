//! N-Triples to Delta Lake parser CLI
//!
//! High-performance streaming parser that converts N-Triples files
//! to Delta Lake tables with a simple (subject, predicate, object) schema.
//! Supports multiple input files and glob patterns for split KG datasets.

use anyhow::{bail, Result};
use clap::Parser;
use glob::glob;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod parser;
mod writer;

/// N-Triples to Delta Lake converter
#[derive(Parser, Debug)]
#[command(name = "nt_parser")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input N-Triples file path(s) or glob pattern(s).
    /// Accepts multiple paths and/or patterns like "data/*.nt".
    #[arg(short, long, num_args = 1..)]
    input: Vec<String>,

    /// Output Delta table path
    #[arg(short, long)]
    output: String,

    /// Batch size for writing (number of triples per batch)
    #[arg(long, default_value = "100000")]
    batch_size: usize,

    /// Overwrite existing Delta table
    #[arg(long, default_value = "false")]
    overwrite: bool,
}

/// Resolve a list of paths/globs into a sorted list of concrete file paths.
fn resolve_inputs(patterns: &[String]) -> Result<Vec<String>> {
    let mut paths = Vec::new();
    for pattern in patterns {
        let matches: Vec<_> = glob(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", pattern, e))?
            .collect();

        if matches.is_empty() {
            bail!("No files matched pattern: {}", pattern);
        }

        for entry in matches {
            let path = entry
                .map_err(|e| anyhow::anyhow!("Error reading glob entry: {}", e))?;
            paths.push(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("Non-UTF8 path: {:?}", path))?
                    .to_string(),
            );
        }
    }

    paths.sort();
    paths.dedup();

    if paths.is_empty() {
        bail!("No input files found");
    }

    Ok(paths)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let input_files = resolve_inputs(&args.input)?;

    info!(
        "Starting N-Triples to Delta conversion: {} file(s) -> {}",
        input_files.len(),
        args.output
    );
    for f in &input_files {
        info!("  {}", f);
    }

    parser::parse_and_write(&input_files, &args.output, args.batch_size, args.overwrite).await?;

    info!("Conversion complete!");
    Ok(())
}
