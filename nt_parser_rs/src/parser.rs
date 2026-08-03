//! N-Triples streaming parser using rio_turtle
//!
//! Parses N-Triples files in a streaming fashion and batches triples
//! for efficient Delta Lake writing. Supports processing multiple
//! input files into a single Delta table.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};
use rio_api::model::{Literal, NamedNode, Subject, Term, Triple as RioTriple};
use rio_api::parser::TriplesParser;
use rio_turtle::NTriplesParser;
use tracing::{debug, info, warn};

use crate::writer::DeltaTableWriter;

/// Triple data holder for batching
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Get the Arrow schema for the triples table
pub fn triples_schema() -> Schema {
    Schema::new(vec![
        Field::new("subject", DataType::Utf8, false),
        Field::new("predicate", DataType::Utf8, false),
        Field::new("object", DataType::Utf8, false),
    ])
}

/// Convert a rio Subject to string
fn subject_to_string(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(nn) => nn.iri.to_string(),
        Subject::BlankNode(bn) => format!("_:{}", bn.id),
        Subject::Triple(_) => "<quoted_triple>".to_string(),
    }
}

/// Convert a rio NamedNode to string
fn predicate_to_string(predicate: &NamedNode) -> String {
    predicate.iri.to_string()
}

/// Convert a rio Term to string
fn term_to_string(term: &Term) -> String {
    match term {
        Term::NamedNode(nn) => nn.iri.to_string(),
        Term::BlankNode(bn) => format!("_:{}", bn.id),
        Term::Literal(lit) => match lit {
            Literal::Simple { value } => format!("\"{}\"", value),
            Literal::LanguageTaggedString { value, language } => {
                format!("\"{}\"@{}", value, language)
            }
            Literal::Typed { value, datatype } => {
                format!("\"{}\"^^<{}>", value, datatype.iri)
            }
        },
        Term::Triple(_) => "<quoted_triple>".to_string(),
    }
}

/// Convert a batch of triples to an Arrow RecordBatch
pub fn triples_to_record_batch(triples: &[Triple]) -> Result<RecordBatch> {
    let subjects: Vec<&str> = triples.iter().map(|t| t.subject.as_str()).collect();
    let predicates: Vec<&str> = triples.iter().map(|t| t.predicate.as_str()).collect();
    let objects: Vec<&str> = triples.iter().map(|t| t.object.as_str()).collect();

    let subject_array: ArrayRef = Arc::new(StringArray::from(subjects));
    let predicate_array: ArrayRef = Arc::new(StringArray::from(predicates));
    let object_array: ArrayRef = Arc::new(StringArray::from(objects));

    let schema = Arc::new(triples_schema());
    RecordBatch::try_new(schema, vec![subject_array, predicate_array, object_array])
        .context("Failed to create RecordBatch")
}

/// Parse a single N-Triples file, writing batches to the given writer.
/// Returns (triples_written, error_count).
async fn parse_file(
    input_path: &str,
    batch: &mut Vec<Triple>,
    batch_size: usize,
    writer: &mut DeltaTableWriter,
) -> Result<(u64, u64)> {
    let file =
        File::open(input_path).context(format!("Failed to open input file: {}", input_path))?;

    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

    let pb = ProgressBar::new(file_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:45.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta}) | {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    pb.set_message("0 triples parsed");

    let progress_reader = pb.wrap_read(file);
    let reader = BufReader::with_capacity(8 * 1024 * 1024, progress_reader);

    let mut file_triples: u64 = 0;
    let mut error_count: u64 = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                warn!("Error reading line {}: {}", line_num + 1, e);
                error_count += 1;
                continue;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parser = NTriplesParser::new(trimmed.as_bytes());
        let mut parsed = false;

        let parse_result = parser.parse_all(&mut |rio_triple: RioTriple| {
            let triple = Triple {
                subject: subject_to_string(&rio_triple.subject),
                predicate: predicate_to_string(&rio_triple.predicate),
                object: term_to_string(&rio_triple.object),
            };
            batch.push(triple);
            file_triples += 1;
            parsed = true;
            Ok(()) as Result<(), anyhow::Error>
        });

        if let Err(e) = parse_result {
            warn!("Failed to parse line {}: {} - {}", line_num + 1, trimmed, e);
            error_count += 1;
            continue;
        }

        if !parsed {
            debug!("Line {} produced no triple: {}", line_num + 1, trimmed);
        }

        if batch.len() >= batch_size {
            let record_batch = triples_to_record_batch(batch)?;
            writer.write(record_batch).await?;
            debug!(
                "Wrote batch of {} triples (file total: {})",
                batch.len(),
                file_triples
            );
            batch.clear();
            pb.set_message(format!("{} triples parsed", file_triples));
        }
    }

    pb.finish_with_message(format!(
        "{} triples parsed, {} errors",
        file_triples, error_count
    ));

    Ok((file_triples, error_count))
}

/// Parse multiple N-Triples files and write to a single Delta table.
pub async fn parse_and_write(
    input_paths: &[String],
    output_path: &str,
    batch_size: usize,
    overwrite: bool,
) -> Result<()> {
    let mut writer = DeltaTableWriter::new(output_path, overwrite).await?;

    let mut batch: Vec<Triple> = Vec::with_capacity(batch_size);
    let mut total_triples: u64 = 0;
    let mut total_errors: u64 = 0;

    for (i, input_path) in input_paths.iter().enumerate() {
        info!(
            "Parsing file {}/{}: {}",
            i + 1,
            input_paths.len(),
            input_path
        );

        let (file_triples, file_errors) =
            parse_file(input_path, &mut batch, batch_size, &mut writer).await?;

        total_triples += file_triples;
        total_errors += file_errors;

        info!(
            "File {}/{} complete: {} triples, {} errors",
            i + 1,
            input_paths.len(),
            file_triples,
            file_errors
        );
    }

    // Flush remaining triples from the last file
    if !batch.is_empty() {
        let record_batch = triples_to_record_batch(&batch)?;
        writer.write(record_batch).await?;
        debug!("Wrote final batch of {} triples", batch.len());
    }

    writer.close().await?;

    info!(
        "All files complete: {} triples written, {} errors across {} file(s)",
        total_triples,
        total_errors,
        input_paths.len()
    );

    DeltaTableWriter::optimize(output_path).await?;
    info!("Optimize completed");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triples_to_record_batch() {
        let triples = vec![
            Triple {
                subject: "http://example.org/s1".to_string(),
                predicate: "http://example.org/p1".to_string(),
                object: "http://example.org/o1".to_string(),
            },
            Triple {
                subject: "http://example.org/s2".to_string(),
                predicate: "http://example.org/p2".to_string(),
                object: "\"value\"@en".to_string(),
            },
        ];

        let batch = triples_to_record_batch(&triples).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn test_schema() {
        let schema = triples_schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "subject");
        assert_eq!(schema.field(1).name(), "predicate");
        assert_eq!(schema.field(2).name(), "object");
    }
}
