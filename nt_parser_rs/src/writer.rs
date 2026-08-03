//! Delta Lake writer using delta-rs
//!
//! Handles writing Arrow RecordBatches to Delta Lake tables.
//! Periodic flush-and-commit is used to avoid accumulating all batches
//! in memory before the single final flush — which would produce one
//! enormous Parquet file for planet-scale inputs.

use anyhow::{Context, Result};
use arrow::record_batch::RecordBatch;
use deltalake::kernel::{DataType as DeltaDataType, PrimitiveType, StructField, StructType};
use deltalake::operations::create::CreateBuilder;
use deltalake::protocol::SaveMode;
use deltalake::writer::{DeltaWriter, RecordBatchWriter};
use deltalake::DeltaTable;
use deltalake::DeltaOps;
use deltalake::open_table;
use deltalake::operations::optimize::OptimizeType;
use tracing::info;

/// Number of rows to accumulate before flushing to a new Parquet file.
/// At 1 M rows × ~200 bytes/row (three URI strings) ≈ 200 MB per file,
/// which is a comfortable Delta Lake file size.
pub const FLUSH_ROWS: u64 = 1_000_000;

/// Delta table writer wrapper
pub struct DeltaTableWriter {
    table: DeltaTable,
    writer: RecordBatchWriter,
    /// Running count of rows buffered since the last flush.
    buffered_rows: u64,
}

impl DeltaTableWriter {
    /// Create a new Delta table writer
    pub async fn new(output_path: &str, overwrite: bool) -> Result<Self> {
        // Define the Delta schema
        let delta_schema = StructType::new(vec![
            StructField::new(
                "subject",
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
            StructField::new(
                "predicate",
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
            StructField::new(
                "object",
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
        ]);

        // Create or overwrite the Delta table
        let save_mode = if overwrite {
            SaveMode::Overwrite
        } else {
            SaveMode::ErrorIfExists
        };

        info!("Creating Delta table at: {}", output_path);

        let table = CreateBuilder::new()
            .with_location(output_path)
            .with_columns(delta_schema.fields().cloned())
            .with_save_mode(save_mode)
            .await
            .context("Failed to create Delta table")?;

        let writer = RecordBatchWriter::for_table(&table)?;

        Ok(Self {
            table,
            writer,
            buffered_rows: 0,
        })
    }

    /// Write a RecordBatch to the Delta table.
    ///
    pub async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        let batch_rows = batch.num_rows() as u64;

        self.writer
            .write(batch)
            .await
            .context("Failed to write batch to Delta table")?;

        self.buffered_rows += batch_rows;

        if self.buffered_rows >= FLUSH_ROWS {
            self.flush_and_commit().await?;
        }

        Ok(())
    }

    /// Flush the internal buffer to a Parquet file, commit the resulting
    pub async fn flush_and_commit(&mut self) -> Result<()> {
        if self.buffered_rows == 0 {
            return Ok(());
        }

        let adds = self
            .writer
            .flush()
            .await
            .context("Failed to flush Delta writer")?;

        if adds.is_empty() {
            self.buffered_rows = 0;
            return Ok(());
        }

        let actions: Vec<deltalake::kernel::Action> = adds
            .into_iter()
            .map(deltalake::kernel::Action::Add)
            .collect();

        let snapshot = self.table.snapshot().context("Failed to get snapshot")?;
        let log_store = self.table.log_store().clone();

        let precommit = deltalake::operations::transaction::CommitBuilder::default()
            .with_actions(actions)
            .build(
                Some(snapshot),
                log_store,
                deltalake::protocol::DeltaOperation::Write {
                    mode: SaveMode::Append,
                    partition_by: None,
                    predicate: None,
                },
            );

        let _post_commit = precommit
            .await
            .map_err(|e| anyhow::anyhow!("Failed to commit transaction: {}", e))?;

        // Reload the in-memory table state so the next flush_and_commit() call
        // uses the current snapshot version.  Without this, self.table.snapshot()
        // returns the stale version from construction time, causing delta-rs to
        // emit an OCC conflict warning ("Attempting to write transaction N but
        // table has been updated to M") on every single flush after the first.
        self.table
            .load()
            .await
            .context("Failed to reload Delta table state after commit")?;

        info!(
            "Flushed and committed {} rows (Delta version {})",
            self.buffered_rows,
            self.table.version()
        );

        // Reset counter and recreate the writer against the now-current table
        // so the next batch goes into a fresh Parquet file.
        self.buffered_rows = 0;
        self.writer = RecordBatchWriter::for_table(&self.table)
            .context("Failed to recreate RecordBatchWriter after flush")?;

        Ok(())
    }

    /// Close the writer, flushing any remaining buffered rows, then finalize.
    pub async fn close(mut self) -> Result<()> {
        // Flush whatever is still in the buffer (the tail of the input file)
        self.flush_and_commit().await?;

        info!(
            "Delta table finalized at version {}",
            self.table.version()
        );

        Ok(())
    }

    pub async fn optimize(output_path: &str) -> Result<()> {
        let table = open_table(output_path).await?;
        let (_table, metrics) = DeltaOps(table)
            .optimize()
            .with_type(OptimizeType::Compact)
            .await?;
        info!("Optimization statistics {:?}", metrics);
        Ok(())
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flush_rows_constant() {
        // Sanity-check: flush threshold must be positive and at most 10 M
        assert!(FLUSH_ROWS > 0);
        assert!(FLUSH_ROWS <= 10_000_000);
    }

    #[tokio::test]
    async fn test_delta_schema() {
        // Just verify the schema is valid
        let delta_schema = StructType::new(vec![
            StructField::new(
                "subject",
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
            StructField::new(
                "predicate",
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
            StructField::new(
                "object",
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
        ]);
        assert_eq!(delta_schema.fields().count(), 3);
    }
}
