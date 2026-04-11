#!/bin/bash
# Export 768 vectors from Mac databases to Dropbox sync folder.
# Run manually or via launchd to keep the iPhone checkpoint fresh.
#
# Usage: ./tools/write_checkpoint_to_dropbox.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_DIR="$HOME/Library/CloudStorage/Dropbox/__768_sync"
CHECKPOINT_DIR="$SYNC_DIR/checkpoints"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$SYNC_DIR/deltas/mac"
mkdir -p "$SYNC_DIR/deltas/iphone"
mkdir -p "$SYNC_DIR/ack"

echo "Exporting vectors to checkpoint..."
python3 "$SCRIPT_DIR/export_checkpoint.py" "$CHECKPOINT_DIR/latest.db"

SIZE=$(du -h "$CHECKPOINT_DIR/latest.db" | cut -f1)
echo "Checkpoint written: $CHECKPOINT_DIR/latest.db ($SIZE)"
echo "Dropbox will sync to cloud automatically."
