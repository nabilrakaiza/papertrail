#!/usr/bin/env bash
# Downloads GloVe 6B and keeps only the 200d vectors used in Part II.
set -euo pipefail

DEST="embeddings"
URL="https://nlp.stanford.edu/data/glove.6B.zip"
mkdir -p "$DEST"

if [ -f "$DEST/glove.6B.200d.txt" ]; then
  echo "glove.6B.200d.txt already present, skipping."
  exit 0
fi

echo "Downloading GloVe 6B (~822 MB)..."
curl -L --fail -o "$DEST/glove.6B.zip" "$URL"
unzip -o -j "$DEST/glove.6B.zip" "glove.6B.200d.txt" -d "$DEST"
rm "$DEST/glove.6B.zip"
echo "Saved $DEST/glove.6B.200d.txt"
