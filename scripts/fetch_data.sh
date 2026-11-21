#!/usr/bin/env bash
# Download the raw data and tokenizer files used by the experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data_raw vendor/tokenizers
[ -d data_raw/ctq ] || git clone --depth 1 https://github.com/AI4Bharat/CTQScorer.git data_raw/ctq
if [ ! -d data_raw/dcm ]; then
  git clone --depth 1 https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset.git data_raw/dcm
  (cd data_raw/dcm && mkdir -p x && unzip -q -o DravidianCodeMix-2020.zip -d x)
fi
tmp=$(mktemp -d)
for k in gpt4 gpt4o llama3 gemma3 qwen3 deepseek_v4 mistral_nemo; do
  [ -f "vendor/tokenizers/$k.json" ] && continue
  (cd "$tmp" && npm pack "@lenml/tokenizer-$k" >/dev/null && tar xzf lenml-tokenizer-$k-*.tgz)
  cp "$tmp/package/models/tokenizer.json" "vendor/tokenizers/$k.json"
  rm -rf "$tmp/package" "$tmp"/*.tgz
done
echo "done"
