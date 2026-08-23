#!/bin/bash
# upload-to-hf.sh — Upload TinyMetatron + Copilot v2 to Hugging Face Hub.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login  (requires HF_TOKEN with write access)
#
# The script is idempotent — safe to re-run.  OAuth tokens expire; re-run
# `huggingface-cli login` if you see 401 errors.

set -euo pipefail

REPO_ID="Quantum927/TinyMetatron"
REPO_TYPE="model"

echo "=== Step 1: Install / verify huggingface_hub ==="
.venv/bin/pip install -q huggingface_hub

echo "=== Step 2: Create repo (public, skip if exists) ==="
.venv/bin/python - <<'PYTHON'
from huggingface_hub import create_repo, get_token
token = get_token("write")
if not token:
    print("ERROR: Not logged in. Run  huggingface-cli login  first.")
    raise SystemExit(1)
create_repo(
    repo_id="Quantum927/TinyMetatron",
    repo_type="model",
    token=token,
    exist_ok=True,
    private=False,
)
print("Repo ready: https://huggingface.co/Quantum927/TinyMetatron")
PYTHON

echo ""
echo "=== Step 3: Upload model card as README.md ==="
huggingface-cli upload \
    --repo-id "$REPO_ID" \
    --repo-type model \
    --include "hf-model-card.md" \
    --commit-message "docs: add Hugging Face model card"

echo ""
echo "=== Step 4: Upload core model files ==="
huggingface-cli upload \
    --repo-id "$REPO_ID" \
    --repo-type model \
    --include "api.py" \
              "config.py" \
              "tokenizer.py" \
              "metatron_model.py" \
              "metatron_attention.py" \
              "metatron_moe.py" \
              "metatron_memory.py" \
              "train_db.py" \
              "manage_data.py" \
              "db.py" \
              "quality.py" \
              "workers/train.py" \
              "workers/__init__.py" \
    --commit-message "feat: add core TinyMetatron model files"

echo ""
echo "=== Step 5: Upload copilot orchestration layer ==="
huggingface-cli upload \
    --repo-id "$REPO_ID" \
    --repo-type model \
    --include "copilot/" \
    --commit-message "feat: add copilot v2 orchestration layer (17 agents, loops)"

echo ""
echo "=== Step 6: Upload quantum_corpus RAG backbone ==="
huggingface-cli upload \
    --repo-id "$REPO_ID" \
    --repo-type model \
    --include "quantum_corpus/" \
    --commit-message "feat: add quantum-corpus RAG backbone"

echo ""
echo "=== Step 7: Upload loops, workers, tests ==="
huggingface-cli upload \
    --repo-id "$REPO_ID" \
    --repo-type model \
    --include "loops/" \
              "tests/" \
    --commit-message "feat: add training loops and tests"

echo ""
echo "=== Step 8: Upload docs and configs ==="
huggingface-cli upload \
    --repo-id "$REPO_ID" \
    --repo-type model \
    --include "README.md" \
              "IMPLEMENTATION_CONTRACT.md" \
              "Dockerfile" \
              "Dockerfile-hf-space" \
              "TOKENIZER_PILOT.md" \
    --commit-message "docs: add README, contract, and HF Space Dockerfile"

echo ""
echo "========================================"
echo "Upload complete!"
echo ""
echo "View your model at:"
echo "  https://huggingface.co/$REPO_ID"
echo ""
echo "To update the Docker Space after Phase 3 (frontend), run:"
echo "  huggingface-cli repo-sync Quantum927/tinymetatron-slm --token \$HF_TOKEN"
echo ""
echo "Or update the Space Dockerfile to:"
echo "  sed -i 's|FROM python:3.13-slim|FROM python:3.13-slim\nCOPY . /app|' \\
        Dockerfile-hf-space"
echo "then push and let the Space rebuild."
