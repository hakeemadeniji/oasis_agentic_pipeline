#!/bin/bash
#
# Bootstrap the local Ollama edge LLM used by the Clinical Reasoner agent.
# All language reasoning runs on a local Ollama daemon -- no paid API keys.
#
# Usage: ./scripts/setup_ollama.sh [model]
#   model - Ollama model tag to pull (default: llama3.2:3b)
#
set -e

MODEL="${1:-llama3.2:3b}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "[*] Ollama not found. Installing ..."
    if [[ "$(uname -s)" == "Linux" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "[!] Please install Ollama from https://ollama.com/download and re-run."
        exit 1
    fi
fi

# Start the daemon if it is not already serving.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[*] Starting Ollama service ..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
fi

echo "[*] Pulling edge model: ${MODEL} (one-time download) ..."
ollama pull "${MODEL}"

echo ""
echo "[SUCCESS] Ollama ready. Edge model '${MODEL}' available at http://localhost:11434"
echo "          Set OLLAMA_EDGE_MODEL=${MODEL} in your .env to use it."
