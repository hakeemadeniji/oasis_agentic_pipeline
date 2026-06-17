<#
.SYNOPSIS
    Bootstrap the local Ollama edge LLM used by the Clinical Reasoner agent.

.DESCRIPTION
    The pipeline performs all language reasoning on a local Ollama daemon — no
    paid API keys. This script installs Ollama (via winget if needed), starts
    the service, and pulls a small Snapdragon-friendly model.

.EXAMPLE
    ./scripts/setup_ollama.ps1
    ./scripts/setup_ollama.ps1 -Model phi3.5
#>
[CmdletBinding()]
param(
    [string]$Model = "llama3.2:3b"
)

$ErrorActionPreference = "Stop"

function Test-Ollama { return [bool](Get-Command ollama -ErrorAction SilentlyContinue) }

if (-not (Test-Ollama)) {
    Write-Host "[*] Ollama not found. Installing via winget ..."
    try {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    } catch {
        Write-Warning "winget install failed. Download manually from https://ollama.com/download then re-run."
        exit 1
    }
}

Write-Host "[*] Ensuring the Ollama service is running ..."
$running = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $running) {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "[*] Pulling edge model: $Model (this is a one-time download) ..."
ollama pull $Model

Write-Host ""
Write-Host "[SUCCESS] Ollama ready. Edge model '$Model' available at http://localhost:11434"
Write-Host "          Set OLLAMA_EDGE_MODEL=$Model in your .env to use it."
Write-Host "          Tip: 'phi3.5' or 'qwen2.5:3b' are also strong small edge models."
