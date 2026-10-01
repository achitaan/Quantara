param([string]$Version = '0.35.0')
$ErrorActionPreference = 'Stop'
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw 'Use a numbered Ollama release.' }
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$executable = Join-Path $workspace "runtime/ollama/$Version/ollama.exe"
if (!(Test-Path -LiteralPath $executable)) { throw 'Run scripts/setup_ollama.ps1 first.' }
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_MODELS = Join-Path $workspace 'runtime/ollama-models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:LLAMA_ARG_CACHE_RAM = '256'
$env:OLLAMA_KEEP_ALIVE = '10m'
& $executable serve
