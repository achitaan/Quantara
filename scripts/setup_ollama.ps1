param([string]$Version = '0.35.0')
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw 'Use a numbered Ollama release.' }
$directory = Join-Path $workspace "runtime/ollama/$Version"
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$executable = Join-Path $directory 'ollama.exe'
if (Test-Path -LiteralPath $executable) {
    Write-Output $executable
    exit 0
}
$release = Invoke-RestMethod "https://api.github.com/repos/ollama/ollama/releases/tags/v$Version"
$asset = $release.assets | Where-Object name -eq 'ollama-windows-amd64.zip'
if (!$asset -or $asset.digest -notmatch '^sha256:[a-f0-9]{64}$') { throw 'Official release checksum unavailable.' }
$archive = Join-Path $directory 'download.zip'
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archive
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ('sha256:' + $digest -ne $asset.digest) { throw 'Ollama checksum mismatch. Download was not executed.' }
Expand-Archive -LiteralPath $archive -DestinationPath $directory -Force
Remove-Item -LiteralPath $archive
$metadata = @{ version=$Version; checksum=$asset.digest; source=$asset.browser_download_url }
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $directory 'release.json') -Encoding utf8
Write-Output $executable
