# Downloads Home Credit data through the Kaggle API and keeps only the Parquet files.
# Requires: competition rules accepted, %USERPROFILE%\.kaggle\kaggle.json present.
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$hc   = Join-Path $repo "data\raw\homecredit"
$tmp  = Join-Path $env:TEMP "creditops_hc"
New-Item -ItemType Directory -Force $hc, $tmp | Out-Null

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) { python -m pip install --user kaggle }

python -m kaggle competitions download -c home-credit-credit-risk-model-stability -p $tmp
$zip = Get-ChildItem $tmp -Filter *.zip | Select-Object -First 1

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip.FullName)
try {
    foreach ($e in $archive.Entries) {
        if ($e.FullName -like "parquet_files/*" -or $e.FullName -eq "feature_definitions.csv") {
            $dest = Join-Path $hc $e.FullName
            if ($e.FullName.EndsWith("/")) { New-Item -ItemType Directory -Force $dest | Out-Null; continue }
            New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dest, $true)
        }
    }
} finally { $archive.Dispose() }

Remove-Item $tmp -Recurse -Force
Write-Host "Home Credit ready dans $hc"
