# Fetch-ExoData.ps1
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

# 1. Corrected TAP Query using the Short Name 'cumulative'
$Query = "SELECT kepid, kepoi_name, koi_disposition, koi_period, koi_time0bk FROM cumulative WHERE koi_period IS NOT NULL"
$EncodedQuery = [uri]::EscapeDataString($Query)
$TapApiUrl = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=$EncodedQuery&format=csv"
$MasterCsv = Join-Path $PSScriptRoot "master_koi_catalog.csv"

Write-Host "Downloading Master KOI Catalog..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $TapApiUrl -OutFile $MasterCsv -ErrorAction Stop
    Write-Host "Success: Catalog saved to $MasterCsv" -ForegroundColor Green
} catch {
    Write-Error "TAP Query failed. Try checking https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&titles=Cumulative+KOI&config=cumulative"
    return
}

# 2. Setup Raw Directory
$RawDir = New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "raw_tbl_files")

# 3. Generate Download URLs using the MAST Archive structure
Write-Host "Generating .tbl download links..." -ForegroundColor Cyan
$Catalog = Import-Csv $MasterCsv
$Urls = foreach ($row in $Catalog) {
    $KID = $row.kepid.PadLeft(9, '0')
    $Prefix = $KID.Substring(0, 4)
    # The directory where individual quarter .tbl files live
    "https://archive.stsci.edu/pub/kepler/lightcurves/$Prefix/$KID/"
}

$Urls | Out-File (Join-Path $PSScriptRoot "all_tbl_links.txt")
Write-Host "Ready! links saved to all_tbl_links.txt" -ForegroundColor Green