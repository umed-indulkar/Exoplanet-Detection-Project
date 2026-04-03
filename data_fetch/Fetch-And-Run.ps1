# D:\ppp\data_fetch\Fetch-Parallel.ps1
$BaseDir      = "D:\ppp\data_fetch"
$FetchedDir   = "D:\ppp\data\fetched"
$RawDir       = "D:\ppp\data\raw_tbl_files"
$LogFile      = "D:\ppp\data\dataset\processed_log.txt"
$LinksFile    = "D:\ppp\data\fetched\all_tbl_links.txt"
$PythonScript = "D:\ppp\data_fetch\process_dataset.py"

# Ensure directories exist
if (-not (Test-Path $RawDir)) { New-Item -ItemType Directory -Path $RawDir -Force | Out-Null }
if (-not (Test-Path (Split-Path $LogFile))) { New-Item -ItemType Directory -Path (Split-Path $LogFile) -Force | Out-Null }
if (-not (Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile -Force | Out-Null }

$Links = Get-Content $LinksFile
Write-Host "--- PARALLEL PIPELINE STARTING (Throttle: 5) ---" -ForegroundColor Cyan

$Links | ForEach-Object -Parallel {
    # Redefine variables for thread scope
    $RawDir       = "D:\ppp\data\raw_tbl_files"
    $LogFile      = "D:\ppp\data\dataset\processed_log.txt"
    $PythonScript = "D:\ppp\data_fetch\process_dataset.py"
    
    $FolderUrl = $_
    $KID = ($FolderUrl -split '/')[-2]
    
    # RESUME LOGIC: Check if KID exists in log
    if (Select-String -Path $LogFile -Pattern $KID -Quiet) { return }

    try {
        $Resp = Invoke-WebRequest -Uri $FolderUrl -TimeoutSec 15 -ErrorAction Stop
        
        # ALIGNED: Now looking for .fits files to match your Python astropy logic
        $FileName = $Resp.Links | Where-Object { $_.href -like "*lc.fits" } | Select-Object -First 1 -ExpandProperty href
        
        if ($FileName) {
            $LocalPath = Join-Path $RawDir "$KID-temp.fits"
            Invoke-WebRequest -Uri ($FolderUrl + $FileName) -OutFile $LocalPath
            
            # Run Python
            & python $PythonScript --file $LocalPath
            
            if ($LASTEXITCODE -eq 0) {
                $KID | Out-File -FilePath $LogFile -Append -Encoding utf8
                Write-Host "[SUCCESS] $KID" -ForegroundColor Green
            }
        } else {
            Write-Host "[NO FITS FILE] $KID" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[TIMEOUT/SKIP] $KID" -ForegroundColor Red
    }
} -ThrottleLimit 5

Write-Host "--- PROCESSING COMPLETE ---" -ForegroundColor Cyan
Write-Host "Run the merge command to combine the CSV files." -ForegroundColor Green