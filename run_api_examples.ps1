param(
    [ValidateSet("classify", "reconstruct", "process", "all")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"

# =========================
# Edit this block only
# =========================
$BaseUrl = "http://127.0.0.1:5000"

$IntrinsicsJson = "[[500,0,320],[0,500,240],[0,0,1]]"
$PosesJson = "[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0,0,0.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]"

# Keep counts aligned with poses (2 poses -> rgb_0/depth_0 and rgb_1/depth_1).
$Frames = @(
    @{
        Rgb = "D:\Downloads_D\Q-GIS\AI_measurements\approach_5_Photogrammetry\Photogrammetry-backend\sample_data\dummy_rgbd\rgb_0.png"
        Depth = "D:\Downloads_D\Q-GIS\AI_measurements\approach_5_Photogrammetry\Photogrammetry-backend\sample_data\dummy_rgbd\depth_0.png"
    },
    @{
        Rgb = "D:\Downloads_D\Q-GIS\AI_measurements\approach_5_Photogrammetry\Photogrammetry-backend\sample_data\dummy_rgbd\rgb_1.png"
        Depth = "D:\Downloads_D\Q-GIS\AI_measurements\approach_5_Photogrammetry\Photogrammetry-backend\sample_data\dummy_rgbd\depth_1.png"
    }
)

# Optional: set category for /process or /reconstruct.
# Leave empty to use automatic routing.
$Category = ""

function Assert-FileExists {
    param([string]$PathValue, [string]$Label)
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label file not found: $PathValue"
    }
}

function Invoke-CurlJson {
    param(
        [string]$Url,
        [string[]]$FormArgs
    )

    $args = @("-sS", "-X", "POST", $Url)
    $args += $FormArgs
    $raw = & curl.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Url"
    }
    return $raw
}

function Build-ClassifyArgs {
    $formArgs = @()
    for ($i = 0; $i -lt $Frames.Count; $i++) {
        Assert-FileExists -PathValue $Frames[$i].Rgb -Label "RGB_$i"
        $formArgs += @("-F", "rgb_$i=@$($Frames[$i].Rgb)")
    }
    return $formArgs
}

function Build-RgbdArgs {
    $formArgs = @(
        "-F", "intrinsics=$IntrinsicsJson",
        "-F", "poses=$PosesJson"
    )
    if (-not [string]::IsNullOrWhiteSpace($Category)) {
        $formArgs += @("-F", "category=$Category")
    }

    for ($i = 0; $i -lt $Frames.Count; $i++) {
        Assert-FileExists -PathValue $Frames[$i].Rgb -Label "RGB_$i"
        Assert-FileExists -PathValue $Frames[$i].Depth -Label "DEPTH_$i"
        $formArgs += @("-F", "rgb_$i=@$($Frames[$i].Rgb)")
        $formArgs += @("-F", "depth_$i=@$($Frames[$i].Depth)")
    }
    return $formArgs
}

function Run-Classify {
    Write-Host "`n=== POST /classify ===" -ForegroundColor Cyan
    $args = Build-ClassifyArgs
    $raw = Invoke-CurlJson -Url "$BaseUrl/classify" -FormArgs $args
    $raw | ConvertFrom-Json | ConvertTo-Json -Depth 20
}

function Run-Reconstruct {
    Write-Host "`n=== POST /reconstruct ===" -ForegroundColor Cyan
    $args = Build-RgbdArgs
    $raw = Invoke-CurlJson -Url "$BaseUrl/reconstruct" -FormArgs $args
    $raw | ConvertFrom-Json | ConvertTo-Json -Depth 20
}

function Run-Process {
    Write-Host "`n=== POST /process ===" -ForegroundColor Cyan
    $args = Build-RgbdArgs
    $raw = Invoke-CurlJson -Url "$BaseUrl/process" -FormArgs $args
    $raw | ConvertFrom-Json | ConvertTo-Json -Depth 20
}

switch ($Mode) {
    "classify" { Run-Classify }
    "reconstruct" { Run-Reconstruct }
    "process" { Run-Process }
    "all" {
        Run-Classify
        Run-Reconstruct
        Run-Process
    }
}
