# カメラ座標別角度比較スクリプト実行用PowerShellスクリプト（詳細版）

param(
    [string]$MpCsvPattern = "CapturedFrames_*.csv",
    [string]$GtCsv = "synced_joint_positions.csv",
    [string]$OutputCsv = "coordinate_angle_mae.csv",
    [string[]]$Joints = $null
)

# 現在のディレクトリを取得
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($scriptDir) {
    Set-Location $scriptDir
}

Write-Host "=== カメラ座標別角度比較スクリプト ===" -ForegroundColor Cyan
Write-Host "作業ディレクトリ: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# ファイルの存在確認
if (-not (Test-Path "coordinate_angle_comparison.py")) {
    Write-Host "エラー: coordinate_angle_comparison.py が見つかりません" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $GtCsv)) {
    Write-Host "エラー: Ground Truth CSVファイルが見つかりません: $GtCsv" -ForegroundColor Red
    exit 1
}

# MediaPipe CSVファイルの確認
$mpFiles = Get-ChildItem -Filter $MpCsvPattern -ErrorAction SilentlyContinue
if ($mpFiles.Count -eq 0) {
    Write-Host "警告: MediaPipe CSVファイルが見つかりません: $MpCsvPattern" -ForegroundColor Yellow
    exit 1
}

Write-Host "MediaPipe CSVファイル数: $($mpFiles.Count)" -ForegroundColor Green
Write-Host "Ground Truth CSV: $GtCsv" -ForegroundColor Green
Write-Host "出力ファイル: $OutputCsv" -ForegroundColor Green
if ($Joints) {
    Write-Host "対象関節: $($Joints -join ', ')" -ForegroundColor Green
}
Write-Host ""

# コマンド構築
$pythonCmd = "python coordinate_angle_comparison.py --mp_csv `"$MpCsvPattern`" --gt_csv `"$GtCsv`" --output_csv `"$OutputCsv`""

if ($Joints) {
    $jointsStr = $Joints -join ' '
    $pythonCmd += " --joints $jointsStr"
}

Write-Host "実行コマンド:" -ForegroundColor Cyan
Write-Host $pythonCmd -ForegroundColor White
Write-Host ""

# 実行
try {
    Invoke-Expression $pythonCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=== 処理完了 ===" -ForegroundColor Green
        if (Test-Path $OutputCsv) {
            Write-Host "結果ファイル: $OutputCsv" -ForegroundColor Green
        }
    } else {
        Write-Host ""
        Write-Host "エラー: スクリプトの実行に失敗しました（終了コード: $LASTEXITCODE）" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host ""
    Write-Host "エラー: スクリプトの実行中にエラーが発生しました" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

