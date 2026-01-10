# カメラ座標別角度比較スクリプト実行用PowerShellコマンド
# 使用方法: PowerShellでこのスクリプトを実行するか、コマンドを直接実行

# 現在のディレクトリに移動（必要に応じて変更）
$currentDir = "C:\projects\MOTIONTRACK\Zeval_DataSet\3_Cal_MAE\Y=0.5,1.5"
Set-Location $currentDir

# 基本コマンド
python coordinate_angle_comparison.py `
    --mp_csv "CapturedFrames_*.csv" `
    --gt_csv "synced_joint_positions.csv" `
    --output_csv "coordinate_angle_mae.csv"

# オプション: 特定の関節のみを比較する場合
# python coordinate_angle_comparison.py `
#     --mp_csv "CapturedFrames_*.csv" `
#     --gt_csv "synced_joint_positions.csv" `
#     --output_csv "coordinate_angle_mae.csv" `
#     --joints L_Elbow R_Elbow L_Knee R_Knee

