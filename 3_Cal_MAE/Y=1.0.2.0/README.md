# カメラ座標別角度比較スクリプト

MediaPipeとGround Truthの関節角度を比較し、MAE（平均絶対誤差）を計算するスクリプトです。

## 概要

このスクリプトは、複数のカメラ座標（`CapturedFrames_X_Y_Z.csv`形式）で撮影されたMediaPipeデータと、単一のGround Truthデータ（`synced_joint_positions.csv`）を比較し、各関節の角度誤差を計算します。

## 前提条件

- Python 3.x
- 必要なライブラリ: pandas, numpy
- MediaPipe CSVファイル（`CapturedFrames_X_Y_Z.csv`形式）
- Ground Truth CSVファイル（`synced_joint_positions.csv`）

## ファイル構成

- `coordinate_angle_comparison.py` - メインスクリプト
- `run_coordinate_angle_comparison.ps1` - 簡単な実行スクリプト
- `run_analysis.ps1` - 詳細な実行スクリプト（エラーチェック付き）

## 使用方法

### 方法1: 直接Pythonコマンドを実行（推奨）

PowerShellで以下のコマンドを実行：

```powershell
# 基本実行（全関節を比較）
python coordinate_angle_comparison.py `
    --mp_csv "CapturedFrames_*.csv" `
    --gt_csv "synced_joint_positions.csv" `
    --output_csv "coordinate_angle_mae.csv"
```

**ワンライナー版：**

```powershell
python coordinate_angle_comparison.py --mp_csv "CapturedFrames_*.csv" --gt_csv "synced_joint_positions.csv" --output_csv "coordinate_angle_mae.csv"
```

### 方法2: 簡単なスクリプトを使用

```powershell
.\run_coordinate_angle_comparison.ps1
```

### 方法3: 詳細版スクリプトを使用（推奨）

エラーチェックや詳細な出力が含まれます。

```powershell
# 基本実行
.\run_analysis.ps1

# 出力ファイル名を指定
.\run_analysis.ps1 -OutputCsv "my_results.csv"

# 特定の関節のみを比較
.\run_analysis.ps1 -Joints @("L_Elbow", "R_Elbow", "L_Knee", "R_Knee")

# すべてのパラメータを指定
.\run_analysis.ps1 `
    -MpCsvPattern "CapturedFrames_*.csv" `
    -GtCsv "synced_joint_positions.csv" `
    -OutputCsv "results.csv" `
    -Joints @("L_Elbow", "R_Elbow")
```

## コマンドライン引数

### `coordinate_angle_comparison.py` の引数

| 引数 | 必須 | 説明 | デフォルト |
|------|------|------|-----------|
| `--mp_csv` | はい | MediaPipe CSVファイルのパス（ワイルドカード使用可） | - |
| `--gt_csv` | はい | Ground Truth CSVファイルのパス | - |
| `--output_csv` | いいえ | 出力CSVファイル名 | `coordinate_angle_mae.csv` |
| `--joints` | いいえ | 比較する関節（複数指定可） | 全関節 |

### 使用可能な関節

- `L_Elbow` - 左肘
- `R_Elbow` - 右肘
- `L_Knee` - 左膝
- `R_Knee` - 右膝
- `L_Shoulder` - 左肩
- `R_Shoulder` - 右肩
- `L_Hip` - 左股関節
- `R_Hip` - 右股関節

## 実行例

### 例1: 全関節を比較

```powershell
python coordinate_angle_comparison.py `
    --mp_csv "CapturedFrames_*.csv" `
    --gt_csv "synced_joint_positions.csv"
```

### 例2: ヒンジ関節のみを比較

```powershell
python coordinate_angle_comparison.py `
    --mp_csv "CapturedFrames_*.csv" `
    --gt_csv "synced_joint_positions.csv" `
    --output_csv "hinge_joints_mae.csv" `
    --joints L_Elbow R_Elbow L_Knee R_Knee
```

### 例3: 左側の関節のみを比較

```powershell
python coordinate_angle_comparison.py `
    --mp_csv "CapturedFrames_*.csv" `
    --gt_csv "synced_joint_positions.csv" `
    --joints L_Elbow L_Knee L_Shoulder L_Hip
```

## 出力ファイル

### 出力CSVの構造

`coordinate_angle_mae.csv` には以下の列が含まれます：

- `folder_name` - ファイル名（拡張子なし）
- `camera_x` - カメラX座標
- `camera_y` - カメラY座標
- `camera_z` - カメラZ座標
- `L_Elbow` - 左肘のMAE（度）
- `R_Elbow` - 右肘のMAE（度）
- `L_Knee` - 左膝のMAE（度）
- `R_Knee` - 右膝のMAE（度）
- ...（その他の関節）

### コンソール出力

スクリプト実行時、以下の統計情報が表示されます：

```
=== カメラ座標別角度MAE結果 ===
処理座標数: 120
L_Elbow: MAE=12.34° (範囲: 5.67°-25.89°)
R_Elbow: MAE=13.45° (範囲: 6.12°-26.34°)
...

カメラ座標範囲:
X: -6.0 ～ 6.0
Y: 0.5 ～ 1.5
Z: -6.0 ～ 6.0
```

## トラブルシューティング

### エラー: MediaPipe CSVが見つかりません

- `--mp_csv` で指定したパスが正しいか確認してください
- ワイルドカードが正しく機能しているか、直接ファイル名を指定してテストしてください

### エラー: Ground Truth CSVが見つかりません

- `synced_joint_positions.csv` が同じディレクトリにあるか確認してください
- フルパスを指定することもできます

### エラー: モジュールが見つかりません

必要なライブラリをインストール：

```powershell
pip install pandas numpy
```

## ログファイル

実行ログは `coordinate_angle_comparison.log` に保存されます。

## 注意事項

- Ground Truth CSV（`synced_joint_positions.csv`）は単一ファイルとして使用されます
- すべてのMediaPipe CSVファイルに対して同じGround Truthデータが使用されます
- フレームIDが一致するデータのみが比較されます

## ライセンス

このスクリプトは内部使用目的で作成されました。

