# MediaPipe姿勢推定 精度評価システム

GroundTruth（Unity Kinematika）とMediaPipe（RGB画像ベース）の3D関節位置データを比較・可視化するインタラクティブダッシュボードの実行手順と技術仕様を説明します。

> **📚 論文執筆者の方へ**: 研究知見や技術詳細は [`docs/paper/`](docs/paper/) フォルダを参照してください。  
> **📖 ドキュメント一覧**: [`docs/README.md`](docs/README.md) で全ドキュメントの構成を確認できます。

---

## 📋 目次

1. [研究背景と目的](#-研究背景と目的)
2. [前提条件](#-前提条件)
3. [実行手順](#-実行手順)
4. [ダッシュボードの機能詳細](#-ダッシュボードの機能詳細)
5. [データセットの詳細](#-データセットの詳細)
6. [技術詳細](#-技術詳細)
7. [出力ファイルの説明](#-出力ファイルの説明)
8. [データ処理パイプライン](#-データ処理パイプライン)
9. [トラブルシューティング](#-トラブルシューティング)
10. [よくある質問](#-よくある質問)
11. [📚 ドキュメント構成](#-ドキュメント構成)

---

## 🎯 研究背景と目的

### 研究目的
本研究は、単眼RGB画像から3D人体姿勢を推定する技術（MediaPipe）の精度を、高精度な3DモーションキャプチャシステムであるGroundTruth（Unity Kinematika）と比較・評価することを目的としています。

### 評価指標
従来の3Dユークリッド距離だけでなく、**角度差（Δθ, Δψ）**を主要指標として採用しています。これにより、関節の方向性の誤差をより直感的に評価できます。

### データセット特性
- **フレーム数**: 100フレーム（モーションキャプチャシーケンス）
- **カメラ配置**: XZ平面上に複数の視点（Y座標: 0.5m, 1.0m, 1.5m, 2.0m）
- **対象関節**: 12関節（肩・肘・手首・腰・膝・足首の左右）
- **座標系**: 左手系、Y-up、腰中心の相対座標

---

## 📋 前提条件

### 必須ソフトウェア
以下のソフトウェアがインストールされている必要があります：

| ソフトウェア | バージョン | 確認コマンド |
|------------|----------|------------|
| **Python** | 3.8以上（推奨: 3.9/3.10） | `python --version` |
| **Git** | 任意 | `git --version` |
| **pip** | 最新版推奨 | `pip --version` |

### 推奨環境
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **RAM**: 4GB以上（大規模データ処理時は8GB以上推奨）
- **ブラウザ**: Chrome, Firefox, Edge（最新版）
- **ディスク容量**: 500MB以上の空き容量

---

## 🚀 実行手順

### ステップ1: リポジトリのクローン

```bash
git clone https://github.com/MASHMARO2718/Zeval_DataSet.git
cd Zeval_DataSet
```

**確認**: `ls` または `dir` コマンドで以下のディレクトリが存在することを確認
- `7_direction_ditection/`
- `2_medidapipe_proccesed/`
- `README.md`

---

### ステップ2: プロジェクトディレクトリへ移動

```bash
cd 7_direction_ditection
```

**ディレクトリ構造の確認**:
```
7_direction_ditection/
├── interactive_dashboard.py     # 🚀 メイン: ダッシュボード起動
├── config.py                    # 設定ファイル
├── requirements.txt             # Python依存パッケージ
│
├── scripts/                     # データ処理・ユーティリティ
│   ├── process_all_data.py     # 全データ一括処理
│   ├── compute_correlation.py  # 相関分析
│   ├── data_loader.py          # データ読み込み
│   ├── coordinate_transform.py # 座標変換
│   └── ...
│
├── tests/                       # テストスクリプト
│   ├── run_all_tests.py        # テスト一括実行
│   ├── test_01_load_data.py
│   └── ...
│
├── docs/                        # ドキュメント
│   ├── paper/                  # 論文執筆用
│   └── archive/                # 古いドキュメント
│
├── data/                        # 入力データ
│   ├── 1_GroundTruth/
│   └── 2_medidapipe_proccesed/
│
└── output/                      # 処理結果
    ├── processed_data/         # CSV結果
    └── correlation_analysis/   # 相関分析結果
```

---

### ステップ3: 仮想環境のセットアップ

仮想環境を使用することで、システム全体のPython環境に影響を与えずにパッケージを管理できます。

#### Windows (PowerShell)の場合:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**PowerShellの実行ポリシーエラーが出た場合**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Windows (コマンドプロンプト)の場合:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### macOS/Linuxの場合:
```bash
python -m venv venv
source venv/bin/activate
```

**仮想環境が有効化されたことの確認**:
- プロンプトの先頭に `(venv)` が表示される
- `which python` (Windows: `where python`) で仮想環境のPythonが使われていることを確認

---

### ステップ4: 依存パッケージのインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**インストールされる主要パッケージ**:
- `pandas`: データ処理・分析
- `numpy`: 数値計算
- `plotly`: インタラクティブグラフ生成
- `dash`: Webダッシュボードフレームワーク
- `seaborn`, `matplotlib`: データ可視化
- `tqdm`: プログレスバー表示

**インストール確認**:
```bash
pip list | grep -E "(pandas|plotly|dash)"  # macOS/Linux
pip list | findstr "pandas plotly dash"    # Windows
```

---

### ステップ5: ダッシュボードの起動

```bash
python interactive_dashboard.py
```

**起動時の処理**:
1. CSVデータの読み込み（約5-10秒）
2. 統計データの前処理
3. Dashサーバーの起動

**成功時の出力例**:
```
[INFO] Loading processed data...
[INFO] Loaded detailed results: 12345 rows
[INFO] Loaded summary results: 678 rows
[INFO] Loaded joint-level stats: 90 rows
[INFO] Loading correlation matrices...
[OK] Dashboard is ready!
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'interactive_dashboard'
 * Debug mode: on
```

---

### ステップ6: ブラウザでアクセス

1. Webブラウザを開く
2. アドレスバーに以下を入力:
   ```
   http://127.0.0.1:8050/
   ```
3. ダッシュボードが表示される

**初回アクセス時**:
- グラフの初期描画に数秒かかる場合があります
- デフォルトでフレーム20、Y座標0.5が選択されています

---

## 📊 ダッシュボードの機能詳細

### 1. 📍 2Dカメラマップ（Camera Position Map）

**機能**:
- XZ平面上のカメラ配置を俯瞰表示
- データの有無を色で識別
- クリック操作でカメラを選択

**表示要素**:
| 要素 | 色 | 意味 |
|------|---|------|
| **カメラ（データあり）** | 🟢 緑 | このカメラ位置でMediaPipeデータが利用可能 |
| **カメラ（データなし）** | ⚪ 薄灰色 | データが存在しない（処理失敗等） |
| **選択中のカメラ** | ⭐ 黄色（星型） | 現在選択中のカメラ位置 |
| **ボット位置** | 🔴 赤 | GroundTruthの腰（Hip）の位置 |

**操作方法**:
1. **フレーム選択**: 左上のドロップダウンでフレーム番号を選択
2. **Y座標選択**: カメラの高さ（0.5m, 1.0m, 1.5m, 2.0m）を選択
3. **カメラ選択**: マップ上の緑の点をクリック

**ホバー情報**:
- カメラ座標 (X, Y, Z)
- データの有無
- フレーム番号

---

### 2. 🎯 3D骨格表示（3D Skeleton Visualization）

**機能**:
- GroundTruthとMediaPipeの3D関節位置を並べて表示
- リアルタイムに回転・ズーム・パン操作が可能
- 関節を線で結んだスティックフィギュア形式

**表示内容**:

#### 左側: GroundTruth (Unity Kinematika)
- **青色の関節**: 高精度なモーションキャプチャデータ
- **灰色の骨格線**: 関節を結ぶ接続線
- **座標軸**: X（赤）、Y（緑）、Z（青）

#### 右側: MediaPipe (RGB画像推定)
- **赤色の関節**: MediaPipeによる推定位置
- **灰色の骨格線**: 推定された骨格構造
- **座標軸**: 同じ座標系で表示

**マウス操作**:
| 操作 | 機能 |
|------|------|
| **ドラッグ** | 3D空間を回転 |
| **スクロール** | ズームイン・ズームアウト |
| **Shift+ドラッグ** | パン（平行移動） |
| **ダブルクリック** | ビューをリセット |

**骨格接続**:
```
肩 ━━ 肘 ━━ 手首
│
腰
│
膝 ━━ 足首
```

**座標範囲**:
- 自動で最適な表示範囲を計算
- グリッド線: 0.2m間隔
- アスペクト比: 立方体（等方的）

---

### 3. 📈 誤差分析グラフ

#### 3.1 フレーム・カメラ別角度誤差（Frame-Camera Error）

**表示内容**:
- X軸: フレーム番号（0-99）
- Y軸: 平均角度誤差（度）
- 青線: Δθ（XY平面角度誤差）
- 赤線: Δψ（XZ平面角度誤差）

**読み取り方**:
- 高いピーク: その視点・フレームで推定精度が低い
- 低い値: MediaPipeの推定が正確
- 経時変化: モーション全体での誤差の推移

#### 3.2 関節別角度誤差（Joint-Level Error）

**表示形式**: 横棒グラフ
- 各関節の平均角度誤差を降順に表示
- エラーバー: 標準偏差を表示
- 色分け: θ（青）、ψ（赤）

**解釈例**:
```
LEFT_SHOULDER:  15.2° ± 5.3°  ← 誤差が大きい（推定困難）
RIGHT_ELBOW:     8.7° ± 3.1°
LEFT_WRIST:      6.4° ± 2.8°  ← 誤差が小さい（推定精度高）
```

#### 3.3 時系列グラフ（Time Series）

**機能**:
- 選択した関節の誤差の時間変化を表示
- 全フレームにわたる推定精度の変動を確認

---

### 4. 📊 相関分析（Correlation Analysis）

**機能**:
- 12関節間の誤差相関を可視化
- 共起する誤差パターンを発見

**ヒートマップの読み方**:
| 相関係数 | 色 | 意味 |
|---------|---|------|
| **+1.0** | 🔴 濃い赤 | 完全な正の相関（一方の誤差が大きいと他方も大きい） |
| **0.0** | ⚪ 白 | 相関なし |
| **-1.0** | 🔵 濃い青 | 負の相関（通常は稀） |

**3つのタブ**:
1. **θ（XY平面）**: 水平方向の角度誤差の相関
2. **ψ（XZ平面）**: 垂直方向の角度誤差の相関
3. **3D距離**: ユークリッド距離誤差の相関

**高相関ペアの表**:
- 相関係数が0.7以上のペアを自動抽出
- 例: `(LEFT_SHOULDER, RIGHT_SHOULDER)` → 両肩は連動して誤差が発生

---

## 📁 データセットの詳細

### データソース

#### GroundTruth（`data/1_GroundTruth/`）
- **生成ツール**: Unity Kinematika
- **形式**: CSV（1ファイル、全フレーム含む）
- **精度**: サブミリメートル級
- **フレームレート**: 30fps（100フレーム = 約3.3秒）
- **座標系**: 左手系、Y-up、ワールド座標

**CSVフォーマット**:
```csv
frame,Hips_X,Hips_Y,Hips_Z,LeftShoulder_X,LeftShoulder_Y,...
0,-0.123,0.951,-0.456,0.234,1.234,...
1,-0.125,0.950,-0.458,0.236,1.232,...
```

#### MediaPipe（`data/2_medidapipe_proccesed/`）
- **生成ツール**: Google MediaPipe Pose
- **入力**: RGB画像（各カメラ・各フレーム）
- **形式**: CSV（カメラ位置ごとに1ファイル）
- **座標系**: 正規化座標（0-1範囲）→ 左手系に変換済み

**ディレクトリ構造**:
```
2_medidapipe_proccesed/
├── Y=0.5,1.5/
│   ├── CapturedFrames_-1.0_0.5_-3.0.csv
│   ├── CapturedFrames_-1.0_0.5_-4.0.csv
│   └── ...
└── Y=1.0.2.0/
    ├── CapturedFrames_-1.0_1.0_-3.0.csv
    └── ...
```

**ファイル命名規則**:
`CapturedFrames_{X}_{Y}_{Z}.csv`
- 例: `CapturedFrames_-1.0_0.5_-3.0.csv` → カメラ位置 (X=-1.0, Y=0.5, Z=-3.0)

### カメラ配置

**XZ平面のグリッド配置**:
```
Z軸（奥行き）
  ^
  |
  6m   ●  ●  ●  ●  ●  ●  ●
  5m   ●  ●  ●  ●  ●  ●  ●
  4m   ●  ●  ●  ●  ●  ●  ●
  3m   ●  ●  ●  ●  ●  ●  ●
  0m   ●  ●  🤖 ●  ●  ●  ●  (ボット位置)
 -3m   ●  ●  ●  ●  ●  ●  ●
 -4m   ●  ●  ●  ●  ●  ●  ●
 -5m   ●  ●  ●  ●  ●  ●  ●
 -6m   ●  ●  ●  ●  ●  ●  ●
  +---|---|---|---|---|---|---> X軸
    -6 -4 -2  0  2  4  6 (m)
```

**Y座標（高さ）**:
- 0.5m: 腰の高さ
- 1.0m: 胸の高さ
- 1.5m: 肩・頭の高さ
- 2.0m: 俯瞰視点

### 対象関節

| MediaPipe名 | GroundTruth名 | 身体部位 |
|------------|--------------|---------|
| LEFT_SHOULDER | LeftShoulder | 左肩 |
| RIGHT_SHOULDER | RightShoulder | 右肩 |
| LEFT_ELBOW | LeftLowerArm | 左肘 |
| RIGHT_ELBOW | RightLowerArm | 右肘 |
| LEFT_WRIST | LeftHand | 左手首 |
| RIGHT_WRIST | RightHand | 右手首 |
| LEFT_HIP | Hips（左側） | 左腰 |
| RIGHT_HIP | Hips（右側） | 右腰 |
| LEFT_KNEE | LeftLowerLeg | 左膝 |
| RIGHT_KNEE | RightLowerLeg | 右膝 |
| LEFT_ANKLE | LeftFoot | 左足首 |
| RIGHT_ANKLE | RightFoot | 右足首 |

---

## 📖 技術詳細

### 座標系の変換

#### 1. 右手系 → 左手系変換（不要と判明）
初期実装では以下を想定していましたが、実際のGroundTruthデータは既に左手系でした：
```python
# 実際には不要だった変換
x_lh = x_gt
y_lh = -y_gt  # Y軸反転は不要
z_lh = z_gt
```

#### 2. 絶対座標 → 相対座標変換
両データセットを腰（Hip）中心の相対座標に変換：

```python
# GroundTruth
hip_center_x = (left_hip_x + right_hip_x) / 2
hip_center_y = (left_hip_y + right_hip_y) / 2
hip_center_z = (left_hip_z + right_hip_z) / 2

joint_x_rel = joint_x - hip_center_x
joint_y_rel = joint_y - hip_center_y
joint_z_rel = joint_z - hip_center_z

# MediaPipeも同様に変換
```

**変換の理由**:
- カメラ位置の影響を除去
- ボットの移動の影響を除去
- 関節間の相対的な位置関係のみを評価

### 誤差指標の計算

#### 1. 角度誤差（Δθ）- XY平面
```python
# GroundTruthの角度
theta_gt = np.arctan2(y_rel_gt, x_rel_gt)

# MediaPipeの角度
theta_mp = np.arctan2(y_rel_mp, x_rel_mp)

# 角度差（-π to π に正規化）
delta_theta = np.arctan2(
    np.sin(theta_gt - theta_mp),
    np.cos(theta_gt - theta_mp)
)

# 度数法に変換
delta_theta_deg = np.degrees(delta_theta)
```

**意味**: 真上から見た時の方位のずれ

#### 2. 角度誤差（Δψ）- XZ平面
```python
psi_gt = np.arctan2(z_rel_gt, x_rel_gt)
psi_mp = np.arctan2(z_rel_mp, x_rel_mp)

delta_psi = np.arctan2(
    np.sin(psi_gt - psi_mp),
    np.cos(psi_gt - psi_mp)
)

delta_psi_deg = np.degrees(delta_psi)
```

**意味**: 真横（側面）から見た時の角度のずれ

#### 3. 3Dユークリッド距離
```python
error_3d = np.sqrt(
    (x_rel_gt - x_rel_mp)**2 +
    (y_rel_gt - y_rel_mp)**2 +
    (z_rel_gt - z_rel_mp)**2
)
```

**意味**: 3D空間での直線距離の誤差（メートル単位）

### 統計量の計算

各関節・カメラ・フレームの組み合わせで以下を計算：

```python
stats = {
    'mean': np.mean(errors),           # 平均誤差
    'median': np.median(errors),       # 中央値
    'std': np.std(errors),             # 標準偏差
    'min': np.min(errors),             # 最小誤差
    'max': np.max(errors),             # 最大誤差
    'q25': np.percentile(errors, 25),  # 第1四分位
    'q75': np.percentile(errors, 75)   # 第3四分位
}
```

### 相関分析

**Pearson相関係数**を使用：

```python
import numpy as np

# 各関節の誤差を列ベクトルとした行列
error_matrix = np.array([
    [joint1_error_frame0, joint1_error_frame1, ...],
    [joint2_error_frame0, joint2_error_frame1, ...],
    ...
])

# 相関行列の計算
correlation_matrix = np.corrcoef(error_matrix)
```

**解釈**:
- `corr(joint_i, joint_j) > 0.7`: 強い正の相関
- `corr(joint_i, joint_j) < 0.3`: 弱い相関（独立）

---

## 📂 出力ファイルの説明

### 1. 詳細結果（`output/detailed_results.csv`）

**各行**: 1フレーム×1カメラ×1関節のデータ

**列の説明**:
| 列名 | 説明 | 単位 |
|------|------|------|
| `frame_id` | フレーム番号 | 0-99 |
| `camera_position` | カメラ位置 | 例: `CapturedFrames_-1.0_0.5_-3.0` |
| `joint_name` | 関節名 | 例: `LEFT_SHOULDER` |
| `gt_x`, `gt_y`, `gt_z` | GroundTruth相対座標 | メートル |
| `mp_x`, `mp_y`, `mp_z` | MediaPipe相対座標 | メートル |
| `error_x`, `error_y`, `error_z` | 各軸の誤差 | メートル |
| `error_3d_norm` | 3D距離誤差 | メートル |
| `delta_theta` | XY平面角度誤差 | ラジアン |
| `delta_psi` | XZ平面角度誤差 | ラジアン |
| `delta_theta_deg` | XY平面角度誤差 | 度 |
| `delta_psi_deg` | XZ平面角度誤差 | 度 |

**行数**: 約80,000-120,000行（フレーム数×カメラ数×関節数）

### 2. サマリ結果（`output/summary_results.csv`）

**各行**: 1フレーム×1カメラの集約統計

**列の説明**:
| 列名 | 説明 |
|------|------|
| `frame_id` | フレーム番号 |
| `camera_position` | カメラ位置 |
| `mean_error_3d` | 全関節の平均3D誤差 |
| `mean_delta_theta` | 全関節の平均θ誤差 |
| `mean_delta_psi` | 全関節の平均ψ誤差 |
| `max_error_3d` | 最大3D誤差 |
| `max_delta_theta` | 最大θ誤差 |
| `max_delta_psi` | 最大ψ誤差 |
| `std_error_3d` | 3D誤差の標準偏差 |

### 3. 関節別統計（`output/joint_level_summary.csv`）

**各行**: 1フレーム×1カメラ×1関節の統計

### 4. 相関分析（`output/correlation_analysis/`）

**ファイル一覧**:
- `correlation_matrix_theta.csv`: θ誤差の相関行列
- `correlation_matrix_psi.csv`: ψ誤差の相関行列
- `correlation_matrix_3d_norm.csv`: 3D誤差の相関行列
- `high_correlation_pairs_theta.csv`: 高相関ペア（θ）
- `high_correlation_pairs_psi.csv`: 高相関ペア（ψ）
- `high_correlation_pairs_3d_norm.csv`: 高相関ペア（3D）
- `heatmap_*.png`: ヒートマップ画像（Gitでは除外）

---

## 🔄 データ処理パイプライン

### 全体フロー

```
[1] データ読み込み
     ↓
[2] 座標変換（絶対→相対）
     ↓
[3] 誤差計算（角度・距離）
     ↓
[4] 統計量算出
     ↓
[5] 相関分析
     ↓
[6] CSV出力
     ↓
[7] ダッシュボード表示
```

### 各スクリプトの役割

#### `scripts/data_loader.py`
- GroundTruthとMediaPipeのCSVを読み込み
- 関節名のマッピング
- カメラ位置のパース
- Y範囲に応じたディレクトリ選択

#### `scripts/coordinate_transform.py`
- 左手系への変換（実際には不要と判明）
- 腰中心の相対座標への変換
- 角度差の計算（arctan2ベース）
- 3D距離の計算

#### `scripts/process_all_data.py`
- 全フレーム・全カメラを一括処理
- プログレスバー表示（tqdm）
- 詳細結果、サマリ、関節別統計をCSV出力
- 処理時間: 約3-5分（データ量による）

#### `scripts/compute_correlation.py`
- 詳細結果CSVから相関行列を計算
- ヒートマップ生成（seaborn）
- 高相関ペア（>0.7）を抽出・保存

#### `interactive_dashboard.py`
- Plotly Dashアプリの起動
- データの動的更新
- グラフのインタラクティブ表示

### 手動でデータ再処理する場合

```bash
# すべてのフレーム・カメラのデータを再処理
python scripts/process_all_data.py

# 相関分析を再実行
python scripts/compute_correlation.py

# ダッシュボードを起動
python interactive_dashboard.py
```

---

## ❓ トラブルシューティング

### 1. ポート8050が既に使用されている

**症状**:
```
OSError: [Errno 48] Address already in use
```

**解決方法1**: ポート番号を変更
```python
# interactive_dashboard.py の最終行を編集
app.run(debug=True, port=8051)  # 8050 → 8051
```

**解決方法2**: 既存プロセスを停止
```bash
# Windows
netstat -ano | findstr :8050
taskkill /PID <プロセスID> /F

# macOS/Linux
lsof -ti:8050 | xargs kill -9
```

---

### 2. データが表示されない

**チェックリスト**:

1. **データファイルの存在確認**:
```bash
ls data/1_GroundTruth/*.csv
ls data/2_medidapipe_proccesed/Y=0.5,1.5/*.csv
ls data/2_medidapipe_proccesed/Y=1.0.2.0/*.csv
```

2. **処理済みファイルの存在確認**:
```bash
ls output/detailed_results.csv
ls output/summary_results.csv
ls output/joint_level_summary.csv
```

3. **データの再処理**:
```bash
python scripts/process_all_data.py
```

---

### 3. 依存パッケージのインストールエラー

**症状**:
```
ERROR: Could not find a version that satisfies the requirement...
```

**解決方法**:
```bash
# pipのアップグレード
python -m pip install --upgrade pip

# 個別にインストール
pip install pandas numpy plotly dash

# キャッシュをクリアして再インストール
pip cache purge
pip install -r requirements.txt
```

---

### 4. UnicodeEncodeError (Windows)

**症状**:
```
UnicodeEncodeError: 'cp932' codec can't encode character
```

**解決方法**:
環境変数を設定:
```powershell
$env:PYTHONIOENCODING = "utf-8"
python interactive_dashboard.py
```

または`config.py`で設定:
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

---

### 5. メモリ不足エラー

**症状**:
```
MemoryError: Unable to allocate array
```

**解決方法**:
1. 他のアプリケーションを終了してメモリを確保
2. 処理するフレーム数を制限（`config.py`を編集）
3. より大きなRAMを搭載したマシンを使用

---

### 6. グラフが表示されない

**症状**: ダッシュボードは起動するがグラフが空白

**解決方法**:
1. ブラウザのキャッシュをクリア（Ctrl+Shift+Del）
2. 別のブラウザで試す（Chrome推奨）
3. ブラウザのコンソール（F12）でエラーを確認
4. `debug=True`でDashを起動し、ターミナルのエラーログを確認

---

### 7. PowerShell実行ポリシーエラー

**症状**:
```
.\venv\Scripts\Activate.ps1 : このシステムではスクリプトの実行が無効になっているため...
```

**解決方法**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 💡 よくある質問

### Q1: 処理時間はどのくらいかかりますか？

**A**: 
- データ読み込み: 約10秒
- 全データ処理（`scripts/process_all_data.py`）: 約3-5分
- 相関分析（`scripts/compute_correlation.py`）: 約30秒
- ダッシュボード起動: 約5秒

合計で初回セットアップは約10分程度です。2回目以降は処理済みデータを読み込むだけなので数秒で起動します。

---

### Q2: カメラの位置はどのように選択されましたか？

**A**: 
XZ平面上に1メートル間隔でグリッド配置し、ボット（原点）を中心とした多視点データを収集しました。Y座標（高さ）は人体の各部位に対応する4段階（0.5m, 1.0m, 1.5m, 2.0m）を設定しています。

---

### Q3: なぜ角度誤差を主要指標にしたのですか？

**A**: 
3Dユークリッド距離だけでは、関節の「方向性」の誤差を適切に評価できません。例えば、肘が10cm外側にずれた場合と手前にずれた場合では、距離は同じでも意味が異なります。角度誤差を使うことで、関節の方向性の違いを直感的に評価できます。

---

### Q4: データセットを拡張できますか？

**A**: 
はい、可能です。以下の手順で新しいデータを追加できます：

1. GroundTruthのCSVを`data/1_GroundTruth/`に追加
2. MediaPipeのCSVを`data/2_medidapipe_proccesed/`の適切なY範囲ディレクトリに追加
3. `python scripts/process_all_data.py`で再処理
4. `python interactive_dashboard.py`でダッシュボードを起動

---

### Q5: 商用利用は可能ですか？

**A**: 
研究・教育目的での利用を想定しています。商用利用については、開発者にお問い合わせください。

---

### Q6: ダッシュボードをオンラインで公開できますか？

**A**: 
可能です。以下の方法があります：

1. **Herokuへのデプロイ**: 無料プランあり
2. **Render.comへのデプロイ**: 無料プランあり
3. **Google Cloud Run**: 従量課金
4. **自前サーバー**: Nginxなどでリバースプロキシ設定

詳細なデプロイ手順は別途ドキュメントを参照してください。

---

### Q7: エラーログはどこに保存されますか？

**A**: 
`output/logs/`ディレクトリに日付付きログファイルが保存されます：
```
output/logs/process_20260110_143522.log
```

---

### Q8: データをエクスポートできますか？

**A**: 
はい、すべての処理結果はCSV形式で`output/`に保存されています。Excel、Python、R等で直接読み込んで分析できます。

---

## 📧 サポート

### 問題が解決しない場合

1. **ログファイルの確認**: `output/logs/` のエラーログを確認
2. **Issueの作成**: GitHubリポジトリでIssueを作成
3. **開発者への連絡**: メールで詳細な状況を共有

### 報告時に含めるべき情報

- OS名とバージョン
- Pythonバージョン（`python --version`）
- エラーメッセージの全文
- 実行したコマンド
- ログファイルの内容

---

## 📚 参考資料

### 使用技術
- [Plotly Dash公式ドキュメント](https://dash.plotly.com/)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Unity Kinematika](https://docs.unity3d.com/Packages/com.unity.kinematica@0.8/manual/index.html)

---

## 📚 ドキュメント構成

プロジェクトのドキュメントは目的別に整理されています：

### 📁 実行ガイド
- **README.md**（このファイル）: ダッシュボード実行手順

### 📁 論文執筆用（`docs/paper/`）
- **TECHNICAL_REPORT_FOR_PAPER.md**: 技術詳細レポート（Methods/Results用）
- **MEDIAPIPE_ERROR_ANALYSIS.md**: MediaPipe誤差の根本原因分析（Discussion用）
- **FINDINGS_AND_INSIGHTS.md**: 研究知見と洞察の総まとめ

### 📁 アーカイブ（`docs/archive/`）
開発過程で作成された古いドキュメント（参考用）

**詳細**: [`docs/README.md`](docs/README.md) で全ドキュメントの構成と推奨閲覧順序を確認できます。

---

**プロジェクト作成日**: 2026年1月  
**最終更新**: 2026年1月18日  
**バージョン**: 2.0.0  
**ライセンス**: 研究・教育目的での利用を想定

---

**🎉 インタラクティブダッシュボードをお楽しみください！**
