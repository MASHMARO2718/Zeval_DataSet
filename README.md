# Motion Tracking Data Analysis Project

このプロジェクトは、GroundTruth（Unity Kinematika）とMediaPipeの3D人体関節位置データを比較・分析するシステムです。

## 🎯 プロジェクト概要

- **GroundTruthデータ**: Unity Kinematikaで生成された正確な3D関節位置
- **MediaPipeデータ**: RGB画像から推定された3D関節位置
- **比較指標**: 角度誤差（Δθ、Δψ）および3D距離誤差
- **インタラクティブ可視化**: Plotly DashによるWebベースのダッシュボード

## 📂 ディレクトリ構成

```
Zeval_DataSet/
├── 1_Output_Photos/           # 元画像（大容量のため.gitignoreで除外）
├── 4_MAE_HEATMAP/             # 初期分析結果
├── 7_direction_ditection/     # ⭐ メインプロジェクト
│   ├── data/
│   │   ├── 1_GroundTruth/     # GroundTruthデータ（CSV）
│   │   └── 2_medidapipe_proccesed/  # MediaPipe処理済みデータ
│   ├── output/                # 分析結果（CSV、HTML、PNG）
│   ├── scripts/               # データ処理スクリプト
│   ├── tests/                 # テストスクリプト
│   ├── config.py              # 設定ファイル
│   ├── interactive_dashboard.py  # ⭐ ダッシュボード本体
│   ├── process_all_data.py    # データ一括処理
│   ├── compute_correlation.py # 相関分析
│   ├── requirements.txt       # Python依存パッケージ
│   └── README_FOR_PROFESSOR.md  # ⭐ 詳細な実行手順
└── README.md                  # このファイル
```

## 🚀 クイックスタート（教授向け）

### 1. リポジトリのクローン
```bash
git clone <リポジトリURL>
cd Zeval_DataSet/7_direction_ditection
```

### 2. 仮想環境のセットアップと依存関係のインストール
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS/Linux
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. ダッシュボードの起動
```bash
python interactive_dashboard.py
```

### 4. ブラウザでアクセス
```
http://127.0.0.1:8050/
```

**詳細な手順**: `7_direction_ditection/README_FOR_PROFESSOR.md` を参照してください。

## 📊 主要機能

### インタラクティブダッシュボード
- **2Dカメラマップ**: XZ平面上のカメラ配置とボット位置を可視化
- **3D骨格表示**: GroundTruthとMediaPipeの関節位置を並べて表示
- **誤差分析**: 角度誤差（Δθ、Δψ）および3D距離誤差のグラフ表示
- **相関分析**: 関節間の誤差相関をヒートマップで表示

### データ処理パイプライン
1. **データ読み込み**: CSV形式のGroundTruthとMediaPipeデータ
2. **座標変換**: 右手系→左手系、腰中心の相対座標系
3. **誤差計算**: 角度差（arctan2ベース）と3D距離
4. **統計分析**: 平均、中央値、標準偏差、最大・最小値
5. **相関分析**: Pearson相関係数による関節間誤差の関連性

## 📖 技術詳細

### 座標系の変換
- **GroundTruth**: 左手系、Y-up、ワールド座標
- **MediaPipe**: 正規化座標（0-1）→ 左手系、Y-up、腰原点相対座標

### 誤差指標
1. **Δθ（XY平面角度誤差）**: `arctan2(y, x)` の差、度数法で表示
2. **Δψ（XZ平面角度誤差）**: `arctan2(z, x)` の差、度数法で表示
3. **3D距離誤差**: `√((x_gt - x_mp)² + (y_gt - y_mp)² + (z_gt - z_mp)²)`

### 対応関節
- 肩（左右）、肘（左右）、手首（左右）
- 腰（左右）、膝（左右）、足首（左右）

## 🔬 研究成果

詳細な分析結果と考察は以下のファイルを参照：
- `7_direction_ditection/FINDINGS_AND_INSIGHTS.md`: 研究論文向け考察
- `7_direction_ditection/output/`: 統計データ、グラフ、相関分析結果

## 📦 必要な依存パッケージ

主要なPythonパッケージ：
- `pandas`: データ処理
- `numpy`: 数値計算
- `plotly`: インタラクティブグラフ
- `dash`: Webダッシュボード
- `seaborn`: ヒートマップ可視化
- `tqdm`: プログレスバー

完全なリストは `7_direction_ditection/requirements.txt` を参照。

## 🛠️ 開発者向け情報

### テストの実行
```bash
cd 7_direction_ditection
python tests/test_01_load_data.py
python tests/test_02_transform.py
python tests/test_03_visualize.py
python tests/test_04_full_pipeline.py
```

### データの再処理
```bash
python process_all_data.py  # すべてのフレーム・カメラのデータを処理
python compute_correlation.py  # 相関分析の実行
```

### 設定の変更
`7_direction_ditection/config.py` でパス、対象関節、Y範囲などを設定可能。

## ⚠️ 注意事項

- **画像ファイル**: サイズが大きいため、`.gitignore` で除外されています
  - PNG、JPEG、GIF、MP4などの画像・動画ファイル
  - `1_Output_Photos/` ディレクトリ全体
- **処理済みデータ**: CSV、HTML、MDファイルはGitで管理されています
- **仮想環境**: `venv/` ディレクトリは各ユーザーがローカルで作成する必要があります

## 📧 お問い合わせ

質問や不明点がありましたら、開発者までご連絡ください。

---

**プロジェクト開始**: 2026年1月
**最終更新**: 2026年1月10日
