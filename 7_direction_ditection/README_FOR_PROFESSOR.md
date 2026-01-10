# インタラクティブダッシュボード実行手順

このドキュメントは、GroundTruthとMediaPipeの3D関節位置データを比較・可視化するインタラクティブダッシュボードの実行手順を説明します。

## 📋 前提条件

以下のソフトウェアがインストールされている必要があります：
- **Python 3.8以上**（推奨: Python 3.9 or 3.10）
- **Git**

## 🚀 実行手順

### 1. リポジトリのクローン

```bash
git clone <リポジトリURL>
cd Zeval_DataSet
```

### 2. プロジェクトディレクトリへ移動

```bash
cd 7_direction_ditection
```

### 3. 仮想環境のセットアップ

#### Windows (PowerShell)の場合:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
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

### 4. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 5. ダッシュボードの起動

```bash
python interactive_dashboard.py
```

### 6. ブラウザでアクセス

ターミナルに以下のようなメッセージが表示されます：

```
Dash is running on http://127.0.0.1:8050/
```

Webブラウザで以下のURLにアクセスしてください：
```
http://127.0.0.1:8050/
```

## 📊 ダッシュボードの機能

### 主要機能

1. **📍 2Dカメラマップ**
   - XZ平面上のカメラ位置を可視化
   - クリックでカメラを選択可能
   - 緑：データあり、灰色：データなし、黄色：選択中
   - 赤い点：ボット（GroundTruth腰）の位置

2. **🎯 3D骨格表示**
   - GroundTruthとMediaPipeの3D関節位置を並べて表示
   - マウス操作で回転・ズーム・パン可能
   - スティックフィギュア形式で骨格を可視化

3. **📈 誤差分析グラフ**
   - **角度誤差（Δθ、Δψ）** - XY平面とXZ平面の角度差
   - **3D距離誤差** - ユークリッド距離
   - フレーム別・カメラ別・関節別の詳細分析

4. **📊 相関分析**
   - 関節間の誤差相関をヒートマップで表示
   - θ（XY平面）、ψ（XZ平面）、3D距離の3つの指標

### データの操作

- **フレーム選択**: ドロップダウンで解析フレームを選択
- **Y座標選択**: カメラのY座標（高さ）を選択
- **カメラ選択**: マップをクリック、または手動ドロップダウンで選択

## 🛑 ダッシュボードの停止

ターミナルで `Ctrl + C` を押してください。

## 📁 データ構造

- `data/1_GroundTruth/`: Unity Kinematikaの3D関節位置データ（CSV）
- `data/2_medidapipe_proccesed/`: MediaPipe処理済みデータ（CSV）
- `output/`: 処理結果（CSV、HTML、相関分析結果）

## 📖 技術詳細

### 座標系
- **GroundTruth**: 左手系、Y-up
- **MediaPipe**: 正規化後、腰を原点とした相対座標系に変換

### 誤差指標
1. **角度誤差（Δθ）**: XY平面の方位角の差（arctan2ベース）
2. **角度誤差（Δψ）**: XZ平面の仰角の差（arctan2ベース）
3. **3D距離誤差**: ユークリッド距離 `√(Δx² + Δy² + Δz²)`

## ❓ トラブルシューティング

### ポート8050が既に使用されている場合
`interactive_dashboard.py` の最終行を編集：
```python
app.run(debug=True, port=8051)  # ポート番号を変更
```

### データが表示されない場合
1. `data/` ディレクトリにGroundTruthとMediaPipeのCSVファイルが存在するか確認
2. `output/` ディレクトリに処理済みCSVファイルが存在するか確認
3. 必要に応じて `python process_all_data.py` でデータを再処理

### 依存パッケージのエラー
```bash
pip install --upgrade -r requirements.txt
```

## 📧 お問い合わせ

質問や不明点がありましたら、開発者までご連絡ください。

---

**最終更新**: 2026年1月10日

