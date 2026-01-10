# 🚀 クイックスタートガイド

## 📦 セットアップ（初回のみ）

### 1. ライブラリのインストール

```bash
cd 7_direction_ditection
pip install -r requirements.txt
```

### 2. 設定の確認

```bash
python config.py
```

これで以下が作成されます：
- `output/html_reports/` - HTML可視化ファイル
- `output/logs/` - ログファイル
- `output/debug_data/` - デバッグデータ
- `output/validation_results/` - 検証結果

---

## 🧪 テストの実行（推奨順序）

### Step 1: データ読み込みテスト

```bash
python tests/test_01_load_data.py
```

**確認事項:**
- ✅ GroundTruthデータが読み込めるか
- ✅ MediaPipeデータが読み込めるか
- ✅ カメラ位置のリストが取得できるか
- ✅ 1フレームの座標が抽出できるか

**成功したら:** test_02へ進む

---

### Step 2: 座標変換テスト

```bash
python tests/test_02_transform.py
```

**確認事項:**
- ✅ 右手系→左手系変換が動作するか
- ✅ 腰が原点(0,0,0)になっているか
- ✅ 座標範囲が妥当か
- ✅ 左右対称性が保たれているか
- ✅ GroundTruthとMediaPipeの差分が計算できるか

**成功したら:** test_03へ進む

**デバッグデータ:** `output/debug_data/` に中間データが保存されます

---

### Step 3: Plotly可視化テスト

```bash
python tests/test_03_visualize.py
```

**確認事項:**
- ✅ 左右並べて比較プロットが生成されるか
- ✅ オーバーレイプロットが生成されるか
- ✅ 多視点プロットが生成されるか
- ✅ 誤差テーブルが生成されるか

**生成されるHTMLファイル:**
- `frame_0000_side_by_side.html` - 左右比較
- `frame_0000_overlay.html` - オーバーレイ
- `frame_0000_multi_view.html` - 多視点
- `frame_0000_error_table.html` - 誤差テーブル

**HTMLファイルの確認:**
```bash
# Windowsの場合
explorer output\html_reports

# または直接ブラウザで開く
start output\html_reports\frame_0000_side_by_side.html
```

**成功したら:** test_04へ進む

---

### Step 4: 完全パイプラインテスト

```bash
python tests/test_04_full_pipeline.py
```

**確認事項:**
- ✅ 複数フレーム（5フレーム）が処理されるか
- ✅ 各フレームのHTMLが生成されるか
- ✅ 結果CSVが生成されるか
- ✅ 統計サマリーが正しいか

**生成されるファイル:**
- `output/html_reports/pipeline_frame_*.html` - 各フレームのオーバーレイ
- `output/validation_results/pipeline_results_*.csv` - 統計結果

**成功したら:** 🎉 すべて完了！

---

## 📊 結果の確認

### ログファイルの確認

```bash
# 最新のログを確認
type output\logs\latest.log

# または
notepad output\logs\latest.log
```

### デバッグデータの確認

```bash
# デバッグデータフォルダを開く
explorer output\debug_data
```

以下のファイルが含まれます：
- `ground_truth_sample.csv` - GTサンプルデータ
- `mediapipe_*_sample.csv` - MPサンプルデータ
- `relative_coordinates.txt` - 相対座標
- `differences.txt` - 差分情報
- `coordinate_stats.txt` - 座標統計
- `validation_results.txt` - 検証結果

### HTML可視化の確認

```bash
explorer output\html_reports
```

**操作方法:**
- 🖱️ **左ドラッグ**: 回転
- 🖱️ **右ドラッグ**: 平行移動
- 🖱️ **スクロール**: ズーム
- 🖱️ **ダブルクリック**: リセット
- 💡 **ホバー**: 詳細情報表示

---

## ⚙️ 設定のカスタマイズ

### デバッグモードの切り替え

`config.py` を編集：

```python
# デバッグモードON（詳細ログ＋中間データ保存）
DEBUG_MODE = True

# デバッグモードOFF（通常ログのみ）
DEBUG_MODE = False
```

### Y範囲の変更

```python
# config.py で変更
DEFAULT_Y_RANGE = "Y=1.0.2.0"  # デフォルトを変更
```

または、各テストで直接指定：

```python
mp_df = loader.load_mediapipe(camera_position, y_range="Y=1.0.2.0")
```

---

## 🐛 トラブルシューティング

### 問題1: データが見つからない

```
FileNotFoundError: GroundTruth CSV not found
```

**解決策:**
- `config.py` のパス設定を確認
- `synced_joint_positions.csv` が存在するか確認

### 問題2: 腰が原点にない

```
⚠️  Hip is NOT at origin
```

**解決策:**
- `output/debug_data/relative_coordinates.txt` を確認
- 腰の座標が本当に (0,0,0) に近いか確認
- Y軸反転が正しく行われているか確認

### 問題3: Plotlyが表示されない

```
ModuleNotFoundError: No module named 'plotly'
```

**解決策:**
```bash
pip install plotly
```

### 問題4: カメラが見つからない

```
❌ No cameras found
```

**解決策:**
- `2_medidapipe_proccesed/Y=0.5,1.5/` フォルダが存在するか確認
- CSVファイルが含まれているか確認

---

## 📝 ログレベルの調整

`config.py` で設定：

```python
LOG_LEVEL = "DEBUG"   # 最も詳細
LOG_LEVEL = "INFO"    # 標準
LOG_LEVEL = "WARNING" # 警告のみ
LOG_LEVEL = "ERROR"   # エラーのみ
```

---

## 🎯 次のステップ

すべてのテストが成功したら：

1. **自分のデータで試す**
   - 異なるカメラ位置
   - 異なるY範囲
   - 異なるフレーム範囲

2. **可視化をカスタマイズ**
   - `scripts/plotly_visualizer.py` を編集
   - 色やサイズを変更
   - 追加のプロットを作成

3. **分析を拡張**
   - 時系列分析
   - 統計的分析
   - 機械学習への応用

---

## 📞 サポート

問題が解決しない場合：
1. `output/logs/latest.log` を確認
2. `output/debug_data/` のファイルを確認
3. デバッグモードをONにして再実行

---

**🎉 準備完了です！test_01から順番に実行してください！**




