# 座標変換とPlotly可視化プロジェクト

## 📁 プロジェクト構造

```
7_direction_ditection/
├── README.md                    # このファイル
├── requirements.txt             # 依存ライブラリ
├── config.py                    # 設定ファイル
│
├── scripts/                     # 実装スクリプト
│   ├── __init__.py
│   ├── data_loader.py          # データ読み込み
│   ├── coordinate_transform.py # 座標変換
│   ├── plotly_visualizer.py    # Plotly可視化
│   ├── validation.py           # 検証ユーティリティ
│   └── logger.py               # ログ管理
│
├── tests/                       # テストスクリプト（段階的）
│   ├── test_01_load_data.py    # データ読み込みテスト
│   ├── test_02_transform.py    # 座標変換テスト
│   ├── test_03_visualize.py    # 可視化テスト
│   └── test_04_full_pipeline.py# 完全なパイプライン
│
├── output/                      # 出力ファイル
│   ├── html_reports/           # Plotly HTML
│   ├── logs/                   # ログファイル
│   ├── debug_data/             # デバッグ用データ
│   └── validation_results/     # 検証結果CSV
│
└── notebooks/                   # Jupyter Notebook（オプション）
    ├── 01_data_exploration.ipynb
    └── 02_interactive_test.ipynb
```

## 🚀 クイックスタート

### 1. ライブラリインストール
```bash
cd 7_direction_ditection
pip install -r requirements.txt
```

### 2. 段階的テスト実行
```bash
# Step 1: データ読み込みテスト
python tests/test_01_load_data.py

# Step 2: 座標変換テスト
python tests/test_02_transform.py

# Step 3: 可視化テスト
python tests/test_03_visualize.py

# Step 4: 完全なパイプライン
python tests/test_04_full_pipeline.py
```

### 3. 出力確認
- HTML: `output/html_reports/` フォルダ内のHTMLファイルをブラウザで開く
- ログ: `output/logs/latest.log` でデバッグ情報を確認

## 🐛 デバッグ機能

### ログレベル
- `DEBUG`: 詳細な処理情報
- `INFO`: 重要な処理フロー
- `WARNING`: 警告
- `ERROR`: エラー

### デバッグモード
`config.py` で `DEBUG_MODE = True` に設定すると：
- すべての中間データを `output/debug_data/` に保存
- 詳細なログ出力
- ステップごとの確認プロンプト

## 📊 データパス設定

`config.py` でデータパスを設定：
```python
BASE_DIR = Path(__file__).parent.parent  # Zeval_DataSetルート
GT_CSV = BASE_DIR / "synced_joint_positions.csv"
MP_DIR = BASE_DIR / "2_medidapipe_proccesed"
```

## ✅ テストの進め方

1. **test_01**: データが正しく読み込めるか確認
2. **test_02**: 座標変換が正しく動作するか確認（腰が原点になるか）
3. **test_03**: Plotlyで可視化できるか確認
4. **test_04**: 全体のパイプラインが動作するか確認

各テストは独立して実行可能で、前のステップが成功してから次に進めます。

