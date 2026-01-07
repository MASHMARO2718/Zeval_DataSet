---
license: mit
task_categories:
- image-classification
- pose-estimation
- keypoint-detection
tags:
- motion-tracking
- mediapipe
- ground-truth
- biomechanics
- computer-vision
pretty_name: MotionTrack Zeval Dataset
size_categories:
- 10K<n<100K
language:
- en
- ja
---

# MotionTrack Zeval Dataset 🎥

## 概要 / Overview

MotionTrack Zeval Datasetは、モーショントラッキングと姿勢推定アルゴリズムの評価用に作成された大規模データセットです。このデータセットには、複数カメラアングルから撮影された人間の動作画像と、それに対応するGround Truth（正解データ）、およびMediaPipeによる推定結果が含まれています。

This is a large-scale dataset for evaluating motion tracking and pose estimation algorithms. It includes human motion images captured from multiple camera angles, along with corresponding ground truth data and MediaPipe estimation results.

## データセット構成 / Dataset Structure

```
├── 1_Output_Photos/              # 出力画像データ（約61,629枚）
│   └── Y=1.0,2.0/               # カメラ高さ別の画像
├── 2_medidapipe_proccesed/      # MediaPipe処理済みCSVデータ
│   ├── Y=0.5,1.5/               # 288 CSVファイル
│   └── Y=1.0,2.0/               # 288 CSVファイル
├── 3_Cal_MAE/                   # MAE（平均絶対誤差）計算結果
│   ├── Y=0.5,1.5/               # 290 CSVファイル + スクリプト
│   └── Y=1.0,2.0/               # 290 CSVファイル + スクリプト
├── 4_MAE_HEATMAP/               # MAEヒートマップ可視化
│   ├── Y=0.5,1.5/               # 16 ヒートマップ画像
│   └── Y=1.0,2.0/               # 16 ヒートマップ画像
└── 5_max_angle_error/           # 最大角度誤差分析
    ├── calicuration/            # 校正結果
    ├── max_angle_error_heatmap/ # ヒートマップ
    └── Y=0.5,1.5/ & Y=1.0,2.0/ # 時系列グラフ
```

### ファイル形式 / File Formats

- **画像**: JPEG形式
- **座標データ**: CSV形式（フレームごとの関節座標）
- **評価結果**: CSV形式（MAE、角度誤差など）
- **可視化**: PNG形式（ヒートマップ、グラフ）

## 主な特徴 / Key Features

- ✅ **複数カメラ高さ**: Y=0.5m, 1.0m, 1.5m, 2.0m
- ✅ **Ground Truthデータ**: 高精度な正解データを含む
- ✅ **MediaPipe推定結果**: 比較用の自動推定結果
- ✅ **評価指標**: MAE、最大角度誤差、変動係数（CV）
- ✅ **関節角度**: 肩、肘、股関節、膝の角度データ
- ✅ **可視化済み**: ヒートマップと時系列グラフ

## 測定された関節 / Measured Joints

- 左右の肩 (L/R Shoulder)
- 左右の肘 (L/R Elbow)
- 左右の股関節 (L/R Hip)
- 左右の膝 (L/R Knee)

## 使用方法 / Usage

### 基本的な読み込み / Basic Loading

```python
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import cv2

# データセットをロード
dataset = load_dataset("Mashmaro/motiontrack-zeval-dataset")

# 画像を読み込む例
img_path = "1_Output_Photos/Y=1.0,2.0/camera01_frame001.jpg"
image = cv2.imread(img_path)

# MediaPipe処理済みデータを読み込む例
df = pd.read_csv("2_medidapipe_proccesed/Y=1.0,2.0/camera01_results.csv")
print(df.head())
```

### MAE（平均絶対誤差）の分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# MAEデータを読み込む
mae_data = pd.read_csv("4_MAE_HEATMAP/Y=0.5,1.5/coordinate_angle_mae.csv")

# 関節ごとのMAEを可視化
mae_data.groupby('joint').mean().plot(kind='bar')
plt.title('Mean Absolute Error by Joint')
plt.ylabel('MAE (degrees)')
plt.show()
```

### ヒートマップの表示

```python
from PIL import Image

# ヒートマップを読み込む
heatmap = Image.open("4_MAE_HEATMAP/Y=0.5,1.5/heatmap_r_elbow_y0.5.png")
heatmap.show()
```

## 評価指標 / Evaluation Metrics

1. **MAE (Mean Absolute Error)**: 平均絶対誤差
2. **Max Angle Error**: 最大角度誤差
3. **CV (Coefficient of Variation)**: 変動係数
4. **Frame-wise Error**: フレームごとの誤差

## ユースケース / Use Cases

- 🔬 姿勢推定アルゴリズムのベンチマーク
- 📊 MediaPipeの精度評価
- 🤖 機械学習モデルのトレーニング・検証
- 📈 バイオメカニクス研究
- 🎯 カメラ配置の最適化研究

## システム要件 / Requirements

```bash
pip install pandas numpy opencv-python matplotlib seaborn mediapipe
```

## データ収集方法 / Data Collection Method

- **撮影環境**: 複数カメラによる同期撮影
- **カメラ高さ**: 0.5m, 1.0m, 1.5m, 2.0m
- **フレームレート**: 30 FPS
- **解像度**: 1920x1080

## ライセンス / License

このデータセットは **MIT License** の下で公開されています。

- ✅ 商用利用可能
- ✅ 修正・再配布可能
- ✅ 私的利用可能
- ⚠️ ライセンス表示と著作権表示が必要

## 引用 / Citation

もし研究や論文でこのデータセットを使用する場合は、以下を引用してください：

```bibtex
@dataset{motiontrack_zeval_2026,
  title={MotionTrack Zeval Dataset: A Comprehensive Dataset for Motion Tracking and Pose Estimation Evaluation},
  author={Mashmaro},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/Mashmaro/motiontrack-zeval-dataset}},
  note={Dataset for evaluating motion tracking algorithms with ground truth and MediaPipe results}
}
```

## 関連研究 / Related Work

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Human3.6M Dataset
- COCO Keypoint Dataset

## 制限事項と注意点 / Limitations

- データセットには特定の動作パターンのみが含まれています
- 照明条件は一定の環境下で撮影されています
- 被験者の多様性には限りがあります
- オクルージョン（隠れ）のケースは限定的です

## 更新履歴 / Changelog

### Version 1.0 (2026-01-05)
- 初回リリース
- 約61,629枚の画像を含む
- MediaPipe処理済みデータを追加
- MAEヒートマップと時系列分析を追加

## サポート / Support

質問やフィードバックがある場合は、以下にお問い合わせください：

- 📧 Email: Mashmaro@users.noreply.huggingface.co
- 🐛 Issues: https://huggingface.co/datasets/Mashmaro/motiontrack-zeval-dataset
- 💬 Discussion: https://huggingface.co/datasets/Mashmaro/motiontrack-zeval-dataset/discussions

## 謝辞 / Acknowledgments

このデータセットの作成にあたり、以下のツールとライブラリを使用しました：

- MediaPipe by Google
- OpenCV
- Python scientific computing ecosystem (NumPy, Pandas, Matplotlib)

---

**🌟 このデータセットが役に立った場合は、スターをつけてください！**

**📚 詳細なドキュメント**: https://huggingface.co/datasets/Mashmaro/motiontrack-zeval-dataset


