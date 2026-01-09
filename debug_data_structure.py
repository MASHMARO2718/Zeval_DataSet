"""
デバッグ用: データ構造の確認
"""
import pandas as pd
from pathlib import Path

# GroundTruthのデータ構造確認
gt_csv = Path('/mnt/c/projects/MOTIONTRACK/Zeval_DataSet/synced_joint_positions.csv')
df_gt = pd.read_csv(gt_csv)

print("=== GroundTruth Data ===")
print(f"Shape: {df_gt.shape}")
print(f"Columns (first 10): {list(df_gt.columns[:10])}")
print(f"Frame range: {df_gt['Frame'].min()} - {df_gt['Frame'].max()}")
print(f"\nSample row:")
print(df_gt.iloc[0])

# MediaPipeのデータ構造確認
mp_csv = Path('/mnt/c/projects/MOTIONTRACK/Zeval_DataSet/2_medidapipe_proccesed/Y=0.5,1.5/CapturedFrames_-1.0_0.5_-3.0.csv')
df_mp = pd.read_csv(mp_csv)

print("\n=== MediaPipe Data ===")
print(f"Shape: {df_mp.shape}")
print(f"Columns: {list(df_mp.columns)}")
print(f"Unique frames: {df_mp['frame_id'].nunique()}")
print(f"Frame IDs: {sorted(df_mp['frame_id'].unique())}")
print(f"Unique landmarks: {df_mp['landmark'].nunique()}")
print(f"Landmarks: {sorted(df_mp['landmark'].unique())}")
print(f"\nSample rows:")
print(df_mp.head(10))

# 各ファイルのフレームIDの分布を確認
print("\n=== Frame ID Distribution ===")
y_dir = Path('/mnt/c/projects/MOTIONTRACK/Zeval_DataSet/2_medidapipe_proccesed/Y=0.5,1.5')
for i, csv_file in enumerate(sorted(y_dir.glob('*.csv'))[:5]):
    df = pd.read_csv(csv_file)
    print(f"{csv_file.name}: frames {sorted(df['frame_id'].unique())}")
