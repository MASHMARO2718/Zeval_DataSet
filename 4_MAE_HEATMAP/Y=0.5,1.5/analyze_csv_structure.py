import pandas as pd
import numpy as np

df = pd.read_csv('viewpoint_angles_with_mae.csv')

print('=== CSVファイル構造分析 ===\n')
print(f'総行数: {len(df):,}')
print(f'総カラム数: {len(df.columns)}')
print(f'\nユニークなカメラ位置数: {df[["camera_x", "camera_y", "camera_z"]].drop_duplicates().shape[0]}')
print(f'ユニークなフレーム数: {df["frame_id"].nunique()}')
print(f'フレーム範囲: {int(df["frame_id"].min())} to {int(df["frame_id"].max())}')

print('\n=== カラム一覧 ===')
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()
    print(f'{i:2d}. {col:20s} ({dtype}, 欠損値: {null_count})')

print('\n=== 各カラムの統計情報 ===')
print('\n【カメラ位置】')
print(df[['camera_x', 'camera_y', 'camera_z']].describe())

print('\n【アバター位置】')
print(df[['avatar_x', 'avatar_y', 'avatar_z']].describe())

print('\n【視点角度】')
print(df[['theta_deg', 'phi_deg', 'distance', 'relative_height']].describe())

print('\n【MAE誤差】')
mae_cols = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip']
print(df[mae_cols].describe())

print('\n=== データの例（最初のカメラ位置の最初の5フレーム）===')
first_camera = df.iloc[0]['folder_name']
sample = df[df['folder_name'] == first_camera].head(5)
print(sample[['frame_id', 'camera_x', 'camera_y', 'camera_z', 'avatar_z', 'theta_deg', 'phi_deg', 'distance', 'L_Elbow']].to_string())

