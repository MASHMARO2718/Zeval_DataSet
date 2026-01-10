import pandas as pd
import numpy as np

mae = pd.read_csv('coordinate_angle_mae_combined.csv')
df = pd.read_csv('動的視点角度計算/viewpoint_angles_with_mae.csv')

mae_cameras = set(zip(mae['camera_x'], mae['camera_y'], mae['camera_z']))
csv_cameras = set(zip(df['camera_x'], df['camera_y'], df['camera_z']))
missing = list(mae_cameras - csv_cameras)

print('=== 欠損しているカメラ位置のMAEデータ詳細分析 ===\n')

mae_cols = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 
            'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip']

# 欠損しているカメラ位置のMAEデータを分析
all_nan_count = 0
partial_nan_count = 0
all_valid_count = 0
empty_folder_count = 0

for cam in missing:
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    
    # NaN値の数をカウント
    nan_count = sum(pd.isna(row[col]) for col in mae_cols)
    valid_count = sum(not pd.isna(row[col]) for col in mae_cols)
    
    if nan_count == len(mae_cols):
        all_nan_count += 1
    elif nan_count > 0:
        partial_nan_count += 1
    else:
        all_valid_count += 1
    
    # folder_nameが空かチェック
    if pd.isna(row['folder_name']) or str(row['folder_name']).strip() == '':
        empty_folder_count += 1

print(f'欠損しているカメラ位置: {len(missing)}個')
print(f'  すべてのMAE値がNaN: {all_nan_count}個')
print(f'  一部のMAE値がNaN: {partial_nan_count}個')
print(f'  すべてのMAE値が有効: {all_valid_count}個')
print(f'  folder_nameが空: {empty_folder_count}個')

# 詳細な例を表示
print('\n=== 詳細例 ===')
print('\n【すべてNaNの例】')
all_nan_examples = []
for cam in missing:
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    if all(pd.isna(row[col]) for col in mae_cols):
        all_nan_examples.append((cam, row))
        if len(all_nan_examples) <= 5:
            print(f'  カメラ{cam}: folder_name={row["folder_name"]}')

print(f'\n  合計: {len(all_nan_examples)}個')

print('\n【一部NaNの例】')
partial_nan_examples = []
for cam in missing:
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    nan_count = sum(pd.isna(row[col]) for col in mae_cols)
    if 0 < nan_count < len(mae_cols):
        partial_nan_examples.append((cam, row, nan_count))
        if len(partial_nan_examples) <= 5:
            print(f'  カメラ{cam}: NaN数={nan_count}/8')
            for col in mae_cols:
                if pd.isna(row[col]):
                    print(f'    {col}: NaN')

print(f'\n  合計: {len(partial_nan_examples)}個')

print('\n【すべて有効な例】')
all_valid_examples = []
for cam in missing:
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    if all(not pd.isna(row[col]) for col in mae_cols):
        all_valid_examples.append((cam, row))
        if len(all_valid_examples) <= 5:
            print(f'  カメラ{cam}: folder_name={row["folder_name"]}')
            print(f'    L_Elbow={row["L_Elbow"]:.2f}, R_Elbow={row["R_Elbow"]:.2f}')

print(f'\n  合計: {len(all_valid_examples)}個')

# CSVに含まれているカメラ位置のMAEデータも確認
print('\n=== CSVに含まれているカメラ位置のMAEデータ ===')
csv_nan_stats = []
for cam in list(csv_cameras)[:100]:  # 最初の100個をサンプル
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    nan_count = sum(pd.isna(row[col]) for col in mae_cols)
    csv_nan_stats.append(nan_count)

print(f'サンプル100個のNaN値の分布:')
print(f'  すべて有効: {csv_nan_stats.count(0)}個')
print(f'  一部NaN: {sum(1 for x in csv_nan_stats if 0 < x < len(mae_cols))}個')
print(f'  すべてNaN: {csv_nan_stats.count(len(mae_cols))}個')

# 元のスクリプトがNaNをスキップした可能性を確認
print('\n=== 推測 ===')
if all_nan_count > 0:
    print(f'✓ {all_nan_count}個のカメラ位置は、すべてのMAE値がNaNのためスキップされた可能性')
if partial_nan_count > 0:
    print(f'✓ {partial_nan_count}個のカメラ位置は、一部のMAE値がNaNのためスキップされた可能性')
if all_valid_count > 0:
    print(f'⚠ {all_valid_count}個のカメラ位置は、すべてのMAE値が有効なのにスキップされている（別の原因の可能性）')

