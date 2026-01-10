import pandas as pd

mae = pd.read_csv('coordinate_angle_mae_combined.csv')
df = pd.read_csv('動的視点角度計算/viewpoint_angles_with_mae.csv')

mae_cameras = set(zip(mae['camera_x'], mae['camera_y'], mae['camera_z']))
csv_cameras = set(zip(df['camera_x'], df['camera_y'], df['camera_z']))
missing = list(mae_cameras - csv_cameras)

print('欠損しているカメラ位置のMAEデータを確認:')
print(f'欠損数: {len(missing)}個\n')

# 欠損しているカメラ位置のMAEデータを確認
for i, cam in enumerate(missing[:10]):
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    
    print(f'{i+1}. カメラ{cam}:')
    print(f'   folder_name: {row["folder_name"]}')
    print(f'   L_Elbow: {row["L_Elbow"]} (NaN: {pd.isna(row["L_Elbow"])})')
    print(f'   R_Elbow: {row["R_Elbow"]} (NaN: {pd.isna(row["R_Elbow"])})')
    print(f'   L_Shoulder: {row["L_Shoulder"]} (NaN: {pd.isna(row["L_Shoulder"])})')
    
    # すべてのMAE値がNaNかチェック
    mae_cols = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 
                'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip']
    all_nan = all(pd.isna(row[col]) for col in mae_cols)
    print(f'   すべてNaN: {all_nan}')
    print()

print('\n=== まとめ ===')
all_nan_count = 0
for cam in missing:
    idx = mae[(mae['camera_x'] == cam[0]) & 
              (mae['camera_y'] == cam[1]) & 
              (mae['camera_z'] == cam[2])].index[0]
    row = mae.iloc[idx]
    mae_cols = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 
                'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip']
    if all(pd.isna(row[col]) for col in mae_cols):
        all_nan_count += 1

print(f'欠損している{len(missing)}個のカメラ位置のうち、')
print(f'すべてのMAE値がNaN: {all_nan_count}個')
print(f'一部またはすべてのMAE値がある: {len(missing) - all_nan_count}個')

