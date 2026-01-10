import pandas as pd

df = pd.read_csv('動的視点角度計算/viewpoint_angles_with_mae.csv')
mae = pd.read_csv('coordinate_angle_mae_combined.csv')
pos = pd.read_csv('synced_joint_positions.csv')

print('=== フレーム数の不一致を確認 ===\n')

print('【synced_joint_positions.csv】')
print(f'  総行数: {len(pos)}')
print(f'  フレーム範囲: {pos["Frame"].min()} to {pos["Frame"].max()}')
print(f'  ユニークなフレーム数: {pos["Frame"].nunique()}')

print('\n【viewpoint_angles_with_mae.csv】')
print(f'  総行数: {len(df)}')
print(f'  フレーム範囲: {int(df["frame_id"].min())} to {int(df["frame_id"].max())}')
print(f'  ユニークなフレーム数: {df["frame_id"].nunique()}')

print('\n【coordinate_angle_mae_combined.csv】')
print(f'  総行数: {len(mae)}')
print(f'  ユニークなカメラ位置数: {mae[["camera_x", "camera_y", "camera_z"]].drop_duplicates().shape[0]}')

print('\n=== 行数の計算 ===')
expected_rows = len(mae) * len(pos)
actual_rows = len(df)
print(f'期待される行数: {len(mae)} カメラ × {len(pos)} フレーム = {expected_rows}')
print(f'実際の行数: {actual_rows}')
print(f'差: {expected_rows - actual_rows} 行 ({((expected_rows - actual_rows) / expected_rows * 100):.1f}%)')

print('\n=== カメラ位置の比較 ===')
mae_cameras = set(zip(mae['camera_x'], mae['camera_y'], mae['camera_z']))
csv_cameras = set(zip(df['camera_x'], df['camera_y'], df['camera_z']))
print(f'MAEデータのカメラ位置数: {len(mae_cameras)}')
print(f'CSVデータのカメラ位置数: {len(csv_cameras)}')

missing_cameras = mae_cameras - csv_cameras
extra_cameras = csv_cameras - mae_cameras

if missing_cameras:
    print(f'\n【MAEにあってCSVにないカメラ位置】: {len(missing_cameras)}個')
    print('  例:')
    for i, cam in enumerate(list(missing_cameras)[:10], 1):
        print(f'    {i}. {cam}')

if extra_cameras:
    print(f'\n【CSVにあってMAEにないカメラ位置】: {len(extra_cameras)}個')
    print('  例:')
    for i, cam in enumerate(list(extra_cameras)[:10], 1):
        print(f'    {i}. {cam}')

print('\n=== フレームの比較 ===')
pos_frames = set(pos['Frame'].unique())
csv_frames = set(df['frame_id'].unique())
missing_frames = pos_frames - csv_frames
extra_frames = csv_frames - pos_frames

if missing_frames:
    print(f'【syncedにあってCSVにないフレーム】: {sorted(missing_frames)}')
if extra_frames:
    print(f'【CSVにあってsyncedにないフレーム】: {sorted(extra_frames)}')
if not missing_frames and not extra_frames:
    print('フレームは一致しています')

print('\n=== 各カメラ位置のフレーム数を確認 ===')
camera_frame_counts = df.groupby(['camera_x', 'camera_y', 'camera_z']).size()
print(f'各カメラ位置のフレーム数統計:')
print(camera_frame_counts.describe())
print(f'\nフレーム数が113未満のカメラ位置: {(camera_frame_counts < 113).sum()}個')
if (camera_frame_counts < 113).any():
    print('  例:')
    incomplete = camera_frame_counts[camera_frame_counts < 113].head(10)
    for (cx, cy, cz), count in incomplete.items():
        print(f'    カメラ({cx}, {cy}, {cz}): {count}フレーム (欠損: {113 - count}フレーム)')

