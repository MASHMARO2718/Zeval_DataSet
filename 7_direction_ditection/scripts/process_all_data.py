"""
全データ処理スクリプト
全フレーム・全カメラ位置に対して座標変換と角度差計算を実行し、
わかりやすいCSVファイルに出力
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from scripts.data_loader import DataLoader
from scripts.coordinate_transform import CoordinateTransformer
from scripts.logger import get_logger


def process_single_frame(loader, transformer, gt_df, mp_df, frame_id, camera_name):
    """
    単一フレーム・単一カメラの処理
    
    Returns:
        list of dict: 各関節の処理結果
    """
    try:
        # 座標取得
        gt_coords_raw = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
        mp_coords_raw = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
        
        if not gt_coords_raw or not mp_coords_raw:
            return None
        
        # 座標変換（両方とも相対座標化）
        gt_coords_relative, gt_hip_center = transformer.transform_ground_truth(gt_coords_raw)
        mp_coords_relative, mp_hip_center = transformer.transform_mediapipe(mp_coords_raw)
        
        # 差分計算
        differences = transformer.calculate_differences(gt_coords_relative, mp_coords_relative)
        
        # 各関節の結果をリスト化
        results = []
        for joint_name, diff in differences.items():
            result = {
                'frame_id': frame_id,
                'camera': camera_name,
                'joint': joint_name,
                # GroundTruth相対座標
                'gt_x': diff['gt_coord'][0],
                'gt_y': diff['gt_coord'][1],
                'gt_z': diff['gt_coord'][2],
                # MediaPipe相対座標
                'mp_x': diff['mp_coord'][0],
                'mp_y': diff['mp_coord'][1],
                'mp_z': diff['mp_coord'][2],
                # 角度
                'theta_gt_deg': diff['theta_gt'] * 180 / np.pi,
                'theta_mp_deg': diff['theta_mp'] * 180 / np.pi,
                'delta_theta_deg': diff['delta_theta_deg'],
                'psi_gt_deg': diff['psi_gt'] * 180 / np.pi,
                'psi_mp_deg': diff['psi_mp'] * 180 / np.pi,
                'delta_psi_deg': diff['delta_psi_deg'],
                # 3D誤差（参考値）
                'error_3d': diff['error_3d'],
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        return None


def create_summary(df_detailed):
    """
    詳細データからサマリーを作成
    
    Args:
        df_detailed: 詳細データのDataFrame
        
    Returns:
        DataFrame: サマリーデータ
    """
    summary_data = []
    
    # フレーム・カメラごとに集計
    for (frame_id, camera), group in df_detailed.groupby(['frame_id', 'camera']):
        # HIPを除外（原点のため角度が不安定）
        group_no_hip = group[~group['joint'].str.contains('HIP')]
        
        if len(group_no_hip) == 0:
            continue
        
        summary = {
            'frame_id': frame_id,
            'camera': camera,
            'num_joints': len(group_no_hip),
            'mean_abs_delta_theta': group_no_hip['delta_theta_deg'].abs().mean(),
            'mean_abs_delta_psi': group_no_hip['delta_psi_deg'].abs().mean(),
            'max_abs_delta_theta': group_no_hip['delta_theta_deg'].abs().max(),
            'max_abs_delta_psi': group_no_hip['delta_psi_deg'].abs().max(),
            'median_abs_delta_theta': group_no_hip['delta_theta_deg'].abs().median(),
            'median_abs_delta_psi': group_no_hip['delta_psi_deg'].abs().median(),
            'std_delta_theta': group_no_hip['delta_theta_deg'].std(),
            'std_delta_psi': group_no_hip['delta_psi_deg'].std(),
        }
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def create_joint_summary(df_detailed):
    """
    関節ごとの統計を作成
    
    Args:
        df_detailed: 詳細データのDataFrame
        
    Returns:
        DataFrame: 関節別サマリー
    """
    # HIPを除外
    df_no_hip = df_detailed[~df_detailed['joint'].str.contains('HIP')]
    
    joint_summary = df_no_hip.groupby('joint').agg({
        'delta_theta_deg': ['mean', 'std', 'min', 'max', lambda x: x.abs().mean()],
        'delta_psi_deg': ['mean', 'std', 'min', 'max', lambda x: x.abs().mean()],
        'error_3d': ['mean', 'std', 'min', 'max'],
    }).round(2)
    
    # カラム名をフラット化
    joint_summary.columns = ['_'.join(col).strip() for col in joint_summary.columns.values]
    joint_summary.columns = [
        'theta_mean', 'theta_std', 'theta_min', 'theta_max', 'theta_abs_mean',
        'psi_mean', 'psi_std', 'psi_min', 'psi_max', 'psi_abs_mean',
        'error3d_mean', 'error3d_std', 'error3d_min', 'error3d_max',
    ]
    
    return joint_summary.reset_index()


def main():
    """メイン処理"""
    logger = get_logger("ProcessAllData")
    logger.section("Process All Data")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    output_dir = config.OUTPUT_DIR / "processed_data"
    output_dir.mkdir(exist_ok=True)
    
    # 初期化
    logger.step(1, "Initialize components")
    loader = DataLoader()
    transformer = CoordinateTransformer()
    
    # GroundTruth読み込み
    logger.step(2, "Load GroundTruth data")
    gt_df = loader.load_ground_truth()
    logger.info(f"GroundTruth: {len(gt_df)} rows")
    
    # カメラ位置取得（全てのY範囲）
    logger.step(3, "Get camera positions from all Y ranges")
    all_cameras = []
    for y_range in config.Y_RANGES:
        cameras_in_range = loader.list_available_cameras(y_range=y_range)
        all_cameras.extend(cameras_in_range)
        logger.info(f"Found {len(cameras_in_range)} cameras in {y_range}")
    
    cameras = all_cameras
    logger.info(f"Total cameras: {len(cameras)}")
    
    # 処理するカメラ数を制限（テスト用）
    # 全データを処理する場合はこの行をコメントアウト
    # cameras = cameras[:5]  # 最初の5カメラのみ
    logger.info(f"Processing {len(cameras)} cameras")
    
    # 全データ処理
    logger.step(4, "Process all frames and cameras")
    all_results = []
    
    total_cameras = len(cameras)
    for camera_idx, camera_name in enumerate(cameras, 1):
        logger.info(f"\n[{camera_idx}/{total_cameras}] Processing camera: {camera_name}")
        
        try:
            # MediaPipe読み込み
            mp_df = loader.load_mediapipe(camera_name)
            
            if mp_df is None or len(mp_df) == 0:
                logger.warning(f"  No MediaPipe data for {camera_name}")
                continue
            
            # 共通フレームを取得
            gt_frames = set(gt_df['Frame'].unique())
            mp_frames = set(mp_df['frame_id'].unique())
            common_frames = sorted(gt_frames.intersection(mp_frames))
            
            if not common_frames:
                logger.warning(f"  No common frames for {camera_name}")
                continue
            
            logger.info(f"  Processing {len(common_frames)} frames")
            
            # 各フレームを処理
            for frame_id in tqdm(common_frames, desc=f"  {camera_name}", leave=False):
                results = process_single_frame(
                    loader, transformer, gt_df, mp_df, frame_id, camera_name
                )
                
                if results:
                    all_results.extend(results)
            
            logger.info(f"  ✓ Completed {camera_name}: {len(common_frames)} frames")
            
        except Exception as e:
            logger.error(f"  ✗ Failed {camera_name}: {e}")
            continue
    
    # 結果をDataFrameに変換
    logger.step(5, "Create DataFrames")
    if not all_results:
        logger.error("No results to process!")
        return 1
    
    df_detailed = pd.DataFrame(all_results)
    logger.info(f"Detailed results: {len(df_detailed)} rows")
    
    # サマリー作成
    df_summary = create_summary(df_detailed)
    logger.info(f"Summary results: {len(df_summary)} rows")
    
    df_joint_summary = create_joint_summary(df_detailed)
    logger.info(f"Joint summary: {len(df_joint_summary)} rows")
    
    # CSV出力
    logger.step(6, "Save CSV files")
    
    # 1. 詳細データ（全関節・全フレーム）
    detailed_file = output_dir / "detailed_results.csv"
    df_detailed.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ Saved: {detailed_file}")
    
    # 2. フレーム・カメラ別サマリー
    summary_file = output_dir / "frame_camera_summary.csv"
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ Saved: {summary_file}")
    
    # 3. 関節別統計
    joint_summary_file = output_dir / "joint_summary.csv"
    df_joint_summary.to_csv(joint_summary_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ Saved: {joint_summary_file}")
    
    # 統計サマリー表示
    logger.section("Overall Statistics")
    logger.info(f"Total frames processed: {df_detailed['frame_id'].nunique()}")
    logger.info(f"Total cameras processed: {df_detailed['camera'].nunique()}")
    logger.info(f"Total data points: {len(df_detailed)}")
    logger.info(f"\nAngle Error Statistics (excluding HIP):")
    logger.info(f"  Mean |Δθ|: {df_summary['mean_abs_delta_theta'].mean():.2f}°")
    logger.info(f"  Mean |Δψ|: {df_summary['mean_abs_delta_psi'].mean():.2f}°")
    logger.info(f"  Median |Δθ|: {df_summary['median_abs_delta_theta'].median():.2f}°")
    logger.info(f"  Median |Δψ|: {df_summary['median_abs_delta_psi'].median():.2f}°")
    
    logger.info("\n" + "="*60)
    logger.info("[SUCCESS] All data processed successfully!")
    logger.info("="*60)
    logger.info(f"\nOutput files:")
    logger.info(f"  1. {detailed_file}")
    logger.info(f"  2. {summary_file}")
    logger.info(f"  3. {joint_summary_file}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

