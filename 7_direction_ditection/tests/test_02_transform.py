"""
テスト02: 座標変換テスト
右手系→左手系変換、相対座標化が正しく動作するか確認
"""

import sys
from pathlib import Path
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from scripts.data_loader import DataLoader
from scripts.coordinate_transform import CoordinateTransformer
from scripts.validation import Validator
from scripts.logger import get_logger


def main():
    """メイン処理"""
    # ロガー初期化
    logger = get_logger("Test02_Transform")
    logger.section("Test 02: Coordinate Transformation")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    success_count = 0
    total_tests = 5
    
    # データ読み込み
    logger.step(1, "Load data")
    try:
        loader = DataLoader()
        gt_df = loader.load_ground_truth()
        cameras = loader.list_available_cameras()
        
        if not cameras:
            logger.error("❌ No cameras found")
            return 1
        
        mp_df = loader.load_mediapipe(cameras[0])
        
        # 共通のフレームIDを見つける
        gt_frames = set(gt_df['Frame'].unique())
        mp_frames = set(mp_df['frame_id'].unique())
        common_frames = sorted(gt_frames.intersection(mp_frames))
        
        if not common_frames:
            logger.error("No common frames found between GroundTruth and MediaPipe")
            return 1
        
        # 最初の共通フレームを使用
        frame_id = common_frames[0]
        logger.info(f"Using frame {frame_id} (common frame)")
        
        # 1フレームの座標取得
        gt_coords_raw = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
        mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   GroundTruth joints: {len(gt_coords_raw)}")
        logger.info(f"   MediaPipe joints: {len(mp_coords)}")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return 1
    
    # 座標変換
    logger.step(2, "Transform coordinates")
    try:
        transformer = CoordinateTransformer()
        
        # GroundTruthの変換
        gt_coords_relative, gt_hip_center = transformer.transform_ground_truth(gt_coords_raw)
        
        # MediaPipeの変換（腰を原点とした相対座標に）
        mp_coords_relative, mp_hip_center = transformer.transform_mediapipe(mp_coords)
        
        logger.info(f"✅ Coordinate transformation completed")
        logger.info(f"   GT hip center: {gt_hip_center}")
        logger.info(f"   MP hip center: {mp_hip_center}")
        logger.info(f"   GT joints: {len(gt_coords_relative)}")
        logger.info(f"   MP joints: {len(mp_coords_relative)}")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Coordinate transformation failed: {e}")
        return 1
    
    # 検証1: 腰が原点にあるか
    logger.step(3, "Validate hip at origin")
    try:
        validator = Validator()
        hip_at_origin = validator.check_hip_at_origin(gt_coords_relative)
        
        if hip_at_origin:
            logger.info("✅ Hip is at origin")
            success_count += 1
        else:
            logger.error("❌ Hip is NOT at origin")
        
    except Exception as e:
        logger.error(f"❌ Hip validation failed: {e}")
    
    # 検証2: 座標範囲の確認
    logger.step(4, "Check coordinate ranges")
    try:
        validator = Validator()
        
        logger.info("\n--- GroundTruth Stats ---")
        gt_stats = validator.check_coordinate_ranges(gt_coords_relative)
        
        logger.info("\n--- MediaPipe Stats (Relative Coordinates) ---")
        mp_stats = validator.check_coordinate_ranges(mp_coords_relative)
        
        logger.info("✅ Coordinate ranges checked")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Coordinate range check failed: {e}")
    
    # 検証3: 対称性の確認
    logger.step(5, "Check symmetry")
    try:
        validator = Validator()
        
        logger.info("\n--- GroundTruth Symmetry ---")
        gt_symmetry = validator.check_symmetry(gt_coords_relative)
        
        logger.info("\n--- MediaPipe Symmetry (Relative Coordinates) ---")
        mp_symmetry = validator.check_symmetry(mp_coords_relative)
        
        logger.info("✅ Symmetry checked")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Symmetry check failed: {e}")
    
    # 差分計算
    logger.step(6, "Calculate differences")
    try:
        differences = transformer.calculate_differences(gt_coords_relative, mp_coords_relative)
        
        logger.info(f"✅ Differences calculated for {len(differences)} joints")
        
        # 主要関節の角度誤差表示（主要評価指標）
        key_joints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                     'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
        
        logger.info("\n--- Key Joint Angle Errors (Primary Metric) ---")
        for joint in key_joints:
            if joint in differences:
                diff = differences[joint]
                logger.info(f"{joint}: Δθ={diff['delta_theta_deg']:+.2f}°, Δψ={diff['delta_psi_deg']:+.2f}°")
        
    except Exception as e:
        logger.error(f"❌ Difference calculation failed: {e}")
    
    # 結果サマリー
    logger.section("Test Summary")
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("[PASS] ===== ALL TESTS PASSED =====")
        print("\n" + "="*60)
        print("[PASS] TEST 02 PASSED - Coordinate transformation successful!")
        print("="*60)
        print(f"\nYou can proceed to test_03_visualize.py")
        print(f"Log file: {config.LOG_DIR / 'latest.log'}")
        print(f"Debug data: {config.DEBUG_DATA_DIR}")
        return 0
    else:
        logger.error("[FAIL] ===== SOME TESTS FAILED =====")
        print("\n" + "="*60)
        print(f"[FAIL] TEST 02 FAILED - {success_count}/{total_tests} tests passed")
        print("="*60)
        print(f"\nPlease check the log file: {config.LOG_DIR / 'latest.log'}")
        print("Fix the issues before proceeding to the next test.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

