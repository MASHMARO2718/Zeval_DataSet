"""
テスト01: データ読み込みテスト
GroundTruthとMediaPipeデータが正しく読み込めるか確認
"""

import sys
import os
from pathlib import Path

# Windows UTF-8対応
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from scripts.data_loader import DataLoader
from scripts.logger import get_logger


def main():
    """メイン処理"""
    # ロガー初期化
    logger = get_logger("Test01_LoadData")
    logger.section("Test 01: Data Loading")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    # データローダー初期化
    logger.step(1, "Initialize DataLoader")
    loader = DataLoader()
    
    success_count = 0
    total_tests = 4
    
    # テスト1: GroundTruth読み込み
    logger.step(2, "Load GroundTruth data")
    try:
        gt_df = loader.load_ground_truth()
        logger.info(f"✅ GroundTruth loaded: {gt_df.shape}")
        logger.info(f"   Columns: {gt_df.columns.tolist()[:10]}...")
        
        # フレーム範囲確認
        min_frame, max_frame = loader.get_frame_range(gt_df, is_mediapipe=False)
        logger.info(f"   Frame range: {min_frame} - {max_frame}")
        
        success_count += 1
    except Exception as e:
        logger.error(f"❌ GroundTruth loading failed: {e}")
    
    # テスト2: 利用可能なカメラのリスト
    logger.step(3, "List available cameras")
    try:
        cameras = loader.list_available_cameras()
        logger.info(f"✅ Found {len(cameras)} cameras")
        logger.info(f"   First 5: {cameras[:5]}")
        
        if len(cameras) > 0:
            success_count += 1
        else:
            logger.error("❌ No cameras found")
    except Exception as e:
        logger.error(f"❌ Camera listing failed: {e}")
        cameras = []
    
    # テスト3: MediaPipe読み込み
    if cameras:
        logger.step(4, "Load MediaPipe data")
        try:
            mp_df = loader.load_mediapipe(cameras[0])
            logger.info(f"✅ MediaPipe loaded: {mp_df.shape}")
            logger.info(f"   Columns: {mp_df.columns.tolist()}")
            
            # フレーム範囲確認
            min_frame, max_frame = loader.get_frame_range(mp_df, is_mediapipe=True)
            logger.info(f"   Frame range: {min_frame} - {max_frame}")
            
            success_count += 1
        except Exception as e:
            logger.error(f"❌ MediaPipe loading failed: {e}")
            mp_df = None
    else:
        logger.warning("⚠️  Skipping MediaPipe loading (no cameras)")
        mp_df = None
    
    # テスト4: 1フレームの座標取得
    if mp_df is not None and not gt_df.empty:
        logger.step(5, "Extract single frame coordinates")
        try:
            # 共通のフレームを見つける
            gt_frames = set(gt_df['Frame'].unique())
            mp_frames = set(mp_df['frame_id'].unique())
            common_frames = sorted(gt_frames.intersection(mp_frames))
            
            if not common_frames:
                logger.error("No common frames found between GroundTruth and MediaPipe")
            else:
                frame_id = common_frames[0]
                logger.info(f"Using frame {frame_id} (common frame)")
                
                gt_coords = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
                mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
                
                logger.info(f"✅ Extracted coordinates for frame {frame_id}")
                logger.info(f"   GroundTruth joints: {len(gt_coords)}")
                logger.info(f"   MediaPipe joints: {len(mp_coords)}")
                
                # サンプル表示
                if 'LEFT_SHOULDER' in gt_coords:
                    logger.info(f"   Sample (GT LEFT_SHOULDER): {gt_coords['LEFT_SHOULDER']}")
                if 'LEFT_SHOULDER' in mp_coords:
                    logger.info(f"   Sample (MP LEFT_SHOULDER): {mp_coords['LEFT_SHOULDER']}")
                
                if len(gt_coords) > 0 and len(mp_coords) > 0:
                    success_count += 1
                else:
                    logger.warning("Some coordinates are empty")
        except Exception as e:
            logger.error(f"❌ Coordinate extraction failed: {e}")
    else:
        logger.warning("⚠️  Skipping coordinate extraction (data not loaded)")
    
    # 結果サマリー
    logger.section("Test Summary")
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("[PASS] ===== ALL TESTS PASSED =====")
        print("\n" + "="*60)
        print("[PASS] TEST 01 PASSED - Data loading successful!")
        print("="*60)
        print(f"\nYou can proceed to test_02_transform.py")
        print(f"Log file: {config.LOG_DIR / 'latest.log'}")
        return 0
    else:
        logger.error("[FAIL] ===== SOME TESTS FAILED =====")
        print("\n" + "="*60)
        print(f"[FAIL] TEST 01 FAILED - {success_count}/{total_tests} tests passed")
        print("="*60)
        print(f"\nPlease check the log file: {config.LOG_DIR / 'latest.log'}")
        print("Fix the issues before proceeding to the next test.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

