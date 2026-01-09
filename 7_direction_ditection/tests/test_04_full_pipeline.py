"""
テスト04: 完全パイプラインテスト
複数フレームでの一連の処理を実行
"""

import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from scripts.data_loader import DataLoader
from scripts.coordinate_transform import CoordinateTransformer
from scripts.validation import Validator
from scripts.plotly_visualizer import PlotlyVisualizer
from scripts.logger import get_logger


def process_single_frame(loader, transformer, validator, visualizer, 
                        gt_df, mp_df, frame_id):
    """
    単一フレームの完全処理
    
    Args:
        loader: DataLoader
        transformer: CoordinateTransformer
        validator: Validator
        visualizer: PlotlyVisualizer
        gt_df: GroundTruthデータフレーム
        mp_df: MediaPipeデータフレーム
        frame_id: フレームID
        
    Returns:
        処理結果の辞書
    """
    logger = get_logger("FullPipeline")
    
    try:
        # 座標取得
        gt_coords_raw = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
        mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
        
        if not gt_coords_raw or not mp_coords:
            logger.warning(f"Frame {frame_id}: Missing data")
            return None
        
        # 座標変換
        gt_coords_relative, hip_center = transformer.transform_ground_truth(gt_coords_raw)
        
        # 差分計算
        differences = transformer.calculate_differences(gt_coords_relative, mp_coords)
        
        # 検証
        hip_at_origin = validator.check_hip_at_origin(gt_coords_relative)
        
        # 可視化（主要なもののみ）
        fig_overlay = visualizer.plot_overlay(gt_coords_relative, mp_coords, frame_id, differences)
        visualizer.save_html(fig_overlay, f"pipeline_frame_{frame_id:04d}_overlay.html")
        
        # 結果
        errors = [diff['error_3d'] for diff in differences.values()]
        result = {
            'frame_id': frame_id,
            'hip_at_origin': hip_at_origin,
            'num_joints': len(differences),
            'mean_error': sum(errors) / len(errors) if errors else 0,
            'max_error': max(errors) if errors else 0,
            'min_error': min(errors) if errors else 0,
        }
        
        logger.info(f"Frame {frame_id}: mean_error={result['mean_error']:.4f}, max_error={result['max_error']:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Frame {frame_id} processing failed: {e}")
        return None


def main():
    """メイン処理"""
    # ロガー初期化
    logger = get_logger("Test04_FullPipeline")
    logger.section("Test 04: Full Pipeline")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    # 初期化
    logger.step(1, "Initialize all components")
    try:
        loader = DataLoader()
        transformer = CoordinateTransformer()
        validator = Validator()
        visualizer = PlotlyVisualizer()
        logger.info("✅ All components initialized")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return 1
    
    # データ読み込み
    logger.step(2, "Load dataset")
    try:
        gt_df = loader.load_ground_truth()
        cameras = loader.list_available_cameras()
        
        if not cameras:
            logger.error("❌ No cameras found")
            return 1
        
        # 最初のカメラを使用
        camera_position = cameras[0]
        mp_df = loader.load_mediapipe(camera_position)
        
        logger.info(f"✅ Dataset loaded")
        logger.info(f"   Camera: {camera_position}")
        logger.info(f"   GroundTruth: {gt_df.shape}")
        logger.info(f"   MediaPipe: {mp_df.shape}")
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return 1
    
    # フレーム範囲の決定
    logger.step(3, "Determine frame range")
    try:
        # 共通のフレームを取得
        gt_frames = set(gt_df['Frame'].unique())
        mp_frames = set(mp_df['frame_id'].unique())
        common_frames = sorted(gt_frames.intersection(mp_frames))
        
        if not common_frames:
            logger.error("No common frames found")
            return 1
        
        # テスト用に最初の5フレームのみ処理
        test_frames = common_frames[:5]
        
        logger.info(f"✅ Frame range determined")
        logger.info(f"   Available: {frame_min} - {frame_max}")
        logger.info(f"   Testing frames: {test_frames}")
        
    except Exception as e:
        logger.error(f"❌ Frame range determination failed: {e}")
        return 1
    
    # 複数フレームの処理
    logger.step(4, f"Process {len(test_frames)} frames")
    results = []
    
    for frame_id in test_frames:
        logger.info(f"\n--- Processing Frame {frame_id} ---")
        result = process_single_frame(
            loader, transformer, validator, visualizer,
            gt_df, mp_df, frame_id
        )
        if result:
            results.append(result)
    
    # 結果の集計
    logger.step(5, "Aggregate results")
    if results:
        # DataFrameに変換
        results_df = pd.DataFrame(results)
        
        # CSV保存
        csv_path = config.VALIDATION_DIR / f"pipeline_results_{camera_position}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Results saved: {csv_path}")
        
        # 統計サマリー
        logger.info("\n=== Results Summary ===")
        logger.info(f"Frames processed: {len(results)}/{len(test_frames)}")
        logger.info(f"Mean error (avg): {results_df['mean_error'].mean():.4f}")
        logger.info(f"Max error (overall): {results_df['max_error'].max():.4f}")
        logger.info(f"Hip at origin (all): {results_df['hip_at_origin'].all()}")
        
        # テーブル表示
        print("\n" + "="*80)
        print("Pipeline Results:")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
    else:
        logger.error("❌ No frames processed successfully")
        return 1
    
    # 最終サマリー
    logger.section("Test Summary")
    
    success_rate = len(results) / len(test_frames)
    
    if success_rate == 1.0:
        logger.info("[PASS] ===== ALL FRAMES PROCESSED SUCCESSFULLY =====")
        print("\n" + "="*60)
        print("[PASS] TEST 04 PASSED - Full pipeline successful!")
        print("="*60)
        print(f"\nResults:")
        print(f"  - Frames processed: {len(results)}/{len(test_frames)}")
        print(f"  - Results CSV: {csv_path}")
        print(f"  - HTML reports: {config.HTML_DIR}")
        print(f"  - Log file: {config.LOG_DIR / 'latest.log'}")
        print(f"\n[SUCCESS] All tests completed successfully!")
        print(f"You can now use this pipeline for your analysis.")
        return 0
    else:
        logger.warning(f"[WARN] ===== {success_rate*100:.1f}% FRAMES PROCESSED =====")
        print("\n" + "="*60)
        print(f"[WARN] TEST 04 PARTIAL - {len(results)}/{len(test_frames)} frames processed")
        print("="*60)
        print(f"\nSome frames failed. Check log: {config.LOG_DIR / 'latest.log'}")
        return 0 if success_rate > 0.5 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

