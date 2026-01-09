"""
テスト03: Plotly可視化テスト
インタラクティブな3D可視化が正しく動作するか確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from scripts.data_loader import DataLoader
from scripts.coordinate_transform import CoordinateTransformer
from scripts.plotly_visualizer import PlotlyVisualizer
from scripts.logger import get_logger


def main():
    """メイン処理"""
    # ロガー初期化
    logger = get_logger("Test03_Visualize")
    logger.section("Test 03: Plotly Visualization")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    success_count = 0
    total_tests = 4
    
    # データ読み込みと変換
    logger.step(1, "Load and transform data")
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
            logger.error("No common frames found")
            return 1
        
        # 最初の共通フレームを使用
        frame_id = common_frames[0]
        logger.info(f"Using frame {frame_id} (common frame)")
        
        # 1フレームの座標取得
        gt_coords_raw = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
        mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
        
        # 座標変換
        transformer = CoordinateTransformer()
        gt_coords_relative, hip_center = transformer.transform_ground_truth(gt_coords_raw)
        
        # 差分計算
        differences = transformer.calculate_differences(gt_coords_relative, mp_coords)
        
        logger.info(f"✅ Data loaded and transformed")
        logger.info(f"   Frame: {frame_id}")
        logger.info(f"   GT joints: {len(gt_coords_relative)}")
        logger.info(f"   MP joints: {len(mp_coords)}")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Data loading/transformation failed: {e}")
        return 1
    
    # Plotly可視化初期化
    logger.step(2, "Initialize PlotlyVisualizer")
    try:
        visualizer = PlotlyVisualizer()
        logger.info("✅ PlotlyVisualizer initialized")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ PlotlyVisualizer initialization failed: {e}")
        return 1
    
    # 可視化1: 左右並べて比較
    logger.step(3, "Create side-by-side plot")
    try:
        fig1 = visualizer.plot_side_by_side(gt_coords_relative, mp_coords, frame_id)
        visualizer.save_html(fig1, f"frame_{frame_id:04d}_side_by_side.html")
        logger.info("✅ Side-by-side plot created")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Side-by-side plot failed: {e}")
    
    # 可視化2: オーバーレイ表示
    logger.step(4, "Create overlay plot")
    try:
        fig2 = visualizer.plot_overlay(gt_coords_relative, mp_coords, frame_id, differences)
        visualizer.save_html(fig2, f"frame_{frame_id:04d}_overlay.html")
        logger.info("✅ Overlay plot created")
    except Exception as e:
        logger.error(f"❌ Overlay plot failed: {e}")
    
    # 可視化3: 多視点表示
    logger.step(5, "Create multi-view plot")
    try:
        fig3 = visualizer.plot_multi_view(gt_coords_relative, mp_coords, frame_id)
        visualizer.save_html(fig3, f"frame_{frame_id:04d}_multi_view.html")
        logger.info("✅ Multi-view plot created")
    except Exception as e:
        logger.error(f"❌ Multi-view plot failed: {e}")
    
    # 可視化4: 誤差テーブル
    logger.step(6, "Create error table")
    try:
        fig4 = visualizer.create_error_table(differences)
        visualizer.save_html(fig4, f"frame_{frame_id:04d}_error_table.html")
        logger.info("✅ Error table created")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Error table failed: {e}")
    
    # 結果サマリー
    logger.section("Test Summary")
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("[PASS] ===== ALL TESTS PASSED =====")
        print("\n" + "="*60)
        print("[PASS] TEST 03 PASSED - Plotly visualization successful!")
        print("="*60)
        print(f"\nHTML files created in: {config.HTML_DIR}")
        print(f"Open them in a browser to view interactive 3D plots!")
        print(f"\nYou can proceed to test_04_full_pipeline.py")
        print(f"Log file: {config.LOG_DIR / 'latest.log'}")
        
        # HTML ファイルのリスト表示
        html_files = list(config.HTML_DIR.glob("frame_*.html"))
        if html_files:
            print(f"\nGenerated HTML files:")
            for html_file in sorted(html_files):
                print(f"  - {html_file.name}")
        
        return 0
    else:
        logger.error("[FAIL] ===== SOME TESTS FAILED =====")
        print("\n" + "="*60)
        print(f"[FAIL] TEST 03 FAILED - {success_count}/{total_tests} tests passed")
        print("="*60)
        print(f"\nPlease check the log file: {config.LOG_DIR / 'latest.log'}")
        print("Fix the issues before proceeding to the next test.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

