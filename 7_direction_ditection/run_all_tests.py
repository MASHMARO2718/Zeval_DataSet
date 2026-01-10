"""
全テストを順番に実行するスクリプト
"""

import sys
import subprocess
from pathlib import Path

# プロジェクトルート
project_root = Path(__file__).parent

# テストスクリプトのリスト（順番に実行）
tests = [
    ("Test 01: Data Loading", "tests/test_01_load_data.py"),
    ("Test 02: Coordinate Transformation", "tests/test_02_transform.py"),
    ("Test 03: Plotly Visualization", "tests/test_03_visualize.py"),
    ("Test 04: Full Pipeline", "tests/test_04_full_pipeline.py"),
]


def run_test(test_name, test_script):
    """
    テストを実行
    
    Args:
        test_name: テスト名
        test_script: テストスクリプトのパス
        
    Returns:
        bool: 成功したかどうか
    """
    print("\n" + "="*80)
    print(f"🧪 Running: {test_name}")
    print("="*80)
    
    test_path = project_root / test_script
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(project_root),
            capture_output=False,  # 出力を直接表示
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {test_name} - PASSED")
            return True
        else:
            print(f"\n❌ {test_name} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ {test_name} - ERROR: {e}")
        return False


def main():
    """メイン処理"""
    print("="*80)
    print("🚀 Running All Tests")
    print("="*80)
    print(f"\nTotal tests: {len(tests)}")
    print("Tests will run in sequence. Each test must pass to continue.\n")
    
    input("Press Enter to start...")
    
    results = []
    
    for test_name, test_script in tests:
        success = run_test(test_name, test_script)
        results.append((test_name, success))
        
        if not success:
            print("\n" + "="*80)
            print("⚠️  TEST FAILED - STOPPING")
            print("="*80)
            print(f"\nPlease fix the issues in {test_name} before continuing.")
            print(f"Check the log file for details.")
            break
        
        # 最後のテスト以外は続行確認
        if test_script != tests[-1][1]:
            print("\n" + "-"*80)
            input("Press Enter to continue to next test...")
    
    # 最終サマリー
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now use the pipeline for your analysis.")
        print(f"Check output files in: {project_root / 'output'}")
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the test results and fix the issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)



