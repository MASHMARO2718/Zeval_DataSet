#!/usr/bin/env python3
"""
時系列グラフ作成の実行スクリプト（ラッパー）
"""
import os
import glob
import sys

# カレントディレクトリを取得（スクリプトの存在する場所）
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"作業ディレクトリ: {current_dir}")
os.chdir(current_dir)

# CSVファイルを明示的に取得
csv_pattern = os.path.join(current_dir, "CapturedFrames_*.csv")
csv_files = glob.glob(csv_pattern)
print(f"発見されたCSVファイル: {len(csv_files)}個")

if not csv_files:
    print("エラー: CSVファイルが見つかりません")
    print(f"検索パターン: {csv_pattern}")
    print(f"現在のディレクトリ: {os.getcwd()}")
    sys.exit(1)

# GT CSVパス
gt_csv = os.path.join(current_dir, "..", "..", "..", "synced_joint_positions.csv")
gt_csv = os.path.abspath(gt_csv)
print(f"Ground Truth CSV: {gt_csv}")

if not os.path.exists(gt_csv):
    print(f"エラー: Ground Truth CSVが見つかりません: {gt_csv}")
    sys.exit(1)

# 出力ディレクトリ
output_dir = "graphs"

# 関節リスト
joints = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip']

# スクリプトをインポートして実行
from plot_time_series_error import TimeSeriesErrorPlotter

print("時系列グラフ作成を開始...")
plotter = TimeSeriesErrorPlotter(output_dir=output_dir)

# 各CSVファイルを個別に処理（フルパスのCSVパターンを使用）
plotter.process_all_coordinates(csv_pattern, gt_csv, joints)

print("完了！")


