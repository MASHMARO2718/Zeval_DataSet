#!/usr/bin/env python3
"""
カメラ座標別の角度比較スクリプト
CapturedFrames_X_Y_Zの座標ごとにMediaPipeとGround Truthの角度MAEを計算
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import logging
import re

class CoordinateAngleComparator:
    """カメラ座標別の角度比較"""
    
    def __init__(self):
        self.setup_logging()
        
        # 関節定義（3点角）
        self.joint_definitions = {
            # 必須：ヒンジ4関節
            'L_Elbow': {
                'gt_points': ['LeftUpperArm', 'LeftLowerArm', 'LeftHand'],
                'mp_points': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST']
            },
            'R_Elbow': {
                'gt_points': ['RightUpperArm', 'RightLowerArm', 'RightHand'],
                'mp_points': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'L_Knee': {
                'gt_points': ['LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot'],
                'mp_points': ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']
            },
            'R_Knee': {
                'gt_points': ['RightUpperLeg', 'RightLowerLeg', 'RightFoot'],
                'mp_points': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']
            },
            # 任意：肩・股関節
            'L_Shoulder': {
                'gt_points': ['Chest', 'LeftUpperArm', 'LeftLowerArm'],
                'mp_points': ['MID_SHOULDER', 'LEFT_SHOULDER', 'LEFT_ELBOW']
            },
            'R_Shoulder': {
                'gt_points': ['Chest', 'RightUpperArm', 'RightLowerArm'],
                'mp_points': ['MID_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_ELBOW']
            },
            'L_Hip': {
                'gt_points': ['Hips', 'LeftUpperLeg', 'LeftLowerLeg'],
                'mp_points': ['MID_HIP', 'LEFT_HIP', 'LEFT_KNEE']
            },
            'R_Hip': {
                'gt_points': ['Hips', 'RightUpperLeg', 'RightLowerLeg'],
                'mp_points': ['MID_HIP', 'RIGHT_HIP', 'RIGHT_KNEE']
            }
        }
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('coordinate_angle_comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_camera_coordinates(self, file_name: str) -> Tuple[float, float, float]:
        """
        CSVファイル名からカメラ座標を抽出
        
        Args:
            file_name: CapturedFrames_X_Y_Z形式のCSVファイル名（拡張子なし）
            
        Returns:
            Tuple[float, float, float]: (X, Y, Z)座標
        """
        # CapturedFrames_-1.0_1.0_-3.0 -> (-1.0, 1.0, -3.0)
        pattern = r'CapturedFrames_([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)'
        match = re.match(pattern, file_name)
        
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            return (x, y, z)
        else:
            self.logger.warning(f"座標抽出失敗: {file_name}")
            return (0.0, 0.0, 0.0)
    
    def calculate_3point_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        3点角を計算（度数法）
        
        Args:
            p1, p2, p3: 3D座標点（p2が頂点）
            
        Returns:
            float: 角度（度）
        """
        # ベクトル計算
        v1 = p1 - p2  # p2 -> p1
        v2 = p3 - p2  # p2 -> p3
        
        # 内積とノルム
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # ゼロ除算チェック
        if norm1 == 0 or norm2 == 0:
            return np.nan
        
        # 角度計算（ラジアン -> 度）
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数値安定性
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def load_mediapipe_data(self, csv_path: str) -> pd.DataFrame:
        """MediaPipe CSVを読み込み"""
        df = pd.read_csv(csv_path)
        
        # 中点計算
        df = self.calculate_midpoints(df)
        
        return df
    
    def calculate_midpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """MediaPipeデータに中点を追加"""
        # フレームごとに中点を計算
        midpoints_data = []
        
        for frame_id in df['frame_id'].unique():
            frame_data = df[df['frame_id'] == frame_id]
            
            # MID_SHOULDER = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2
            left_shoulder = frame_data[frame_data['landmark'] == 'LEFT_SHOULDER']
            right_shoulder = frame_data[frame_data['landmark'] == 'RIGHT_SHOULDER']
            
            if not left_shoulder.empty and not right_shoulder.empty:
                mid_shoulder = {
                    'frame_id': frame_id,
                    'landmark': 'MID_SHOULDER',
                    'x': (left_shoulder['x'].iloc[0] + right_shoulder['x'].iloc[0]) / 2,
                    'y': (left_shoulder['y'].iloc[0] + right_shoulder['y'].iloc[0]) / 2,
                    'z': (left_shoulder['z'].iloc[0] + right_shoulder['z'].iloc[0]) / 2,
                    'visibility': min(left_shoulder['visibility'].iloc[0], right_shoulder['visibility'].iloc[0]),
                    'image_path': left_shoulder['image_path'].iloc[0]
                }
                midpoints_data.append(mid_shoulder)
            
            # MID_HIP = (LEFT_HIP + RIGHT_HIP) / 2
            left_hip = frame_data[frame_data['landmark'] == 'LEFT_HIP']
            right_hip = frame_data[frame_data['landmark'] == 'RIGHT_HIP']
            
            if not left_hip.empty and not right_hip.empty:
                mid_hip = {
                    'frame_id': frame_id,
                    'landmark': 'MID_HIP',
                    'x': (left_hip['x'].iloc[0] + right_hip['x'].iloc[0]) / 2,
                    'y': (left_hip['y'].iloc[0] + right_hip['y'].iloc[0]) / 2,
                    'z': (left_hip['z'].iloc[0] + right_hip['z'].iloc[0]) / 2,
                    'visibility': min(left_hip['visibility'].iloc[0], right_hip['visibility'].iloc[0]),
                    'image_path': left_hip['image_path'].iloc[0]
                }
                midpoints_data.append(mid_hip)
        
        # 中点データを追加
        if midpoints_data:
            midpoints_df = pd.DataFrame(midpoints_data)
            df = pd.concat([df, midpoints_df], ignore_index=True)
        
        return df
    
    def load_ground_truth_data(self, csv_path: str) -> pd.DataFrame:
        """Ground Truth CSVを読み込み"""
        df = pd.read_csv(csv_path)
        
        # Frame列をframe_idにリネーム
        if 'Frame' in df.columns:
            df = df.rename(columns={'Frame': 'frame_id'})
        
        return df
    
    def get_point_coordinates(self, df: pd.DataFrame, frame_id: int, point_name: str, 
                            data_type: str, visibility_threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        指定フレーム・ポイントの座標を取得
        
        Args:
            df: データフレーム
            frame_id: フレームID
            point_name: ポイント名
            data_type: 'gt' or 'mp'
            visibility_threshold: 可視度閾値
            
        Returns:
            np.ndarray: 3D座標 or None
        """
        if data_type == 'mp':
            # MediaPipeデータ
            point_data = df[(df['frame_id'] == frame_id) & (df['landmark'] == point_name)]
            if point_data.empty:
                return None
            
            # 可視度チェック
            if point_data['visibility'].iloc[0] < visibility_threshold:
                return None
            
            return np.array([point_data['x'].iloc[0], point_data['y'].iloc[0], point_data['z'].iloc[0]])
        
        else:  # gt
            # Ground Truthデータ
            point_data = df[df['frame_id'] == frame_id]
            if point_data.empty:
                return None
            
            # 座標列を検索
            x_col = f"{point_name}_X"
            y_col = f"{point_name}_Y"
            z_col = f"{point_name}_Z"
            
            if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
                return None
            
            return np.array([point_data[x_col].iloc[0], point_data[y_col].iloc[0], point_data[z_col].iloc[0]])
    
    def compare_angles_for_coordinate(self, mp_csv_path: str, gt_csv_path: str, 
                                   joints: List[str] = None) -> Dict[str, float]:
        """
        特定座標での角度比較を実行
        
        Args:
            mp_csv_path: MediaPipe CSVパス
            gt_csv_path: Ground Truth CSVパス
            joints: 比較する関節リスト（Noneの場合は全関節）
            
        Returns:
            Dict[str, float]: 関節別MAE
        """
        # データ読み込み
        mp_df = self.load_mediapipe_data(mp_csv_path)
        gt_df = self.load_ground_truth_data(gt_csv_path)
        
        # 比較する関節を決定
        if joints is None:
            joints = list(self.joint_definitions.keys())
        
        # 結果格納
        joint_errors = {}
        
        # 共通フレームIDを取得
        common_frames = set(mp_df['frame_id'].unique()) & set(gt_df['frame_id'].unique())
        
        for joint in joints:
            if joint not in self.joint_definitions:
                continue
            
            joint_def = self.joint_definitions[joint]
            errors = []
            
            for frame_id in sorted(common_frames):
                # GT角度計算
                gt_points = []
                for point_name in joint_def['gt_points']:
                    point_coord = self.get_point_coordinates(gt_df, frame_id, point_name, 'gt')
                    if point_coord is None:
                        break
                    gt_points.append(point_coord)
                
                if len(gt_points) == 3:
                    gt_angle = self.calculate_3point_angle(gt_points[0], gt_points[1], gt_points[2])
                else:
                    gt_angle = np.nan
                
                # MP角度計算
                mp_points = []
                for point_name in joint_def['mp_points']:
                    point_coord = self.get_point_coordinates(mp_df, frame_id, point_name, 'mp')
                    if point_coord is None:
                        break
                    mp_points.append(point_coord)
                
                if len(mp_points) == 3:
                    mp_angle = self.calculate_3point_angle(mp_points[0], mp_points[1], mp_points[2])
                else:
                    mp_angle = np.nan
                
                # 誤差計算
                if not np.isnan(gt_angle) and not np.isnan(mp_angle):
                    abs_error = abs(gt_angle - mp_angle)
                    errors.append(abs_error)
            
            # MAE計算
            if errors:
                joint_errors[joint] = np.mean(errors)
            else:
                joint_errors[joint] = np.nan
        
        return joint_errors

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='カメラ座標別の角度比較')
    parser.add_argument('--mp_csv', help='MediaPipe CSVファイルパス（ワイルドカード使用可: *.csv）')
    parser.add_argument('--gt_csv', help='Ground Truth CSVファイルパス（synced_joint_positions.csvなど）')
    parser.add_argument('--output_csv', default='coordinate_angle_mae.csv', help='出力CSVファイル')
    parser.add_argument('--joints', nargs='+', help='比較する関節（指定しない場合は全関節）')
    
    args = parser.parse_args()
    
    # 角度比較器初期化
    comparator = CoordinateAngleComparator()
    
    # MediaPipe CSVファイルを検索
    if args.mp_csv:
        mp_csv_files = sorted(glob.glob(args.mp_csv))
        
        if not mp_csv_files:
            comparator.logger.error(f"MediaPipe CSVが見つかりません: {args.mp_csv}")
            return
        
        comparator.logger.info(f"発見されたMediaPipe CSV: {len(mp_csv_files)}個")
    else:
        comparator.logger.error("--mp_csv を指定してください")
        return
    
    # Ground Truth CSVファイルを確認
    if not args.gt_csv:
        comparator.logger.error("--gt_csv を指定してください（例: synced_joint_positions.csv）")
        return
    
    if not os.path.exists(args.gt_csv):
        comparator.logger.error(f"Ground Truth CSVが見つかりません: {args.gt_csv}")
        return
    
    gt_csv_path = args.gt_csv
    comparator.logger.info(f"Ground Truth CSV: {gt_csv_path}")
    
    # 結果格納
    results = []
    
    for mp_csv_path in mp_csv_files:
        # ファイル名から座標を抽出
        file_name = Path(mp_csv_path).stem
        camera_x, camera_y, camera_z = comparator.extract_camera_coordinates(file_name)
        
        # 角度比較実行（すべてのMediaPipe CSVに対して同じGT CSVを使用）
        try:
            joint_errors = comparator.compare_angles_for_coordinate(mp_csv_path, gt_csv_path, args.joints)
            
            # 結果を追加
            result = {
                'folder_name': file_name,
                'camera_x': camera_x,
                'camera_y': camera_y,
                'camera_z': camera_z,
                **joint_errors
            }
            results.append(result)
            
            comparator.logger.info(f"完了: {file_name} ({camera_x}, {camera_y}, {camera_z})")
            
        except Exception as e:
            comparator.logger.error(f"エラー {file_name}: {str(e)}")
    
    # 結果をDataFrameに変換
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        
        comparator.logger.info(f"全処理完了: {len(results)}座標")
        comparator.logger.info(f"結果保存: {args.output_csv}")
        
        # 統計表示
        print("\n=== カメラ座標別角度MAE結果 ===")
        print(f"処理座標数: {len(results_df)}")
        
        # 各関節の全体統計
        joint_columns = [col for col in results_df.columns if col not in ['folder_name', 'camera_x', 'camera_y', 'camera_z']]
        for joint in joint_columns:
            valid_data = results_df[joint].dropna()
            if len(valid_data) > 0:
                print(f"{joint}: MAE={valid_data.mean():.2f}° (範囲: {valid_data.min():.2f}°-{valid_data.max():.2f}°)")
        
        # 座標範囲
        print(f"\nカメラ座標範囲:")
        print(f"X: {results_df['camera_x'].min():.1f} ～ {results_df['camera_x'].max():.1f}")
        print(f"Y: {results_df['camera_y'].min():.1f} ～ {results_df['camera_y'].max():.1f}")
        print(f"Z: {results_df['camera_z'].min():.1f} ～ {results_df['camera_z'].max():.1f}")
    else:
        comparator.logger.error("処理結果がありません")

if __name__ == "__main__":
    main()


