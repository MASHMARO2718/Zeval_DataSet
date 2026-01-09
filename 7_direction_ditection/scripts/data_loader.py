"""
データ読み込みモジュール
GroundTruthとMediaPipeデータの読み込みを管理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import config
from scripts.logger import get_logger


class DataLoader:
    """GroundTruthとMediaPipeデータの読み込みクラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = get_logger("DataLoader")
        self.gt_csv = config.GT_CSV
        self.mp_dir = config.MP_DIR
        self.joint_mapping = config.JOINT_MAPPING
        
        self.logger.info("DataLoader initialized")
        self.logger.debug(f"GroundTruth CSV: {self.gt_csv}")
        self.logger.debug(f"MediaPipe Dir: {self.mp_dir}")
    
    def load_ground_truth(self, frame_id: Optional[int] = None) -> pd.DataFrame:
        """
        GroundTruthデータを読み込む
        
        Args:
            frame_id: 特定のフレームID（Noneの場合は全フレーム）
            
        Returns:
            pd.DataFrame: GroundTruthデータ
        """
        self.logger.step(1, "Loading GroundTruth data")
        
        if not self.gt_csv.exists():
            self.logger.error(f"GroundTruth CSV not found: {self.gt_csv}")
            raise FileNotFoundError(f"GroundTruth CSV not found: {self.gt_csv}")
        
        df = pd.read_csv(self.gt_csv)
        self.logger.info(f"GroundTruth loaded: {len(df)} rows")
        self.logger.debug(f"Columns: {df.columns.tolist()[:10]}...")
        
        if frame_id is not None:
            df = df[df['Frame'] == frame_id]
            self.logger.info(f"Filtered to frame {frame_id}: {len(df)} rows")
        
        # デバッグモードでサンプルデータを保存
        if config.SAVE_INTERMEDIATE_DATA:
            debug_file = config.DEBUG_DATA_DIR / "ground_truth_sample.csv"
            df.head(10).to_csv(debug_file, index=False)
            self.logger.debug(f"Sample data saved: {debug_file}")
        
        return df
    
    def load_mediapipe(self, camera_position: str, 
                       y_range: str = None) -> pd.DataFrame:
        """
        MediaPipe処理済みデータを読み込む
        
        Args:
            camera_position: 例: "CapturedFrames_-1.0_0.5_-3.0"
            y_range: 例: "Y=0.5,1.5" (Noneの場合はデフォルト)
            
        Returns:
            pd.DataFrame: MediaPipeデータ
        """
        if y_range is None:
            y_range = config.DEFAULT_Y_RANGE
        
        self.logger.step(1, f"Loading MediaPipe data: {camera_position}")
        
        mp_dir = self.mp_dir / y_range
        csv_file = mp_dir / f"{camera_position}.csv"
        
        if not csv_file.exists():
            self.logger.error(f"MediaPipe CSV not found: {csv_file}")
            raise FileNotFoundError(f"MediaPipe CSV not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        self.logger.info(f"MediaPipe loaded: {len(df)} rows from {csv_file.name}")
        self.logger.debug(f"Columns: {df.columns.tolist()}")
        
        unique_frames = sorted(df['frame_id'].unique())
        self.logger.info(f"Unique frames: {len(unique_frames)} frames")
        self.logger.debug(f"Frame IDs: {unique_frames[:10]}...")
        
        # デバッグモードでサンプルデータを保存
        if config.SAVE_INTERMEDIATE_DATA:
            debug_file = config.DEBUG_DATA_DIR / f"mediapipe_{camera_position}_sample.csv"
            df.head(50).to_csv(debug_file, index=False)
            self.logger.debug(f"Sample data saved: {debug_file}")
        
        return df
    
    def get_frame_coordinates(self, df: pd.DataFrame, 
                            frame_id: int,
                            is_mediapipe: bool = False) -> Dict[str, np.ndarray]:
        """
        特定フレームの関節座標を抽出
        
        Args:
            df: データフレーム
            frame_id: フレームID
            is_mediapipe: MediaPipeデータかどうか
            
        Returns:
            Dict[joint_name, np.array([x, y, z])]
        """
        self.logger.debug(f"Extracting coordinates for frame {frame_id} (MediaPipe: {is_mediapipe})")
        
        if is_mediapipe:
            coords = self._extract_mediapipe_coords(df, frame_id)
        else:
            coords = self._extract_ground_truth_coords(df, frame_id)
        
        self.logger.info(f"Extracted {len(coords)} joints for frame {frame_id}")
        
        # デバッグモードで座標を保存
        if config.SAVE_INTERMEDIATE_DATA:
            data_type = "mediapipe" if is_mediapipe else "ground_truth"
            debug_file = config.DEBUG_DATA_DIR / f"{data_type}_frame_{frame_id}_coords.txt"
            with open(debug_file, 'w') as f:
                for joint, coord in coords.items():
                    f.write(f"{joint}: {coord}\n")
            self.logger.debug(f"Coordinates saved: {debug_file}")
        
        return coords
    
    def _extract_mediapipe_coords(self, df: pd.DataFrame, 
                                 frame_id: int) -> Dict[str, np.ndarray]:
        """MediaPipe座標を抽出"""
        frame_data = df[df['frame_id'] == frame_id]
        
        if len(frame_data) == 0:
            self.logger.warning(f"No data found for frame {frame_id}")
            return {}
        
        coords = {}
        for joint_name in self.joint_mapping.keys():
            joint_data = frame_data[frame_data['landmark'] == joint_name]
            if not joint_data.empty:
                coords[joint_name] = np.array([
                    joint_data['x'].values[0],
                    joint_data['y'].values[0],
                    joint_data['z'].values[0]
                ], dtype=np.float64)
        
        return coords
    
    def _extract_ground_truth_coords(self, df: pd.DataFrame, 
                                    frame_id: int) -> Dict[str, np.ndarray]:
        """GroundTruth座標を抽出"""
        frame_data = df[df['Frame'] == frame_id]
        
        if len(frame_data) == 0:
            self.logger.warning(f"No data found for frame {frame_id}")
            return {}
        
        coords = {}
        
        # カラム名のパターンを確認
        sample_cols = df.columns.tolist()
        self.logger.debug(f"Available columns (first 20): {sample_cols[:20]}")
        
        # GroundTruthのカラム名マッピング（実際のデータ構造に基づく）
        gt_to_mediapipe_mapping = {
            'Hips': ['LEFT_HIP', 'RIGHT_HIP'],  # Hipsは両方の腰の中点として使用
            'LeftShoulder': 'LEFT_SHOULDER',
            'RightShoulder': 'RIGHT_SHOULDER',
            'LeftLowerArm': 'LEFT_ELBOW',  # LeftLowerArmは肘の位置
            'RightLowerArm': 'RIGHT_ELBOW',
            'LeftLowerLeg': 'LEFT_KNEE',  # LeftLowerLegは膝の位置
            'RightLowerLeg': 'RIGHT_KNEE',
            'LeftFoot': 'LEFT_ANKLE',
            'RightFoot': 'RIGHT_ANKLE',
            'LeftHand': 'LEFT_WRIST',
            'RightHand': 'RIGHT_WRIST',
        }
        
        # カラム名から座標を抽出
        for gt_name, mp_name in gt_to_mediapipe_mapping.items():
            x_col = f"{gt_name}_X"
            y_col = f"{gt_name}_Y"
            z_col = f"{gt_name}_Z"
            
            if x_col in frame_data.columns:
                try:
                    coord = np.array([
                        float(frame_data[x_col].values[0]),
                        float(frame_data[y_col].values[0]),
                        float(frame_data[z_col].values[0])
                    ], dtype=np.float64)
                    
                    # Hipsは特別処理（左右両方の腰として使用）
                    if isinstance(mp_name, list):
                        for name in mp_name:
                            coords[name] = coord.copy()
                    else:
                        coords[mp_name] = coord
                        
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to extract {gt_name}: {e}")
        
        self.logger.debug(f"Extracted GT joints: {list(coords.keys())}")
        
        return coords
    
    def list_available_cameras(self, y_range: str = None) -> List[str]:
        """
        利用可能なカメラ位置をリスト
        
        Args:
            y_range: Y範囲（Noneの場合はデフォルト）
            
        Returns:
            List[str]: カメラ位置のリスト
        """
        if y_range is None:
            y_range = config.DEFAULT_Y_RANGE
        
        self.logger.info(f"Listing available cameras for {y_range}")
        
        mp_dir = self.mp_dir / y_range
        
        if not mp_dir.exists():
            self.logger.error(f"MediaPipe directory not found: {mp_dir}")
            return []
        
        csv_files = list(mp_dir.glob("*.csv"))
        camera_positions = [f.stem for f in csv_files]
        
        self.logger.info(f"Found {len(camera_positions)} camera positions")
        self.logger.debug(f"First 5: {camera_positions[:5]}")
        
        return sorted(camera_positions)
    
    def get_frame_range(self, df: pd.DataFrame, is_mediapipe: bool = False) -> tuple:
        """
        フレーム範囲を取得
        
        Args:
            df: データフレーム
            is_mediapipe: MediaPipeデータかどうか
            
        Returns:
            (min_frame, max_frame)
        """
        if is_mediapipe:
            frames = df['frame_id'].unique()
        else:
            frames = df['Frame'].unique()
        
        min_frame = int(frames.min())
        max_frame = int(frames.max())
        
        self.logger.info(f"Frame range: {min_frame} to {max_frame} ({len(frames)} frames)")
        
        return min_frame, max_frame


if __name__ == "__main__":
    # テスト実行
    print("=== DataLoader Test ===\n")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    # データローダー初期化
    loader = DataLoader()
    
    # GroundTruth読み込みテスト
    try:
        gt_df = loader.load_ground_truth()
        print(f"✅ GroundTruth loaded: {gt_df.shape}")
        
        # フレーム範囲確認
        min_frame, max_frame = loader.get_frame_range(gt_df, is_mediapipe=False)
        print(f"✅ Frame range: {min_frame} - {max_frame}")
        
    except Exception as e:
        print(f"❌ GroundTruth loading failed: {e}")
    
    # MediaPipe読み込みテスト
    try:
        cameras = loader.list_available_cameras()
        if cameras:
            print(f"✅ Available cameras: {len(cameras)}")
            
            # 最初のカメラのデータを読み込み
            mp_df = loader.load_mediapipe(cameras[0])
            print(f"✅ MediaPipe loaded: {mp_df.shape}")
            
            # フレーム範囲確認
            min_frame, max_frame = loader.get_frame_range(mp_df, is_mediapipe=True)
            print(f"✅ Frame range: {min_frame} - {max_frame}")
            
            # 1フレームの座標取得
            frame_id = min_frame
            gt_coords = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
            mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
            
            print(f"✅ GT joints: {len(gt_coords)}")
            print(f"✅ MP joints: {len(mp_coords)}")
            
            # サンプル表示
            if 'LEFT_SHOULDER' in gt_coords:
                print(f"\nSample - LEFT_SHOULDER (GT): {gt_coords['LEFT_SHOULDER']}")
            if 'LEFT_SHOULDER' in mp_coords:
                print(f"Sample - LEFT_SHOULDER (MP): {mp_coords['LEFT_SHOULDER']}")
        else:
            print("❌ No cameras found")
            
    except Exception as e:
        print(f"❌ MediaPipe loading failed: {e}")
    
    print(f"\n✅ Test completed. Check logs: {config.LOG_DIR / 'latest.log'}")

