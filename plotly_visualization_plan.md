# Plotlyを使った座標変換検証プラン

## 📋 プロジェクト概要

GroundTruthデータとMediaPipe出力を座標変換し、Plotlyでインタラクティブに可視化・検証する。

---

## 🎯 実装の目標

1. GroundTruth座標系からMediaPipe座標系への変換を実装
2. 1フレームでの座標変換を視覚的に検証
3. Plotlyで回転・ズーム可能な3D表示を実現
4. 複数フレームでの比較機能
5. HTMLレポートの自動生成

---

## 📦 必要なライブラリ

```bash
# インストールコマンド
pip install plotly pandas numpy matplotlib
```

### ライブラリバージョン（推奨）
- plotly >= 5.0.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0 (補助用)

---

## 📁 ファイル構成（新規作成）

```
Zeval_DataSet/
├── coordinate_transform_plan.md          # 既存の計画書
├── plotly_visualization_plan.md          # この計画書
│
├── scripts/                              # 新規作成
│   ├── __init__.py
│   ├── coordinate_transform.py           # 座標変換ロジック
│   ├── data_loader.py                    # データ読み込み
│   ├── plotly_visualizer.py              # Plotly可視化
│   ├── validation.py                     # 検証ユーティリティ
│   └── config.py                         # 設定ファイル
│
├── notebooks/                            # 新規作成
│   ├── 01_data_exploration.ipynb         # データ構造確認
│   ├── 02_single_frame_test.ipynb        # 1フレームテスト
│   ├── 03_multi_frame_validation.ipynb   # 複数フレーム検証
│   └── 04_interactive_report.ipynb       # インタラクティブレポート
│
└── output/                               # 新規作成
    ├── html_reports/                     # Plotly HTMLレポート
    ├── validation_results/               # 検証結果CSV
    └── screenshots/                      # スクリーンショット
```

---

## 🚀 実装フェーズ

### **Phase 1: データ読み込みと構造確認（Day 1）**

#### 目的
- GroundTruthとMediaPipeデータの構造を理解
- 1フレーム分のデータを抽出

#### 実装内容

**1.1 データローダーの実装**

```python
# scripts/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

class DataLoader:
    """GroundTruthとMediaPipeデータの読み込みクラス"""
    
    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Zeval_DataSetのルートディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.gt_csv = self.base_dir / "synced_joint_positions.csv"
        
        # MediaPipe関節名マッピング（MediaPipeのインデックス）
        self.joint_mapping = {
            'NOSE': 0,
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15,
            'RIGHT_WRIST': 16,
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28,
        }
        
    def load_ground_truth(self, frame_id: Optional[int] = None) -> pd.DataFrame:
        """
        GroundTruthデータを読み込む
        
        Args:
            frame_id: 特定のフレームID（Noneの場合は全フレーム）
            
        Returns:
            pd.DataFrame: GroundTruthデータ
        """
        df = pd.read_csv(self.gt_csv)
        
        if frame_id is not None:
            df = df[df['Frame'] == frame_id]
        
        print(f"GroundTruth loaded: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()[:10]}...")  # 最初の10列を表示
        
        return df
    
    def load_mediapipe(self, camera_position: str, 
                       y_range: str = "Y=0.5,1.5") -> pd.DataFrame:
        """
        MediaPipe処理済みデータを読み込む
        
        Args:
            camera_position: 例: "CapturedFrames_-1.0_0.5_-3.0"
            y_range: 例: "Y=0.5,1.5" or "Y=1.0.2.0"
            
        Returns:
            pd.DataFrame: MediaPipeデータ
        """
        mp_dir = self.base_dir / "2_medidapipe_proccesed" / y_range
        csv_file = mp_dir / f"{camera_position}.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"MediaPipe CSV not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        print(f"MediaPipe loaded: {len(df)} rows from {csv_file.name}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Unique frames: {sorted(df['frame_id'].unique())}")
        
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
        if is_mediapipe:
            frame_data = df[df['frame_id'] == frame_id]
            coords = {}
            
            for joint_name in self.joint_mapping.keys():
                joint_data = frame_data[frame_data['landmark'] == joint_name]
                if not joint_data.empty:
                    coords[joint_name] = np.array([
                        joint_data['x'].values[0],
                        joint_data['y'].values[0],
                        joint_data['z'].values[0]
                    ])
            
            return coords
        else:
            # GroundTruthの場合（カラム名に応じて調整必要）
            frame_data = df[df['Frame'] == frame_id]
            coords = {}
            
            # GroundTruthのカラム構造に応じて実装
            # 例: 'LEFT_SHOULDER_X', 'LEFT_SHOULDER_Y', 'LEFT_SHOULDER_Z'
            for joint_name in self.joint_mapping.keys():
                x_col = f"{joint_name}_X"
                y_col = f"{joint_name}_Y"
                z_col = f"{joint_name}_Z"
                
                if x_col in frame_data.columns:
                    coords[joint_name] = np.array([
                        frame_data[x_col].values[0],
                        frame_data[y_col].values[0],
                        frame_data[z_col].values[0]
                    ])
            
            return coords
    
    def list_available_cameras(self, y_range: str = "Y=0.5,1.5") -> List[str]:
        """
        利用可能なカメラ位置をリスト
        
        Args:
            y_range: Y範囲
            
        Returns:
            List[str]: カメラ位置のリスト
        """
        mp_dir = self.base_dir / "2_medidapipe_proccesed" / y_range
        csv_files = list(mp_dir.glob("*.csv"))
        
        camera_positions = [f.stem for f in csv_files]
        print(f"Found {len(camera_positions)} camera positions")
        
        return sorted(camera_positions)
```

**1.2 データ構造確認スクリプト**

```python
# notebooks/01_data_exploration.ipynb の内容

from scripts.data_loader import DataLoader
import pandas as pd

# データローダー初期化
loader = DataLoader(".")

# GroundTruthの構造確認
print("=== GroundTruth Data Structure ===")
gt_df = loader.load_ground_truth()
print(gt_df.head())
print(f"\nShape: {gt_df.shape}")
print(f"Columns: {gt_df.columns.tolist()}")

# MediaPipeの構造確認
print("\n=== MediaPipe Data Structure ===")
cameras = loader.list_available_cameras()
print(f"Available cameras (first 5): {cameras[:5]}")

# 1つのカメラのデータを確認
mp_df = loader.load_mediapipe(cameras[0])
print(mp_df.head(10))

# 特定フレームの座標取得テスト
print("\n=== Single Frame Test ===")
frame_id = 0
gt_coords = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)

print(f"GroundTruth joints: {list(gt_coords.keys())}")
print(f"MediaPipe joints: {list(mp_coords.keys())}")
print(f"\nLEFT_SHOULDER (GT): {gt_coords.get('LEFT_SHOULDER')}")
print(f"LEFT_SHOULDER (MP): {mp_coords.get('LEFT_SHOULDER')}")
```

---

### **Phase 2: 座標変換の実装（Day 2-3）**

#### 目的
- 右手系→左手系の変換
- 腰を原点とした相対座標化
- 変換の正しさを検証

#### 実装内容

**2.1 座標変換クラス**

```python
# scripts/coordinate_transform.py

import numpy as np
from typing import Dict, Tuple

class CoordinateTransformer:
    """座標系変換クラス"""
    
    def __init__(self):
        """初期化"""
        self.hip_joints = ['LEFT_HIP', 'RIGHT_HIP']
    
    def right_to_left_hand(self, coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        右手座標系から左手座標系への変換
        
        Args:
            coords: {joint_name: np.array([x, y, z])}
            
        Returns:
            変換後の座標
        """
        transformed = {}
        
        for joint_name, coord in coords.items():
            x_gt, y_gt, z_gt = coord
            
            # Step 1: 座標系変換
            x_lh = x_gt
            y_lh = -y_gt  # Y軸反転（重要！）
            z_lh = z_gt
            
            transformed[joint_name] = np.array([x_lh, y_lh, z_lh])
        
        return transformed
    
    def calculate_hip_center(self, coords: Dict[str, np.ndarray]) -> np.ndarray:
        """
        腰の中心座標を計算
        
        Args:
            coords: 座標辞書
            
        Returns:
            np.array([hip_x, hip_y, hip_z])
        """
        left_hip = coords.get('LEFT_HIP')
        right_hip = coords.get('RIGHT_HIP')
        
        if left_hip is None or right_hip is None:
            raise ValueError("Hip joints not found in coordinates")
        
        hip_center = (left_hip + right_hip) / 2.0
        
        return hip_center
    
    def to_relative_coordinates(self, coords: Dict[str, np.ndarray],
                               hip_center: np.ndarray) -> Dict[str, np.ndarray]:
        """
        腰を原点とした相対座標に変換
        
        Args:
            coords: 座標辞書
            hip_center: 腰の中心座標
            
        Returns:
            相対座標辞書
        """
        relative = {}
        
        for joint_name, coord in coords.items():
            relative[joint_name] = coord - hip_center
        
        return relative
    
    def transform_ground_truth(self, gt_coords: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        GroundTruth座標を完全に変換
        
        Args:
            gt_coords: GroundTruth座標（右手系）
            
        Returns:
            (相対座標, 腰の中心座標)
        """
        # Step 1: 右手系→左手系
        lh_coords = self.right_to_left_hand(gt_coords)
        
        # Step 2: 腰の中心を計算
        hip_center = self.calculate_hip_center(lh_coords)
        
        # Step 3: 相対座標化
        relative_coords = self.to_relative_coordinates(lh_coords, hip_center)
        
        return relative_coords, hip_center
    
    def calculate_angle_xy(self, coord: np.ndarray) -> float:
        """
        XY平面での角度θを計算
        
        Args:
            coord: np.array([x, y, z])
            
        Returns:
            角度（ラジアン）
        """
        return np.arctan2(coord[1], coord[0])
    
    def calculate_angle_xz(self, coord: np.ndarray) -> float:
        """
        XZ平面での角度ψを計算
        
        Args:
            coord: np.array([x, y, z])
            
        Returns:
            角度（ラジアン）
        """
        return np.arctan2(coord[2], coord[0])
    
    def normalize_angle(self, angle: float) -> float:
        """
        角度を-πからπの範囲に正規化
        
        Args:
            angle: 角度（ラジアン）
            
        Returns:
            正規化された角度
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def calculate_differences(self, gt_coords: Dict[str, np.ndarray],
                            mp_coords: Dict[str, np.ndarray]) -> Dict:
        """
        GroundTruthとMediaPipeの差分を計算
        
        Args:
            gt_coords: GroundTruth相対座標
            mp_coords: MediaPipe座標
            
        Returns:
            差分情報の辞書
        """
        differences = {}
        
        for joint_name in gt_coords.keys():
            if joint_name not in mp_coords:
                continue
            
            gt = gt_coords[joint_name]
            mp = mp_coords[joint_name]
            
            # 3D誤差
            error_3d = np.linalg.norm(mp - gt)
            
            # XY平面の角度
            theta_gt = self.calculate_angle_xy(gt)
            theta_mp = self.calculate_angle_xy(mp)
            delta_theta = self.normalize_angle(theta_mp - theta_gt)
            
            # XZ平面の角度
            psi_gt = self.calculate_angle_xz(gt)
            psi_mp = self.calculate_angle_xz(mp)
            delta_psi = self.normalize_angle(psi_mp - psi_gt)
            
            differences[joint_name] = {
                'gt_coord': gt,
                'mp_coord': mp,
                'delta_xyz': mp - gt,
                'error_3d': error_3d,
                'theta_gt': theta_gt,
                'theta_mp': theta_mp,
                'delta_theta': delta_theta,
                'psi_gt': psi_gt,
                'psi_mp': psi_mp,
                'delta_psi': delta_psi
            }
        
        return differences
```

**2.2 検証ユーティリティ**

```python
# scripts/validation.py

import numpy as np
from typing import Dict

class Validator:
    """座標変換の検証クラス"""
    
    @staticmethod
    def check_hip_at_origin(coords: Dict[str, np.ndarray], 
                          tolerance: float = 1e-6) -> bool:
        """
        腰が原点にあるか確認
        
        Args:
            coords: 相対座標辞書
            tolerance: 許容誤差
            
        Returns:
            bool: 原点にある場合True
        """
        if 'LEFT_HIP' not in coords or 'RIGHT_HIP' not in coords:
            return False
        
        left_hip = coords['LEFT_HIP']
        right_hip = coords['RIGHT_HIP']
        
        hip_center = (left_hip + right_hip) / 2.0
        distance_from_origin = np.linalg.norm(hip_center)
        
        is_at_origin = distance_from_origin < tolerance
        
        print(f"Hip center: {hip_center}")
        print(f"Distance from origin: {distance_from_origin:.10f}")
        print(f"At origin: {is_at_origin}")
        
        return is_at_origin
    
    @staticmethod
    def check_coordinate_ranges(coords: Dict[str, np.ndarray]) -> Dict:
        """
        座標の範囲を確認
        
        Args:
            coords: 座標辞書
            
        Returns:
            統計情報
        """
        all_coords = np.array(list(coords.values()))
        
        stats = {
            'x_range': (all_coords[:, 0].min(), all_coords[:, 0].max()),
            'y_range': (all_coords[:, 1].min(), all_coords[:, 1].max()),
            'z_range': (all_coords[:, 2].min(), all_coords[:, 2].max()),
            'x_mean': all_coords[:, 0].mean(),
            'y_mean': all_coords[:, 1].mean(),
            'z_mean': all_coords[:, 2].mean(),
        }
        
        print("=== Coordinate Ranges ===")
        print(f"X: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}] (mean: {stats['x_mean']:.3f})")
        print(f"Y: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}] (mean: {stats['y_mean']:.3f})")
        print(f"Z: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}] (mean: {stats['z_mean']:.3f})")
        
        return stats
    
    @staticmethod
    def check_symmetry(coords: Dict[str, np.ndarray]) -> Dict:
        """
        左右の対称性を確認
        
        Args:
            coords: 座標辞書
            
        Returns:
            対称性情報
        """
        pairs = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_ELBOW', 'RIGHT_ELBOW'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_KNEE', 'RIGHT_KNEE'),
        ]
        
        symmetry_info = {}
        
        print("=== Symmetry Check ===")
        for left, right in pairs:
            if left in coords and right in coords:
                left_coord = coords[left]
                right_coord = coords[right]
                
                # X座標の符号が逆であることを期待
                x_symmetry = abs(left_coord[0] + right_coord[0])
                
                # Y, Z座標はほぼ同じであることを期待
                y_diff = abs(left_coord[1] - right_coord[1])
                z_diff = abs(left_coord[2] - right_coord[2])
                
                symmetry_info[left] = {
                    'x_symmetry': x_symmetry,
                    'y_diff': y_diff,
                    'z_diff': z_diff
                }
                
                print(f"{left} <-> {right}:")
                print(f"  X symmetry (should be ~0): {x_symmetry:.3f}")
                print(f"  Y difference: {y_diff:.3f}")
                print(f"  Z difference: {z_diff:.3f}")
        
        return symmetry_info
```

---

### **Phase 3: Plotly可視化の実装（Day 4-5）**

#### 目的
- インタラクティブな3D可視化
- GroundTruth vs MediaPipe比較
- HTMLレポート生成

#### 実装内容

**3.1 Plotly可視化クラス**

```python
# scripts/plotly_visualizer.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple

class PlotlyVisualizer:
    """Plotlyによるインタラクティブ可視化クラス"""
    
    def __init__(self):
        """初期化"""
        # MediaPipeの骨格接続定義
        self.skeleton_connections = [
            # 胴体
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            
            # 左腕
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            
            # 右腕
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            
            # 左脚
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            
            # 右脚
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        ]
    
    def create_skeleton_traces(self, coords: Dict[str, np.ndarray],
                              color: str, name: str,
                              show_legend: bool = True) -> List[go.Scatter3d]:
        """
        骨格構造のトレースを作成
        
        Args:
            coords: 座標辞書
            color: 色
            name: 名前
            show_legend: 凡例表示
            
        Returns:
            トレースのリスト
        """
        traces = []
        
        # 骨格の線
        for i, (start_joint, end_joint) in enumerate(self.skeleton_connections):
            if start_joint in coords and end_joint in coords:
                start = coords[start_joint]
                end = coords[end_joint]
                
                trace = go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(color=color, width=6),
                    showlegend=False,
                    hoverinfo='skip'
                )
                traces.append(trace)
        
        # 関節点
        joint_names = list(coords.keys())
        coords_array = np.array([coords[j] for j in joint_names])
        
        joint_trace = go.Scatter3d(
            x=coords_array[:, 0],
            y=coords_array[:, 1],
            z=coords_array[:, 2],
            mode='markers+text',
            marker=dict(size=8, color=color, opacity=0.8),
            text=joint_names,
            textposition="top center",
            textfont=dict(size=10),
            name=name,
            showlegend=show_legend,
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.4f}<br>' +
                         'Y: %{y:.4f}<br>' +
                         'Z: %{z:.4f}<br>' +
                         '<extra></extra>'
        )
        traces.append(joint_trace)
        
        return traces
    
    def plot_side_by_side(self, gt_coords: Dict[str, np.ndarray],
                         mp_coords: Dict[str, np.ndarray],
                         frame_id: int,
                         title: str = "Coordinate Comparison") -> go.Figure:
        """
        左右並べて比較プロット
        
        Args:
            gt_coords: GroundTruth座標
            mp_coords: MediaPipe座標
            frame_id: フレームID
            title: タイトル
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('GroundTruth (Hip-Centered)', 'MediaPipe'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.05
        )
        
        # GroundTruthトレース
        gt_traces = self.create_skeleton_traces(gt_coords, 'blue', 'GroundTruth')
        for trace in gt_traces:
            fig.add_trace(trace, row=1, col=1)
        
        # MediaPipeトレース
        mp_traces = self.create_skeleton_traces(mp_coords, 'red', 'MediaPipe')
        for trace in mp_traces:
            fig.add_trace(trace, row=1, col=2)
        
        # 原点マーカー（腰の位置）
        origin_trace = go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=15, color='black', symbol='diamond'),
            name='Hip Center (Origin)',
            showlegend=True,
            hovertemplate='Origin (0, 0, 0)<extra></extra>'
        )
        fig.add_trace(origin_trace, row=1, col=1)
        fig.add_trace(origin_trace, row=1, col=2)
        
        # レイアウト設定
        fig.update_layout(
            title=f'{title} - Frame {frame_id}',
            height=800,
            showlegend=True,
            legend=dict(x=0.85, y=0.95),
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            scene2=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            )
        )
        
        return fig
    
    def plot_overlay(self, gt_coords: Dict[str, np.ndarray],
                    mp_coords: Dict[str, np.ndarray],
                    frame_id: int,
                    differences: Dict = None) -> go.Figure:
        """
        オーバーレイ表示（両方を重ねて表示）
        
        Args:
            gt_coords: GroundTruth座標
            mp_coords: MediaPipe座標
            frame_id: フレームID
            differences: 差分情報
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # GroundTruthトレース
        gt_traces = self.create_skeleton_traces(gt_coords, 'blue', 'GroundTruth', True)
        for trace in gt_traces:
            fig.add_trace(trace)
        
        # MediaPipeトレース
        mp_traces = self.create_skeleton_traces(mp_coords, 'red', 'MediaPipe', True)
        for trace in mp_traces:
            fig.add_trace(trace)
        
        # 差分ベクトル（矢印）
        if differences:
            for joint_name, diff_info in differences.items():
                if joint_name in gt_coords and joint_name in mp_coords:
                    gt = diff_info['gt_coord']
                    mp = diff_info['mp_coord']
                    error = diff_info['error_3d']
                    
                    # 矢印（差分ベクトル）
                    arrow_trace = go.Scatter3d(
                        x=[gt[0], mp[0]],
                        y=[gt[1], mp[1]],
                        z=[gt[2], mp[2]],
                        mode='lines',
                        line=dict(color='green', width=3, dash='dash'),
                        showlegend=False,
                        hovertemplate=f'<b>{joint_name}</b><br>Error: {error:.4f}<extra></extra>'
                    )
                    fig.add_trace(arrow_trace)
        
        # 原点マーカー
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=15, color='black', symbol='diamond'),
            name='Hip Center (Origin)',
            hovertemplate='Origin (0, 0, 0)<extra></extra>'
        ))
        
        # レイアウト
        fig.update_layout(
            title=f'Overlay Comparison - Frame {frame_id}',
            height=800,
            showlegend=True,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            )
        )
        
        return fig
    
    def plot_multi_view(self, gt_coords: Dict[str, np.ndarray],
                       mp_coords: Dict[str, np.ndarray],
                       frame_id: int) -> go.Figure:
        """
        多視点プロット（XY, XZ, YZ平面）
        
        Args:
            gt_coords: GroundTruth座標
            mp_coords: MediaPipe座標
            frame_id: フレームID
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'GT: XY Plane (Front)', 'GT: XZ Plane (Top)', 'GT: YZ Plane (Side)',
                'MP: XY Plane (Front)', 'MP: XZ Plane (Top)', 'MP: YZ Plane (Side)'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
            horizontal_spacing=0.08,
            vertical_spacing=0.1
        )
        
        views = [
            (0, 1, 'X', 'Y'),  # XY平面
            (0, 2, 'X', 'Z'),  # XZ平面
            (1, 2, 'Y', 'Z'),  # YZ平面
        ]
        
        # GroundTruth（上段）
        for col, (idx1, idx2, label1, label2) in enumerate(views, 1):
            joint_names = list(gt_coords.keys())
            coords_array = np.array([gt_coords[j] for j in joint_names])
            
            fig.add_trace(
                go.Scatter(
                    x=coords_array[:, idx1],
                    y=coords_array[:, idx2],
                    mode='markers+text',
                    marker=dict(size=10, color='blue'),
                    text=joint_names,
                    textposition='top center',
                    textfont=dict(size=8),
                    name='GT',
                    showlegend=(col == 1),
                    hovertemplate=f'<b>%{{text}}</b><br>{label1}: %{{x:.3f}}<br>{label2}: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=col
            )
            
            # 骨格線
            for start_joint, end_joint in self.skeleton_connections:
                if start_joint in gt_coords and end_joint in gt_coords:
                    start = gt_coords[start_joint]
                    end = gt_coords[end_joint]
                    fig.add_trace(
                        go.Scatter(
                            x=[start[idx1], end[idx1]],
                            y=[start[idx2], end[idx2]],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1, col=col
                    )
            
            fig.update_xaxes(title_text=label1, row=1, col=col)
            fig.update_yaxes(title_text=label2, row=1, col=col)
        
        # MediaPipe（下段）
        for col, (idx1, idx2, label1, label2) in enumerate(views, 1):
            joint_names = list(mp_coords.keys())
            coords_array = np.array([mp_coords[j] for j in joint_names])
            
            fig.add_trace(
                go.Scatter(
                    x=coords_array[:, idx1],
                    y=coords_array[:, idx2],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    text=joint_names,
                    textposition='top center',
                    textfont=dict(size=8),
                    name='MP',
                    showlegend=(col == 1),
                    hovertemplate=f'<b>%{{text}}</b><br>{label1}: %{{x:.3f}}<br>{label2}: %{{y:.3f}}<extra></extra>'
                ),
                row=2, col=col
            )
            
            # 骨格線
            for start_joint, end_joint in self.skeleton_connections:
                if start_joint in mp_coords and end_joint in mp_coords:
                    start = mp_coords[start_joint]
                    end = mp_coords[end_joint]
                    fig.add_trace(
                        go.Scatter(
                            x=[start[idx1], end[idx1]],
                            y=[start[idx2], end[idx2]],
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=2, col=col
                    )
            
            fig.update_xaxes(title_text=label1, row=2, col=col)
            fig.update_yaxes(title_text=label2, row=2, col=col)
        
        fig.update_layout(
            title=f'Multi-View Comparison - Frame {frame_id}',
            height=1000,
            showlegend=True
        )
        
        return fig
    
    def create_error_table(self, differences: Dict) -> go.Figure:
        """
        誤差テーブルを作成
        
        Args:
            differences: 差分情報
            
        Returns:
            Plotly Figure
        """
        joint_names = []
        errors_3d = []
        delta_thetas = []
        delta_psis = []
        
        for joint_name, diff in differences.items():
            joint_names.append(joint_name)
            errors_3d.append(f"{diff['error_3d']:.4f}")
            delta_thetas.append(f"{np.degrees(diff['delta_theta']):.2f}°")
            delta_psis.append(f"{np.degrees(diff['delta_psi']):.2f}°")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Joint</b>', '<b>3D Error</b>', '<b>Δθ (XY)</b>', '<b>Δψ (XZ)</b>'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[joint_names, errors_3d, delta_thetas, delta_psis],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title='Error Analysis Table',
            height=400
        )
        
        return fig
```

---

### **Phase 4: Jupyter Notebookでの統合テスト（Day 6）**

**4.1 単一フレームテスト**

```python
# notebooks/02_single_frame_test.ipynb

import sys
sys.path.append('..')

from scripts.data_loader import DataLoader
from scripts.coordinate_transform import CoordinateTransformer
from scripts.validation import Validator
from scripts.plotly_visualizer import PlotlyVisualizer

# ===== データ読み込み =====
print("=== Step 1: Data Loading ===")
loader = DataLoader(".")
gt_df = loader.load_ground_truth()
cameras = loader.list_available_cameras()
mp_df = loader.load_mediapipe(cameras[0])

# ===== 1フレームの座標取得 =====
print("\n=== Step 2: Extract Single Frame ===")
frame_id = 0
gt_coords_raw = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)

print(f"GroundTruth joints: {len(gt_coords_raw)}")
print(f"MediaPipe joints: {len(mp_coords)}")

# ===== 座標変換 =====
print("\n=== Step 3: Coordinate Transformation ===")
transformer = CoordinateTransformer()

# GroundTruthを変換
gt_coords_relative, hip_center = transformer.transform_ground_truth(gt_coords_raw)
print(f"Hip center (before relative): {hip_center}")

# ===== 検証 =====
print("\n=== Step 4: Validation ===")
validator = Validator()

print("\n--- GroundTruth Validation ---")
validator.check_hip_at_origin(gt_coords_relative)
validator.check_coordinate_ranges(gt_coords_relative)
validator.check_symmetry(gt_coords_relative)

print("\n--- MediaPipe Validation ---")
validator.check_coordinate_ranges(mp_coords)
validator.check_symmetry(mp_coords)

# ===== 差分計算 =====
print("\n=== Step 5: Calculate Differences ===")
differences = transformer.calculate_differences(gt_coords_relative, mp_coords)

# 統計サマリー
errors = [diff['error_3d'] for diff in differences.values()]
print(f"Mean 3D error: {np.mean(errors):.4f}")
print(f"Max 3D error: {np.max(errors):.4f}")
print(f"Min 3D error: {np.min(errors):.4f}")

# ===== Plotly可視化 =====
print("\n=== Step 6: Plotly Visualization ===")
visualizer = PlotlyVisualizer()

# 6.1 左右並べて比較
fig1 = visualizer.plot_side_by_side(gt_coords_relative, mp_coords, frame_id)
fig1.show()
fig1.write_html("../output/html_reports/side_by_side_comparison.html")
print("Saved: side_by_side_comparison.html")

# 6.2 オーバーレイ表示
fig2 = visualizer.plot_overlay(gt_coords_relative, mp_coords, frame_id, differences)
fig2.show()
fig2.write_html("../output/html_reports/overlay_comparison.html")
print("Saved: overlay_comparison.html")

# 6.3 多視点表示
fig3 = visualizer.plot_multi_view(gt_coords_relative, mp_coords, frame_id)
fig3.show()
fig3.write_html("../output/html_reports/multi_view_comparison.html")
print("Saved: multi_view_comparison.html")

# 6.4 誤差テーブル
fig4 = visualizer.create_error_table(differences)
fig4.show()
fig4.write_html("../output/html_reports/error_table.html")
print("Saved: error_table.html")

print("\n=== All visualizations completed! ===")
```

---

### **Phase 5: 複数フレーム検証とレポート生成（Day 7）**

```python
# notebooks/03_multi_frame_validation.ipynb

# 複数フレームでの検証
frames_to_test = [0, 10, 20, 30, 40]
all_differences = {}

for frame_id in frames_to_test:
    gt_coords_raw = loader.get_frame_coordinates(gt_df, frame_id, is_mediapipe=False)
    mp_coords = loader.get_frame_coordinates(mp_df, frame_id, is_mediapipe=True)
    
    gt_coords_relative, _ = transformer.transform_ground_truth(gt_coords_raw)
    differences = transformer.calculate_differences(gt_coords_relative, mp_coords)
    
    all_differences[frame_id] = differences
    
    # 各フレームのHTML生成
    fig = visualizer.plot_side_by_side(gt_coords_relative, mp_coords, frame_id)
    fig.write_html(f"../output/html_reports/frame_{frame_id:04d}_comparison.html")

print(f"Generated {len(frames_to_test)} HTML reports")
```

---

## ✅ 実装チェックリスト

- [ ] Phase 1: データローダーの実装
  - [ ] GroundTruth読み込み
  - [ ] MediaPipe読み込み
  - [ ] データ構造確認スクリプト
  
- [ ] Phase 2: 座標変換の実装
  - [ ] 右手系→左手系変換
  - [ ] 相対座標化
  - [ ] 角度計算
  - [ ] 差分計算
  
- [ ] Phase 3: Plotly可視化
  - [ ] 左右並べて比較
  - [ ] オーバーレイ表示
  - [ ] 多視点表示
  - [ ] 誤差テーブル
  
- [ ] Phase 4: 統合テスト
  - [ ] 1フレームでの検証
  - [ ] 検証項目の確認
  - [ ] HTML出力確認
  
- [ ] Phase 5: 複数フレーム
  - [ ] 複数フレームのバッチ処理
  - [ ] 統計分析
  - [ ] 最終レポート生成

---

## 🎯 次のアクション

1. **まず実行すること:**
   ```bash
   # ディレクトリ作成
   mkdir -p scripts notebooks output/html_reports output/validation_results output/screenshots
   
   # ライブラリインストール
   pip install plotly pandas numpy matplotlib
   ```

2. **最初に作成するファイル:**
   - `scripts/data_loader.py`
   - `notebooks/01_data_exploration.ipynb`

3. **最初にテストすること:**
   - GroundTruthのCSVが正しく読み込めるか
   - MediaPipeのCSVが正しく読み込めるか
   - カラム名が想定通りか

---

このプランでPlotlyを使った座標変換検証を段階的に実装できます！まずはPhase 1から始めましょうか？

