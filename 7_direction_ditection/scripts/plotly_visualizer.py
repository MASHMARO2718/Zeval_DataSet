"""
Plotly可視化モジュール
インタラクティブな3D可視化を提供
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List
import config
from scripts.logger import get_logger


class PlotlyVisualizer:
    """Plotlyによるインタラクティブ可視化クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = get_logger("PlotlyVisualizer")
        self.skeleton_connections = config.SKELETON_CONNECTIONS
        self.logger.info("PlotlyVisualizer initialized")
    
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
        self.logger.debug(f"Creating skeleton traces for {name}")
        
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
        
        self.logger.debug(f"Created {len(traces)} traces")
        
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
        self.logger.info(f"Creating side-by-side plot for frame {frame_id}")
        
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
        
        self.logger.info("Side-by-side plot created")
        
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
        self.logger.info(f"Creating overlay plot for frame {frame_id}")
        
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
        
        self.logger.info("Overlay plot created")
        
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
        self.logger.info(f"Creating multi-view plot for frame {frame_id}")
        
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
            self._add_2d_projection(fig, gt_coords, idx1, idx2, label1, label2, 'blue', 'GT', 1, col)
        
        # MediaPipe（下段）
        for col, (idx1, idx2, label1, label2) in enumerate(views, 1):
            self._add_2d_projection(fig, mp_coords, idx1, idx2, label1, label2, 'red', 'MP', 2, col)
        
        fig.update_layout(
            title=f'Multi-View Comparison - Frame {frame_id}',
            height=1000,
            showlegend=True
        )
        
        self.logger.info("Multi-view plot created")
        
        return fig
    
    def _add_2d_projection(self, fig, coords, idx1, idx2, label1, label2, color, name, row, col):
        """2D投影を追加"""
        joint_names = list(coords.keys())
        coords_array = np.array([coords[j] for j in joint_names])
        
        # 関節点
        fig.add_trace(
            go.Scatter(
                x=coords_array[:, idx1],
                y=coords_array[:, idx2],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=joint_names,
                textposition='top center',
                textfont=dict(size=8),
                name=name,
                showlegend=(col == 1),
                hovertemplate=f'<b>%{{text}}</b><br>{label1}: %{{x:.3f}}<br>{label2}: %{{y:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # 骨格線
        for start_joint, end_joint in self.skeleton_connections:
            if start_joint in coords and end_joint in coords:
                start = coords[start_joint]
                end = coords[end_joint]
                fig.add_trace(
                    go.Scatter(
                        x=[start[idx1], end[idx1]],
                        y=[start[idx2], end[idx2]],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text=label1, row=row, col=col)
        fig.update_yaxes(title_text=label2, row=row, col=col)
    
    def create_error_table(self, differences: Dict) -> go.Figure:
        """
        誤差テーブルを作成
        
        Args:
            differences: 差分情報
            
        Returns:
            Plotly Figure
        """
        self.logger.info("Creating error table")
        
        joint_names = []
        errors_3d = []
        delta_thetas = []
        delta_psis = []
        
        for joint_name, diff in differences.items():
            joint_names.append(joint_name)
            errors_3d.append(f"{diff['error_3d']:.4f}")
            delta_thetas.append(f"{diff['delta_theta_deg']:.2f}°")
            delta_psis.append(f"{diff['delta_psi_deg']:.2f}°")
        
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
        
        self.logger.info("Error table created")
        
        return fig
    
    def save_html(self, fig: go.Figure, filename: str):
        """
        HTMLファイルとして保存
        
        Args:
            fig: Plotly Figure
            filename: ファイル名
        """
        output_path = config.HTML_DIR / filename
        fig.write_html(str(output_path))
        self.logger.info(f"HTML saved: {output_path}")
        print(f"✅ HTML saved: {output_path}")


if __name__ == "__main__":
    # テスト実行
    print("=== PlotlyVisualizer Test ===\n")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    # テスト用のダミーデータ
    test_gt = {
        'LEFT_HIP': np.array([0.05, -0.02, 0.0]),
        'RIGHT_HIP': np.array([-0.05, -0.02, 0.0]),
        'LEFT_SHOULDER': np.array([0.2, 0.48, 0.0]),
        'RIGHT_SHOULDER': np.array([-0.2, 0.48, 0.0]),
        'LEFT_ELBOW': np.array([0.3, 0.15, 0.1]),
        'RIGHT_ELBOW': np.array([-0.3, 0.15, 0.1]),
        'LEFT_KNEE': np.array([0.1, -0.48, 0.05]),
        'RIGHT_KNEE': np.array([-0.1, -0.48, 0.05]),
    }
    
    test_mp = {
        'LEFT_HIP': np.array([0.06, -0.03, 0.01]),
        'RIGHT_HIP': np.array([-0.06, -0.03, -0.01]),
        'LEFT_SHOULDER': np.array([0.19, 0.47, 0.02]),
        'RIGHT_SHOULDER': np.array([-0.19, 0.47, -0.02]),
        'LEFT_ELBOW': np.array([0.29, 0.14, 0.11]),
        'RIGHT_ELBOW': np.array([-0.29, 0.14, 0.09]),
        'LEFT_KNEE': np.array([0.11, -0.47, 0.06]),
        'RIGHT_KNEE': np.array([-0.11, -0.47, 0.04]),
    }
    
    # ダミー差分データ
    test_diff = {
        'LEFT_SHOULDER': {
            'gt_coord': test_gt['LEFT_SHOULDER'],
            'mp_coord': test_mp['LEFT_SHOULDER'],
            'error_3d': 0.015,
            'delta_theta_deg': 2.5,
            'delta_psi_deg': 1.8,
        },
        'RIGHT_SHOULDER': {
            'gt_coord': test_gt['RIGHT_SHOULDER'],
            'mp_coord': test_mp['RIGHT_SHOULDER'],
            'error_3d': 0.012,
            'delta_theta_deg': -2.2,
            'delta_psi_deg': -1.5,
        }
    }
    
    # 可視化実行
    visualizer = PlotlyVisualizer()
    frame_id = 0
    
    print("\n--- Creating Side-by-Side Plot ---")
    fig1 = visualizer.plot_side_by_side(test_gt, test_mp, frame_id)
    visualizer.save_html(fig1, "test_side_by_side.html")
    
    print("\n--- Creating Overlay Plot ---")
    fig2 = visualizer.plot_overlay(test_gt, test_mp, frame_id, test_diff)
    visualizer.save_html(fig2, "test_overlay.html")
    
    print("\n--- Creating Multi-View Plot ---")
    fig3 = visualizer.plot_multi_view(test_gt, test_mp, frame_id)
    visualizer.save_html(fig3, "test_multi_view.html")
    
    print("\n--- Creating Error Table ---")
    fig4 = visualizer.create_error_table(test_diff)
    visualizer.save_html(fig4, "test_error_table.html")
    
    print(f"\n✅ Test completed. Check logs: {config.LOG_DIR / 'latest.log'}")
    print(f"✅ HTML files: {config.HTML_DIR}")

