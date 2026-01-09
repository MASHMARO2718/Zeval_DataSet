"""
座標変換モジュール
GroundTruthからMediaPipe座標系への変換を実装
"""

import numpy as np
from typing import Dict, Tuple
import config
from scripts.logger import get_logger


class CoordinateTransformer:
    """座標系変換クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = get_logger("CoordinateTransformer")
        self.hip_joints = ['LEFT_HIP', 'RIGHT_HIP']
        self.logger.info("CoordinateTransformer initialized")
    
    def right_to_left_hand(self, coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        右手座標系から左手座標系への変換
        
        Args:
            coords: {joint_name: np.array([x, y, z])}
            
        Returns:
            変換後の座標
        """
        self.logger.debug("Converting right-hand to left-hand coordinate system")
        
        transformed = {}
        
        for joint_name, coord in coords.items():
            x_gt, y_gt, z_gt = coord
            
            # Step 1: 座標系変換
            # MediaPipeとGroundTruthが同じカメラ視点の場合、Y軸反転は不要
            # ※ MediaPipeも腰を原点とした相対座標になるため、軸の向きは一致すべき
            x_lh = x_gt
            y_lh = y_gt  # ⭐ Y軸反転なし（テスト中）
            z_lh = z_gt
            
            transformed[joint_name] = np.array([x_lh, y_lh, z_lh], dtype=np.float64)
        
        self.logger.debug(f"Transformed {len(transformed)} joints to left-hand system")
        
        return transformed
    
    def calculate_hip_center(self, coords: Dict[str, np.ndarray]) -> np.ndarray:
        """
        腰の中心座標を計算
        
        Args:
            coords: 座標辞書
            
        Returns:
            np.array([hip_x, hip_y, hip_z])
        """
        self.logger.debug("Calculating hip center")
        
        left_hip = coords.get('LEFT_HIP')
        right_hip = coords.get('RIGHT_HIP')
        
        if left_hip is None or right_hip is None:
            self.logger.error("Hip joints not found in coordinates")
            raise ValueError("Hip joints (LEFT_HIP, RIGHT_HIP) not found in coordinates")
        
        hip_center = (left_hip + right_hip) / 2.0
        
        self.logger.info(f"Hip center: {hip_center}")
        self.logger.debug(f"  LEFT_HIP: {left_hip}")
        self.logger.debug(f"  RIGHT_HIP: {right_hip}")
        
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
        self.logger.debug(f"Converting to relative coordinates (origin: {hip_center})")
        
        relative = {}
        
        for joint_name, coord in coords.items():
            relative[joint_name] = coord - hip_center
        
        self.logger.debug(f"Converted {len(relative)} joints to relative coordinates")
        
        # デバッグモードで相対座標を保存
        if config.SAVE_INTERMEDIATE_DATA:
            debug_file = config.DEBUG_DATA_DIR / "relative_coordinates.txt"
            with open(debug_file, 'w') as f:
                f.write(f"Hip center: {hip_center}\n\n")
                for joint, coord in relative.items():
                    f.write(f"{joint}: {coord}\n")
            self.logger.debug(f"Relative coordinates saved: {debug_file}")
        
        return relative
    
    def transform_ground_truth(self, gt_coords: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        GroundTruth座標を完全に変換
        
        Args:
            gt_coords: GroundTruth座標（右手系）
            
        Returns:
            (相対座標, 腰の中心座標)
        """
        self.logger.section("GroundTruth Coordinate Transformation")
        
        # Step 1: 右手系→左手系
        self.logger.step(1, "Right-hand to Left-hand conversion")
        lh_coords = self.right_to_left_hand(gt_coords)
        
        # Step 2: 腰の中心を計算
        self.logger.step(2, "Calculate hip center")
        hip_center = self.calculate_hip_center(lh_coords)
        
        # Step 3: 相対座標化
        self.logger.step(3, "Convert to relative coordinates")
        relative_coords = self.to_relative_coordinates(lh_coords, hip_center)
        
        self.logger.info("✅ GroundTruth transformation completed")
        
        return relative_coords, hip_center
    
    def transform_mediapipe(self, mp_coords: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        MediaPipe座標を相対座標に変換
        
        Args:
            mp_coords: MediaPipeの生座標（画像座標系）
            
        Returns:
            (相対座標, 腰の中心座標)
        """
        self.logger.section("MediaPipe Coordinate Transformation")
        
        # Step 1: 腰の中心を計算
        self.logger.step(1, "Calculate hip center")
        hip_center = self.calculate_hip_center(mp_coords)
        
        # Step 2: 相対座標化（腰を原点とする）
        self.logger.step(2, "Convert to relative coordinates")
        relative_coords = self.to_relative_coordinates(mp_coords, hip_center)
        
        self.logger.info("✅ MediaPipe transformation completed")
        
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
        self.logger.section("Calculating Differences")
        
        differences = {}
        
        for joint_name in gt_coords.keys():
            if joint_name not in mp_coords:
                self.logger.warning(f"Joint {joint_name} not found in MediaPipe data")
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
                'delta_theta_deg': np.degrees(delta_theta),
                'psi_gt': psi_gt,
                'psi_mp': psi_mp,
                'delta_psi': delta_psi,
                'delta_psi_deg': np.degrees(delta_psi),
            }
            
            self.logger.debug(f"{joint_name}: Δθ={np.degrees(delta_theta):.2f}°, Δψ={np.degrees(delta_psi):.2f}° (3D distance={error_3d:.4f})")
        
        # 統計サマリー（角度差を主要指標として）
        delta_thetas = [abs(diff['delta_theta_deg']) for diff in differences.values()]
        delta_psis = [abs(diff['delta_psi_deg']) for diff in differences.values()]
        errors = [diff['error_3d'] for diff in differences.values()]
        
        self.logger.info(f"\n=== Angle Error Statistics (Primary Metric) ===")
        self.logger.info(f"Mean |Δθ|: {np.mean(delta_thetas):.2f}° (XY plane)")
        self.logger.info(f"Mean |Δψ|: {np.mean(delta_psis):.2f}° (XZ plane)")
        self.logger.info(f"Max |Δθ|: {np.max(delta_thetas):.2f}°")
        self.logger.info(f"Max |Δψ|: {np.max(delta_psis):.2f}°")
        
        self.logger.info(f"\n=== 3D Distance (Reference Only - Different Scales) ===")
        self.logger.info(f"Mean: {np.mean(errors):.4f}, Median: {np.median(errors):.4f}")
        self.logger.info(f"Note: GT is in meters, MP is normalized (0-1). Direct comparison is not meaningful.")
        
        # 警告: 大きな角度誤差
        ANGLE_THRESHOLD = 30.0  # 30度以上を警告
        for joint_name, diff in differences.items():
            if abs(diff['delta_theta_deg']) > ANGLE_THRESHOLD or abs(diff['delta_psi_deg']) > ANGLE_THRESHOLD:
                self.logger.warning(f"Large angle error: {joint_name} - Δθ={diff['delta_theta_deg']:.2f}°, Δψ={diff['delta_psi_deg']:.2f}°")
        
        # デバッグモードで差分データを保存
        if config.SAVE_INTERMEDIATE_DATA:
            debug_file = config.DEBUG_DATA_DIR / "differences.txt"
            with open(debug_file, 'w') as f:
                f.write("=== Coordinate Differences ===\n\n")
                for joint_name, diff in differences.items():
                    f.write(f"{joint_name}:\n")
                    f.write(f"  GT: {diff['gt_coord']}\n")
                    f.write(f"  MP: {diff['mp_coord']}\n")
                    f.write(f"  Error 3D: {diff['error_3d']:.4f}\n")
                    f.write(f"  Δθ: {diff['delta_theta_deg']:.2f}°\n")
                    f.write(f"  Δψ: {diff['delta_psi_deg']:.2f}°\n\n")
            self.logger.debug(f"Differences saved: {debug_file}")
        
        return differences


if __name__ == "__main__":
    # テスト実行
    print("=== CoordinateTransformer Test ===\n")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    # テスト用のダミーデータ
    test_coords = {
        'LEFT_HIP': np.array([0.1, 0.5, 0.0]),
        'RIGHT_HIP': np.array([-0.1, 0.5, 0.0]),
        'LEFT_SHOULDER': np.array([0.2, 1.0, 0.0]),
        'RIGHT_SHOULDER': np.array([-0.2, 1.0, 0.0]),
        'LEFT_KNEE': np.array([0.1, 0.0, 0.1]),
        'RIGHT_KNEE': np.array([-0.1, 0.0, 0.1]),
    }
    
    # 変換実行
    transformer = CoordinateTransformer()
    
    print("\n--- Step 1: Right-hand to Left-hand ---")
    lh_coords = transformer.right_to_left_hand(test_coords)
    print(f"✅ Transformed {len(lh_coords)} joints")
    
    print("\n--- Step 2: Calculate Hip Center ---")
    hip_center = transformer.calculate_hip_center(lh_coords)
    print(f"✅ Hip center: {hip_center}")
    
    print("\n--- Step 3: Relative Coordinates ---")
    relative_coords = transformer.to_relative_coordinates(lh_coords, hip_center)
    print(f"✅ Converted {len(relative_coords)} joints to relative coordinates")
    
    # 腰が原点にあるか確認
    hip_center_relative = (relative_coords['LEFT_HIP'] + relative_coords['RIGHT_HIP']) / 2.0
    print(f"\n--- Verification ---")
    print(f"Hip center in relative coords: {hip_center_relative}")
    print(f"Distance from origin: {np.linalg.norm(hip_center_relative):.10f}")
    
    # 完全な変換
    print("\n--- Full Transformation ---")
    relative_coords_full, hip_center_full = transformer.transform_ground_truth(test_coords)
    print(f"✅ Full transformation completed")
    
    # MediaPipeダミーデータとの差分計算
    mp_dummy = {
        'LEFT_HIP': np.array([0.05, -0.02, 0.01]),
        'RIGHT_HIP': np.array([-0.05, -0.02, -0.01]),
        'LEFT_SHOULDER': np.array([0.18, 0.48, 0.02]),
        'RIGHT_SHOULDER': np.array([-0.18, 0.48, -0.02]),
    }
    
    print("\n--- Difference Calculation ---")
    differences = transformer.calculate_differences(relative_coords_full, mp_dummy)
    print(f"✅ Calculated differences for {len(differences)} joints")
    
    print(f"\n✅ Test completed. Check logs: {config.LOG_DIR / 'latest.log'}")

