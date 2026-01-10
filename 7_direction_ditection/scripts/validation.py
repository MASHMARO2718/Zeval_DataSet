"""
検証モジュール
座標変換の正しさを検証するユーティリティ
"""

import numpy as np
from typing import Dict
import config
from scripts.logger import get_logger


class Validator:
    """座標変換の検証クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = get_logger("Validator")
        self.tolerance = config.VALIDATION_TOLERANCE
        self.logger.info("Validator initialized")
    
    def check_hip_at_origin(self, coords: Dict[str, np.ndarray], 
                          tolerance: float = None) -> bool:
        """
        腰が原点にあるか確認
        
        Args:
            coords: 相対座標辞書
            tolerance: 許容誤差（Noneの場合はデフォルト）
            
        Returns:
            bool: 原点にある場合True
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        self.logger.section("Hip Origin Validation")
        
        if 'LEFT_HIP' not in coords or 'RIGHT_HIP' not in coords:
            self.logger.error("Hip joints not found in coordinates")
            return False
        
        left_hip = coords['LEFT_HIP']
        right_hip = coords['RIGHT_HIP']
        
        hip_center = (left_hip + right_hip) / 2.0
        distance_from_origin = np.linalg.norm(hip_center)
        
        is_at_origin = distance_from_origin < tolerance
        
        self.logger.info(f"Left hip: {left_hip}")
        self.logger.info(f"Right hip: {right_hip}")
        self.logger.info(f"Hip center: {hip_center}")
        self.logger.info(f"Distance from origin: {distance_from_origin:.10f}")
        self.logger.info(f"Tolerance: {tolerance}")
        
        if is_at_origin:
            self.logger.info("✅ Hip is at origin")
        else:
            self.logger.warning(f"⚠️  Hip is NOT at origin (distance: {distance_from_origin:.10f})")
        
        return is_at_origin
    
    def check_coordinate_ranges(self, coords: Dict[str, np.ndarray]) -> Dict:
        """
        座標の範囲を確認
        
        Args:
            coords: 座標辞書
            
        Returns:
            統計情報
        """
        self.logger.section("Coordinate Range Check")
        
        all_coords = np.array(list(coords.values()))
        
        stats = {
            'x_range': (all_coords[:, 0].min(), all_coords[:, 0].max()),
            'y_range': (all_coords[:, 1].min(), all_coords[:, 1].max()),
            'z_range': (all_coords[:, 2].min(), all_coords[:, 2].max()),
            'x_mean': all_coords[:, 0].mean(),
            'y_mean': all_coords[:, 1].mean(),
            'z_mean': all_coords[:, 2].mean(),
            'x_std': all_coords[:, 0].std(),
            'y_std': all_coords[:, 1].std(),
            'z_std': all_coords[:, 2].std(),
        }
        
        self.logger.info("Coordinate Ranges:")
        self.logger.info(f"  X: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}] (mean: {stats['x_mean']:.3f}, std: {stats['x_std']:.3f})")
        self.logger.info(f"  Y: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}] (mean: {stats['y_mean']:.3f}, std: {stats['y_std']:.3f})")
        self.logger.info(f"  Z: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}] (mean: {stats['z_mean']:.3f}, std: {stats['z_std']:.3f})")
        
        # デバッグモードで統計を保存
        if config.SAVE_INTERMEDIATE_DATA:
            debug_file = config.DEBUG_DATA_DIR / "coordinate_stats.txt"
            with open(debug_file, 'w') as f:
                f.write("=== Coordinate Statistics ===\n\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
            self.logger.debug(f"Statistics saved: {debug_file}")
        
        return stats
    
    def check_symmetry(self, coords: Dict[str, np.ndarray]) -> Dict:
        """
        左右の対称性を確認
        
        Args:
            coords: 座標辞書
            
        Returns:
            対称性情報
        """
        self.logger.section("Symmetry Check")
        
        pairs = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_ELBOW', 'RIGHT_ELBOW'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_KNEE', 'RIGHT_KNEE'),
            ('LEFT_ANKLE', 'RIGHT_ANKLE'),
        ]
        
        symmetry_info = {}
        
        for left, right in pairs:
            if left in coords and right in coords:
                left_coord = coords[left]
                right_coord = coords[right]
                
                # X座標の符号が逆であることを期待（左右対称）
                x_symmetry = abs(left_coord[0] + right_coord[0])
                
                # Y, Z座標はほぼ同じであることを期待
                y_diff = abs(left_coord[1] - right_coord[1])
                z_diff = abs(left_coord[2] - right_coord[2])
                
                symmetry_info[left] = {
                    'x_symmetry': x_symmetry,
                    'y_diff': y_diff,
                    'z_diff': z_diff,
                    'is_symmetric': x_symmetry < 0.1 and y_diff < 0.1 and z_diff < 0.1
                }
                
                status = "✅" if symmetry_info[left]['is_symmetric'] else "⚠️ "
                self.logger.info(f"{status} {left} <-> {right}:")
                self.logger.info(f"    X symmetry (should be ~0): {x_symmetry:.3f}")
                self.logger.info(f"    Y difference: {y_diff:.3f}")
                self.logger.info(f"    Z difference: {z_diff:.3f}")
        
        return symmetry_info
    
    def check_joint_distances(self, coords: Dict[str, np.ndarray]) -> Dict:
        """
        関節間距離の妥当性を確認
        
        Args:
            coords: 座標辞書
            
        Returns:
            距離情報
        """
        self.logger.section("Joint Distance Check")
        
        # 主要な関節間距離
        distance_pairs = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'shoulder_width'),
            ('LEFT_HIP', 'RIGHT_HIP', 'hip_width'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW', 'left_upper_arm'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'right_upper_arm'),
            ('LEFT_HIP', 'LEFT_KNEE', 'left_thigh'),
            ('RIGHT_HIP', 'RIGHT_KNEE', 'right_thigh'),
        ]
        
        distances = {}
        
        for joint1, joint2, name in distance_pairs:
            if joint1 in coords and joint2 in coords:
                dist = np.linalg.norm(coords[joint1] - coords[joint2])
                distances[name] = dist
                self.logger.info(f"{name}: {dist:.3f}")
        
        # 対称性チェック（左右の長さが似ているか）
        if 'left_upper_arm' in distances and 'right_upper_arm' in distances:
            diff = abs(distances['left_upper_arm'] - distances['right_upper_arm'])
            self.logger.info(f"Upper arm symmetry diff: {diff:.3f}")
            if diff > 0.1:
                self.logger.warning("⚠️  Large difference in upper arm length")
        
        if 'left_thigh' in distances and 'right_thigh' in distances:
            diff = abs(distances['left_thigh'] - distances['right_thigh'])
            self.logger.info(f"Thigh symmetry diff: {diff:.3f}")
            if diff > 0.1:
                self.logger.warning("⚠️  Large difference in thigh length")
        
        return distances
    
    def validate_all(self, gt_coords: Dict[str, np.ndarray],
                    mp_coords: Dict[str, np.ndarray]) -> Dict:
        """
        すべての検証を実行
        
        Args:
            gt_coords: GroundTruth相対座標
            mp_coords: MediaPipe座標
            
        Returns:
            検証結果サマリー
        """
        self.logger.section("Complete Validation")
        
        results = {
            'gt_hip_at_origin': False,
            'gt_stats': {},
            'gt_symmetry': {},
            'gt_distances': {},
            'mp_stats': {},
            'mp_symmetry': {},
            'mp_distances': {},
            'validation_passed': False
        }
        
        # GroundTruth検証
        self.logger.info("\n=== GroundTruth Validation ===")
        results['gt_hip_at_origin'] = self.check_hip_at_origin(gt_coords)
        results['gt_stats'] = self.check_coordinate_ranges(gt_coords)
        results['gt_symmetry'] = self.check_symmetry(gt_coords)
        results['gt_distances'] = self.check_joint_distances(gt_coords)
        
        # MediaPipe検証
        self.logger.info("\n=== MediaPipe Validation ===")
        results['mp_stats'] = self.check_coordinate_ranges(mp_coords)
        results['mp_symmetry'] = self.check_symmetry(mp_coords)
        results['mp_distances'] = self.check_joint_distances(mp_coords)
        
        # 総合判定
        results['validation_passed'] = results['gt_hip_at_origin']
        
        if results['validation_passed']:
            self.logger.info("\n✅ ===== VALIDATION PASSED ===== ✅")
        else:
            self.logger.warning("\n⚠️  ===== VALIDATION FAILED ===== ⚠️ ")
        
        # デバッグモードで検証結果を保存
        if config.SAVE_INTERMEDIATE_DATA:
            debug_file = config.DEBUG_DATA_DIR / "validation_results.txt"
            with open(debug_file, 'w') as f:
                f.write("=== Validation Results ===\n\n")
                f.write(f"Hip at origin: {results['gt_hip_at_origin']}\n")
                f.write(f"Validation passed: {results['validation_passed']}\n\n")
                f.write("See log file for detailed results.\n")
            self.logger.debug(f"Validation results saved: {debug_file}")
        
        return results


if __name__ == "__main__":
    # テスト実行
    print("=== Validator Test ===\n")
    
    # 出力ディレクトリ作成
    config.create_output_dirs()
    
    # テスト用のダミーデータ（相対座標）
    test_coords_relative = {
        'LEFT_HIP': np.array([0.05, -0.02, 0.0]),
        'RIGHT_HIP': np.array([-0.05, -0.02, 0.0]),
        'LEFT_SHOULDER': np.array([0.2, 0.48, 0.0]),
        'RIGHT_SHOULDER': np.array([-0.2, 0.48, 0.0]),
        'LEFT_ELBOW': np.array([0.3, 0.15, 0.1]),
        'RIGHT_ELBOW': np.array([-0.3, 0.15, 0.1]),
        'LEFT_KNEE': np.array([0.1, -0.48, 0.05]),
        'RIGHT_KNEE': np.array([-0.1, -0.48, 0.05]),
    }
    
    test_coords_mp = {
        'LEFT_HIP': np.array([0.06, -0.03, 0.01]),
        'RIGHT_HIP': np.array([-0.06, -0.03, -0.01]),
        'LEFT_SHOULDER': np.array([0.19, 0.47, 0.02]),
        'RIGHT_SHOULDER': np.array([-0.19, 0.47, -0.02]),
        'LEFT_ELBOW': np.array([0.29, 0.14, 0.11]),
        'RIGHT_ELBOW': np.array([-0.29, 0.14, 0.09]),
    }
    
    # 検証実行
    validator = Validator()
    
    print("\n--- Individual Checks ---")
    validator.check_hip_at_origin(test_coords_relative)
    validator.check_coordinate_ranges(test_coords_relative)
    validator.check_symmetry(test_coords_relative)
    validator.check_joint_distances(test_coords_relative)
    
    print("\n--- Complete Validation ---")
    results = validator.validate_all(test_coords_relative, test_coords_mp)
    print(f"\n✅ Validation completed: {'PASSED' if results['validation_passed'] else 'FAILED'}")
    
    print(f"\n✅ Test completed. Check logs: {config.LOG_DIR / 'latest.log'}")



