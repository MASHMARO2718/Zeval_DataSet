"""
設定ファイル
プロジェクト全体の設定を管理
"""

from pathlib import Path

# ========== パス設定 ==========
# プロジェクトルート（7_direction_ditection）
PROJECT_ROOT = Path(__file__).parent

# データセットルート（Zeval_DataSet）
BASE_DIR = PROJECT_ROOT.parent

# データパス
GT_CSV = BASE_DIR / "synced_joint_positions.csv"
MP_DIR = BASE_DIR / "2_medidapipe_proccesed"

# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "output"
HTML_DIR = OUTPUT_DIR / "html_reports"
LOG_DIR = OUTPUT_DIR / "logs"
DEBUG_DATA_DIR = OUTPUT_DIR / "debug_data"
VALIDATION_DIR = OUTPUT_DIR / "validation_results"

# ========== デバッグ設定 ==========
DEBUG_MODE = True  # Trueで詳細デバッグ情報を出力
LOG_LEVEL = "DEBUG" if DEBUG_MODE else "INFO"  # ログレベル
SAVE_INTERMEDIATE_DATA = DEBUG_MODE  # 中間データを保存するか

# ========== MediaPipe関節マッピング ==========
JOINT_MAPPING = {
    'NOSE': 0,
    'LEFT_EYE_INNER': 1,
    'LEFT_EYE': 2,
    'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4,
    'RIGHT_EYE': 5,
    'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'MOUTH_LEFT': 9,
    'MOUTH_RIGHT': 10,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17,
    'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19,
    'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21,
    'RIGHT_THUMB': 22,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32,
}

# ========== 重要な関節（検証用） ==========
KEY_JOINTS = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
]

# ========== 骨格接続定義 ==========
SKELETON_CONNECTIONS = [
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

# ========== 検証設定 ==========
VALIDATION_TOLERANCE = 1e-6  # 原点確認の許容誤差
MAX_ERROR_THRESHOLD = 0.5  # 最大誤差の閾値（警告用）

# ========== Y範囲設定 ==========
Y_RANGES = ["Y=0.5,1.5", "Y=1.0.2.0"]
DEFAULT_Y_RANGE = "Y=0.5,1.5"


def create_output_dirs():
    """出力ディレクトリを作成"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    HTML_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    DEBUG_DATA_DIR.mkdir(exist_ok=True)
    VALIDATION_DIR.mkdir(exist_ok=True)
    print(f"[OK] Output directories created in: {OUTPUT_DIR}")


if __name__ == "__main__":
    # 設定確認用
    print("=== Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Root: {BASE_DIR}")
    print(f"GroundTruth CSV: {GT_CSV}")
    print(f"MediaPipe Dir: {MP_DIR}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print(f"Log Level: {LOG_LEVEL}")
    
    # ディレクトリ作成
    create_output_dirs()

