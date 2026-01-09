"""
ログ管理モジュール
デバッグしやすいログ機能を提供
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import config


class Logger:
    """ログ管理クラス"""
    
    def __init__(self, name: str = "CoordinateTransform", log_file: str = None):
        """
        Args:
            name: ロガー名
            log_file: ログファイルパス（Noneの場合は自動生成）
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # 既存のハンドラをクリア
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラ（UTF-8対応）
        import io
        console_handler = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        )
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # ファイルハンドラ
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = config.LOG_DIR / f"log_{timestamp}.log"
        else:
            log_file = Path(log_file)
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # ファイルには全て記録
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # latest.log へのシンボリックリンク（常に最新ログを参照可能）
        latest_log = config.LOG_DIR / "latest.log"
        try:
            if latest_log.exists():
                latest_log.unlink()
            # Windowsではシンボリックリンクの代わりにコピー
            import shutil
            shutil.copy2(log_file, latest_log)
        except Exception:
            pass
        
        self.log_file = log_file
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def debug(self, msg: str):
        """DEBUGレベルログ"""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """INFOレベルログ"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """WARNINGレベルログ"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """ERRORレベルログ"""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """CRITICALレベルログ"""
        self.logger.critical(msg)
    
    def section(self, title: str):
        """セクション区切り"""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"  {title}")
        self.logger.info(separator)
    
    def step(self, step_num: int, description: str):
        """ステップ表示"""
        self.logger.info(f"\n>>> Step {step_num}: {description}")


# グローバルロガーインスタンス
_global_logger = None


def get_logger(name: str = "CoordinateTransform") -> Logger:
    """
    グローバルロガーを取得
    
    Args:
        name: ロガー名
        
    Returns:
        Loggerインスタンス
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name)
    return _global_logger


if __name__ == "__main__":
    # テスト
    config.create_output_dirs()
    
    logger = get_logger("TestLogger")
    logger.section("Logger Test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.step(1, "First step")
    logger.step(2, "Second step")
    
    print(f"\n✅ Log file created: {logger.log_file}")
    print(f"✅ Latest log: {config.LOG_DIR / 'latest.log'}")

