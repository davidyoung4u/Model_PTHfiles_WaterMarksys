# _*_ conding : utf-8 _*_
# @Time : 2025/4/9 14:26
# @Author : yang_4_ever
# @File : main
# @Project : FinalWMsys

import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from gui import WatermarkApp
from watermark_system import set_global_seed


def install_exception_hook():
    """未捕获异常：打印堆栈并弹窗提示，而不是让窗口静默消失。"""

    def hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        try:
            QMessageBox.critical(
                None,
                "未处理的异常",
                f"{exc_type.__name__}: {exc_value}\n\n详细信息见控制台输出",
            )
        except Exception:
            pass

    sys.excepthook = hook


def main() -> int:
    # 高 DPI 适配必须在 QApplication 创建之前设置
    for attribute in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        if hasattr(Qt, attribute):
            QApplication.setAttribute(getattr(Qt, attribute), True)

    # 固定随机种子（P0：评测可复现）
    set_global_seed()

    app = QApplication(sys.argv)
    install_exception_hook()
    window = WatermarkApp()
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
