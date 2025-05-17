# _*_ conding : utf-8 _*_
# @Time : 2025/4/9 14:26
# @Author : yang_4_ever
# @File : main
# @Project : FinalWMsys

import sys
from PyQt5.QtWidgets import QApplication
from gui import WatermarkApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WatermarkApp()
    window.show()
    sys.exit(app.exec_())