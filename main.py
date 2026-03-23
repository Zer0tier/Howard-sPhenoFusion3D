"""
PhenoFusion3D - Entry point. Launches PyQt application.
Run from project root: python main.py
"""
import sys
import os

# Ensure project root is on path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PyQt5.QtWidgets import QApplication
from phenofusion3d.app import MainWindow, Controller


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PhenoFusion3D")
    app.setOrganizationName("APPN")

    win = MainWindow()
    controller = Controller(win)
    win.controller = controller

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
