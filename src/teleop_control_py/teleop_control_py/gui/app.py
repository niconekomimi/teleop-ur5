import sys

from PySide6.QtWidgets import QApplication

from .main_window import TeleopMainWindow
from .theme import configure_application


def main() -> int:
    app = QApplication(sys.argv)
    configure_application(app)
    window = TeleopMainWindow()
    window.show()
    exit_code = 0
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        try:
            window._shutdown()
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
