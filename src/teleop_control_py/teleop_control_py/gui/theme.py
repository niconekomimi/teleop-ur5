"""Application-wide visual theme for the desktop operator station."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette, QPolygonF
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QProxyStyle,
    QStyle,
    QStyleOption,
    QStyleOptionComplex,
    QWidget,
)


CANVAS = "#F3F5F7"
SURFACE = "#FFFFFF"
SURFACE_SUBTLE = "#F8FAFB"
BORDER = "#D7DCE2"
BORDER_STRONG = "#B8C1CC"
TEXT_PRIMARY = "#1F2933"
TEXT_SECONDARY = "#5F6B76"
TEXT_DISABLED = "#98A2AD"
PRIMARY = "#2563EB"
SUCCESS = "#16794B"
WARNING = "#B15C00"
DANGER = "#B42318"


# Sections are styled by the application stylesheet. Keeping this constant empty
# lets existing panel constructors retain their section_style argument without
# creating a competing widget-local stylesheet.
SECTION_STYLE = ""


APP_STYLESHEET = f"""
QMainWindow,
QDialog {{
    background-color: {CANVAS};
    color: {TEXT_PRIMARY};
}}

QWidget {{
    color: {TEXT_PRIMARY};
    selection-background-color: #DCE8FF;
    selection-color: {TEXT_PRIMARY};
}}

QWidget#workspaceShell {{
    background-color: {CANVAS};
}}

QWidget#workspaceTopBar {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QLabel#workspaceProductName {{
    color: {TEXT_PRIMARY};
    font-size: 16px;
    font-weight: 700;
}}

QWidget[role="workspaceColumn"] {{
    background-color: transparent;
}}

QLabel {{
    background-color: transparent;
}}

QLabel[role="hint"] {{
    color: {TEXT_SECONDARY};
}}

QLabel[role="value"] {{
    color: #174EB6;
    font-weight: 600;
}}

QLabel[role="subsection-title"] {{
    color: {TEXT_PRIMARY};
    font-weight: 700;
}}

QLabel[role="table-header"] {{
    color: {TEXT_SECONDARY};
    font-weight: 600;
}}

QLabel[role="metadata"] {{
    color: {TEXT_SECONDARY};
}}

QLabel[role="image-viewport"] {{
    background-color: #252B31;
    color: #D6DCE2;
    border: 1px solid #4B5560;
    border-radius: 5px;
}}

QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 6px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
}}

QGroupBox[role="subsection"] {{
    background-color: transparent;
    border: 0;
    border-top: 1px solid #E6EAEE;
    border-radius: 0;
    margin-top: 12px;
    padding-top: 7px;
}}

QGroupBox[role="subsection"]::title {{
    left: 0;
    padding: 0 7px 0 0;
    background-color: {SURFACE};
    color: {TEXT_SECONDARY};
    font-weight: 600;
}}

QLineEdit,
QComboBox {{
    min-height: 28px;
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
}}

QLineEdit {{
    padding: 0 8px;
}}

QComboBox {{
    padding: 0 32px 0 8px;
}}

QLineEdit:hover,
QComboBox:hover {{
    border-color: #8F9AA6;
}}

QLineEdit:focus,
QComboBox:focus {{
    border-color: {PRIMARY};
}}

QLineEdit:read-only {{
    background-color: {SURFACE_SUBTLE};
    color: {TEXT_SECONDARY};
}}

QLineEdit:disabled,
QComboBox:disabled {{
    background-color: #EEF1F4;
    color: {TEXT_DISABLED};
    border-color: #E0E4E8;
}}

QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    outline: 0;
    selection-background-color: #EAF1FF;
    selection-color: {TEXT_PRIMARY};
}}

QTabWidget::pane {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    top: -1px;
}}

QTabBar::tab {{
    min-height: 30px;
    padding: 0 14px;
    background-color: #E9EDF1;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-bottom-color: {BORDER_STRONG};
}}

QTabBar::tab:first {{
    border-top-left-radius: 5px;
}}

QTabBar::tab:last {{
    border-top-right-radius: 5px;
}}

QTabBar::tab:middle,
QTabBar::tab:last {{
    border-left: 0;
}}

QTabBar::tab:hover:!selected {{
    background-color: #F2F5F8;
    color: {TEXT_PRIMARY};
}}

QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: #174EB6;
    border-bottom: 2px solid {PRIMARY};
    font-weight: 600;
}}

QSlider:horizontal {{
    min-height: 20px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background-color: #D5DBE2;
    border: 0;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background-color: {PRIMARY};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    background-color: {SURFACE};
    border: 2px solid {PRIMARY};
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background-color: #EAF1FF;
}}

QSlider::handle:horizontal:pressed {{
    background-color: {PRIMARY};
}}

QSlider:disabled::groove:horizontal,
QSlider:disabled::sub-page:horizontal {{
    background-color: #E0E4E8;
}}

QSlider:disabled::handle:horizontal {{
    background-color: #EEF1F4;
    border-color: {TEXT_DISABLED};
}}

QPushButton {{
    min-height: 28px;
    padding: 0 12px;
    background-color: {SURFACE};
    color: #344054;
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {SURFACE_SUBTLE};
    border-color: #8F9AA6;
}}

QPushButton:pressed {{
    background-color: #EDF0F3;
    border-color: #7C8793;
}}

QPushButton:focus {{
    border-color: {PRIMARY};
}}

QPushButton:checked {{
    background-color: #EAF1FF;
    color: #174EB6;
    border-color: {PRIMARY};
}}

QPushButton[role="default"],
QPushButton[role="secondary"] {{
    background-color: {SURFACE};
    color: #344054;
    border-color: {BORDER_STRONG};
}}

QPushButton[role="default"]:hover,
QPushButton[role="secondary"]:hover {{
    background-color: {SURFACE_SUBTLE};
    border-color: #8F9AA6;
}}

QPushButton[role="default"]:checked,
QPushButton[role="secondary"]:checked {{
    background-color: #EAF1FF;
    color: #174EB6;
    border-color: {PRIMARY};
}}

QPushButton[role="primary"] {{
    background-color: {PRIMARY};
    color: #FFFFFF;
    border-color: {PRIMARY};
}}

QPushButton[role="primary"]:hover {{
    background-color: #1D4ED8;
    border-color: #1D4ED8;
}}

QPushButton[role="primary"]:pressed,
QPushButton[role="primary"]:checked {{
    background-color: #1E40AF;
    border-color: #1E40AF;
}}

QPushButton[role="success"] {{
    background-color: {SUCCESS};
    color: #FFFFFF;
    border-color: {SUCCESS};
}}

QPushButton[role="success"]:hover {{
    background-color: #12663F;
    border-color: #12663F;
}}

QPushButton[role="success"]:pressed,
QPushButton[role="success"]:checked {{
    background-color: #0F5334;
    border-color: #0F5334;
}}

QPushButton[role="warning"] {{
    background-color: #FFF4E5;
    color: #8A4600;
    border-color: #D8892F;
}}

QPushButton[role="warning"]:hover {{
    background-color: #FFE8C5;
    border-color: {WARNING};
}}

QPushButton[role="warning"]:pressed,
QPushButton[role="warning"]:checked {{
    background-color: {WARNING};
    color: #FFFFFF;
    border-color: {WARNING};
}}

QPushButton[role="danger"] {{
    background-color: {DANGER};
    color: #FFFFFF;
    border-color: {DANGER};
}}

QPushButton[role="danger"]:hover {{
    background-color: #912018;
    border-color: #912018;
}}

QPushButton[role="danger"]:pressed,
QPushButton[role="danger"]:checked {{
    background-color: #7A1B14;
    border-color: #7A1B14;
}}

QPushButton[role="danger-secondary"] {{
    background-color: {SURFACE};
    color: {DANGER};
    border-color: #DFA6A1;
}}

QPushButton[role="danger-secondary"]:hover {{
    background-color: #FDECEC;
    border-color: {DANGER};
}}

QPushButton:disabled,
QPushButton[role="default"]:disabled,
QPushButton[role="secondary"]:disabled,
QPushButton[role="primary"]:disabled,
QPushButton[role="success"]:disabled,
QPushButton[role="warning"]:disabled,
QPushButton[role="danger"]:disabled,
QPushButton[role="danger-secondary"]:disabled {{
    background-color: #EEF1F4;
    color: {TEXT_DISABLED};
    border-color: #E0E4E8;
}}

QCheckBox {{
    spacing: 7px;
    color: {TEXT_PRIMARY};
}}

QLabel[role="status"],
QLabel[role="status-neutral"],
QLabel[role="status-info"],
QLabel[role="status-success"],
QLabel[role="status-warning"],
QLabel[role="status-danger"] {{
    min-height: 22px;
    padding: 1px 7px;
    background-color: {SURFACE_SUBTLE};
    color: {TEXT_SECONDARY};
    border: 0;
    border-left: 3px solid {TEXT_DISABLED};
    border-radius: 3px;
    font-weight: 600;
}}

QLabel[role="status-info"] {{
    background-color: #EAF1FF;
    color: #174EB6;
    border-left-color: {PRIMARY};
}}

QLabel[role="status-success"] {{
    background-color: #E9F6EF;
    color: {SUCCESS};
    border-left-color: {SUCCESS};
}}

QLabel[role="status-warning"] {{
    background-color: #FFF4E5;
    color: #8A4600;
    border-left-color: {WARNING};
}}

QLabel[role="status-danger"] {{
    background-color: #FDECEC;
    color: {DANGER};
    border-left-color: {DANGER};
}}

QTextEdit,
QPlainTextEdit {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px;
}}

QTextEdit:focus,
QPlainTextEdit:focus {{
    border-color: {PRIMARY};
}}

QTextEdit#runtimeLog,
QPlainTextEdit#runtimeLog,
QTextEdit[role="runtimeLog"],
QPlainTextEdit[role="runtimeLog"],
QTextEdit[role="runtime-log"],
QPlainTextEdit[role="runtime-log"] {{
    background-color: {SURFACE_SUBTLE};
    color: #344054;
    border-color: {BORDER};
    font-family: "Cascadia Mono", "Consolas", monospace;
}}

QTextEdit#actionOutput,
QPlainTextEdit#actionOutput,
QTextEdit[role="actionOutput"],
QPlainTextEdit[role="actionOutput"],
QTextEdit[role="action-output"],
QPlainTextEdit[role="action-output"] {{
    background-color: {SURFACE_SUBTLE};
    color: #344054;
    border-color: {BORDER};
    font-family: "Cascadia Mono", "Consolas", monospace;
}}

QSplitter::handle {{
    background-color: {BORDER};
}}

QSplitter::handle:hover {{
    background-color: {PRIMARY};
}}

QSplitter::handle:horizontal {{
    width: 2px;
    margin: 0 4px;
}}

QSplitter::handle:vertical {{
    height: 2px;
    margin: 4px 0;
}}

QScrollArea {{
    background-color: transparent;
    border: 0;
}}

QScrollArea QWidget#qt_scrollarea_viewport {{
    background-color: transparent;
}}

QScrollBar:vertical {{
    width: 10px;
    margin: 2px;
    background-color: transparent;
}}

QScrollBar::handle:vertical {{
    min-height: 24px;
    background-color: #C3CAD2;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: #9EA8B3;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background-color: transparent;
}}

QScrollBar:horizontal {{
    height: 10px;
    margin: 2px;
    background-color: transparent;
}}

QScrollBar::handle:horizontal {{
    min-width: 24px;
    background-color: #C3CAD2;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: #9EA8B3;
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background-color: transparent;
}}

QToolTip {{
    padding: 5px 7px;
    background-color: #26313C;
    color: #FFFFFF;
    border: 1px solid #26313C;
    border-radius: 4px;
}}
"""


def _preferred_ui_font() -> QFont:
    available_families = set(QFontDatabase.families())
    candidates = (
        "Microsoft YaHei UI",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Segoe UI Variable Text",
        "Segoe UI",
    )
    family = next((name for name in candidates if name in available_families), "")
    font = QFont(family) if family else QFont()
    font.setPointSizeF(9.5)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def _application_palette(app: QApplication) -> QPalette:
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(CANVAS))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_SUBTLE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_DISABLED))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#26313C"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#FFFFFF"))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(TEXT_DISABLED),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(TEXT_DISABLED),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(TEXT_DISABLED),
    )
    return palette


class _OperatorStyle(QProxyStyle):
    """Fusion proxy that keeps form-control arrows crisp under QSS."""

    _ARROWS = {
        QStyle.PrimitiveElement.PE_IndicatorArrowUp,
        QStyle.PrimitiveElement.PE_IndicatorArrowDown,
    }

    def __init__(self) -> None:
        super().__init__("Fusion")
        # Preserve the public style identity expected by diagnostics and tests.
        self.setObjectName("fusion")

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        if element not in self._ARROWS:
            super().drawPrimitive(element, option, painter, widget)
            return

        self._draw_arrow(
            painter,
            option.rect,
            points_up=element == QStyle.PrimitiveElement.PE_IndicatorArrowUp,
            enabled=bool(option.state & QStyle.StateFlag.State_Enabled),
        )

    @staticmethod
    def _draw_arrow(
        painter: QPainter,
        rect: QRect,
        *,
        points_up: bool,
        enabled: bool,
    ) -> None:
        center = rect.center()
        half_width = max(3.0, min(4.0, rect.width() / 2.5))
        half_height = max(2.0, min(3.0, rect.height() / 2.5))
        if points_up:
            points = (
                QPointF(center.x() - half_width, center.y() + half_height),
                QPointF(center.x() + half_width, center.y() + half_height),
                QPointF(center.x(), center.y() - half_height),
            )
        else:
            points = (
                QPointF(center.x() - half_width, center.y() - half_height),
                QPointF(center.x() + half_width, center.y() - half_height),
                QPointF(center.x(), center.y() + half_height),
            )

        color = QColor(TEXT_SECONDARY if enabled else TEXT_DISABLED)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF(points))
        painter.restore()

    def drawComplexControl(
        self,
        control: QStyle.ComplexControl,
        option: QStyleOptionComplex,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        super().drawComplexControl(control, option, painter, widget)
        rect = option.rect
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        if control == QStyle.ComplexControl.CC_ComboBox:
            arrow_rect = QRect(rect.right() - 27, rect.top(), 28, rect.height())
            self._draw_arrow(painter, arrow_rect, points_up=False, enabled=enabled)
        elif control == QStyle.ComplexControl.CC_SpinBox:
            button_height = max(1, rect.height() // 2)
            up_rect = QRect(rect.right() - 21, rect.top(), 22, button_height)
            down_rect = QRect(
                rect.right() - 21,
                rect.top() + button_height,
                22,
                rect.height() - button_height,
            )
            self._draw_arrow(painter, up_rect, points_up=True, enabled=enabled)
            self._draw_arrow(painter, down_rect, points_up=False, enabled=enabled)

    def sizeFromContents(
        self,
        contents_type: QStyle.ContentsType,
        option: QStyleOption,
        size: QSize,
        widget: QWidget | None = None,
    ) -> QSize:
        resolved = super().sizeFromContents(contents_type, option, size, widget)
        if contents_type == QStyle.ContentsType.CT_SpinBox:
            resolved.setHeight(max(30, resolved.height()))
        return resolved


def configure_application(app: QApplication) -> None:
    """Apply the cross-platform operator-station theme to an application."""

    app.setStyle(_OperatorStyle())
    app.setFont(_preferred_ui_font())
    app.setPalette(_application_palette(app))
    app.setStyleSheet(APP_STYLESHEET)


def set_widget_role(widget: QWidget, role: str) -> None:
    """Set a QSS role and immediately refresh the widget's computed style."""

    aliases = {
        "runtime_log": "runtimeLog",
        "action_output": "actionOutput",
    }
    normalized = aliases.get(str(role).strip(), str(role).strip())
    widget.setProperty("role", normalized)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_standard_icon(
    button: QAbstractButton,
    standard_pixmap: QStyle.StandardPixmap,
) -> None:
    """Apply a platform-native 16 px command icon to a button."""

    button.setIcon(button.style().standardIcon(standard_pixmap))
    button.setIconSize(QSize(16, 16))


def status_indicator_style(color: str) -> str:
    """Return a compact status-label style using a semantic color marker."""

    resolved = QColor(str(color).strip())
    if not resolved.isValid():
        resolved = QColor(TEXT_SECONDARY)
    red, green, blue, _alpha = resolved.getRgb()
    foreground = resolved.name(QColor.NameFormat.HexRgb)
    return (
        "QLabel {"
        f" color: {foreground};"
        f" background-color: rgba({red}, {green}, {blue}, 20);"
        " min-height: 22px;"
        " padding: 1px 7px;"
        " border: 0;"
        f" border-left: 3px solid {foreground};"
        " border-radius: 3px;"
        " font-weight: 600;"
        "}"
    )
