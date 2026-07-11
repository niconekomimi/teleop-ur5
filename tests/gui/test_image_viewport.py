from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QPixmap

from teleop_control_py.gui.widgets.image_viewport import ImageViewport


def test_large_frame_does_not_change_viewport_size_hints(qtbot) -> None:
    viewport = ImageViewport(
        "no frame",
        preferred_size=QSize(640, 360),
        minimum_size=QSize(160, 120),
    )
    qtbot.addWidget(viewport)
    viewport.resize(900, 600)
    viewport.show()

    frame = QPixmap(1920, 1080)
    frame.fill(QColor("#2563EB"))
    viewport.set_frame_pixmap(frame)

    assert viewport.minimumSizeHint() == QSize(160, 120)
    assert viewport.sizeHint() == QSize(640, 360)
    assert viewport.source_pixmap().size() == QSize(1920, 1080)
    assert viewport.pixmap().size().width() <= viewport.width()
    assert viewport.pixmap().size().height() <= viewport.height()


def test_clear_frame_restores_empty_text(qtbot) -> None:
    viewport = ImageViewport("waiting")
    qtbot.addWidget(viewport)
    frame = QPixmap(320, 240)
    frame.fill(QColor("#16794B"))

    viewport.set_frame_pixmap(frame)
    viewport.clear_frame("disconnected")

    assert viewport.source_pixmap().isNull()
    assert viewport.pixmap().isNull()
    assert viewport.text() == "disconnected"
