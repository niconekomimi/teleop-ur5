"""Stable image viewport shared by camera-oriented GUI surfaces."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class ImageViewport(QLabel):
    """Display a scaled pixmap without letting it control layout size hints."""

    def __init__(
        self,
        empty_text: str,
        *,
        preferred_size: QSize = QSize(480, 320),
        minimum_size: QSize = QSize(160, 120),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(str(empty_text), parent)
        self._empty_text = str(empty_text)
        self._preferred_size = QSize(preferred_size)
        self._minimum_size = QSize(minimum_size)
        self._source_pixmap = QPixmap()

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.setMinimumSize(self._minimum_size)
        self.setProperty("role", "image-viewport")

    def source_pixmap(self) -> QPixmap:
        return QPixmap(self._source_pixmap)

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = QPixmap(pixmap)
        self._refresh()

    def clear_frame(self, text: str | None = None) -> None:
        self._source_pixmap = QPixmap()
        self.clear()
        self.setText(self._empty_text if text is None else str(text))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._preferred_size)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._minimum_size)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source_pixmap.isNull():
            return
        if self.width() <= 1 or self.height() <= 1:
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)
