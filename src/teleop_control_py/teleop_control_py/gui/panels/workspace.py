"""Reusable, ROS-independent workspace shell for the main window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..theme import set_standard_icon, set_widget_role


class WorkspaceShell(QWidget):
    """Provide a compact status header above two independently scrolling columns."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceShell")
        self.setProperty("role", "workspace")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top_bar = QWidget(self)
        self.top_bar = top_bar
        top_bar.setObjectName("workspaceTopBar")
        top_bar.setProperty("role", "statusBar")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(12, 7, 12, 7)
        top_bar_layout.setSpacing(10)

        product_label = QLabel("Teleop Control", top_bar)
        product_label.setObjectName("workspaceProductName")
        product_label.setProperty("role", "title")
        product_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        top_bar_layout.addWidget(product_label)
        top_bar_layout.addStretch(1)

        self.preview_button = QPushButton("实时预览", top_bar)
        self.preview_button.setObjectName("workspacePreviewButton")
        self.preview_button.setToolTip("打开实时预览与状态窗口")
        self.preview_button.setFixedHeight(30)
        set_widget_role(self.preview_button, "secondary")
        set_standard_icon(
            self.preview_button,
            QStyle.StandardPixmap.SP_ComputerIcon,
        )
        top_bar_layout.addWidget(self.preview_button)

        self.phase_label = self._make_status_label(
            "阶段: IDLE",
            object_name="workspacePhaseStatus",
            minimum_width=112,
        )
        self.runtime_label = self._make_status_label(
            "ROS: 待机",
            object_name="workspaceRuntimeStatus",
            minimum_width=140,
        )
        self.preview_label = self._make_status_label(
            "预览源: 关闭",
            object_name="workspacePreviewStatus",
            minimum_width=120,
        )
        top_bar_layout.addWidget(self.phase_label)
        top_bar_layout.addWidget(self.runtime_label)
        top_bar_layout.addWidget(self.preview_label)
        layout.addWidget(top_bar)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.setProperty("role", "workspaceColumns")
        self.splitter.setChildrenCollapsible(False)

        self.left_scroll, self.left_content, self.left_layout = self._make_column(
            "workspaceLeftScroll",
            "workspaceLeftContent",
        )
        self.right_scroll, self.right_content, self.right_layout = self._make_column(
            "workspaceRightScroll",
            "workspaceRightContent",
        )
        self.splitter.addWidget(self.left_scroll)
        self.splitter.addWidget(self.right_scroll)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 48)
        self.splitter.setStretchFactor(1, 52)
        self.splitter.setSizes([480, 520])
        layout.addWidget(self.splitter, 1)

    @staticmethod
    def _make_status_label(
        text: str,
        *,
        object_name: str,
        minimum_width: int,
    ) -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setProperty("role", "status")
        label.setProperty("status", "idle")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumWidth(minimum_width)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return label

    @staticmethod
    def _make_column(
        scroll_object_name: str,
        content_object_name: str,
    ) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        content = QWidget()
        content.setObjectName(content_object_name)
        content.setProperty("role", "workspaceColumn")
        content.setAttribute(Qt.WA_StyledBackground, True)

        column_layout = QVBoxLayout(content)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setObjectName(scroll_object_name)
        scroll.setProperty("role", "workspaceScrollArea")
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.viewport().setAutoFillBackground(False)
        return scroll, content, column_layout
