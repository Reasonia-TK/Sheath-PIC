from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

LIGHT = dict(
    bg="#FAF9F5",
    surface="#F0EEE6",
    surface_2="#E8E6DC",
    border="#E0DDD4",
    text="#141413",
    text_muted="#6B6A65",
    text_disabled="#A8A69E",
    accent="#C96442",
    accent_strong="#B4532F",
    accent_hover="#A84F31",
    accent_fg="#FFFFFF",
    accent_text="#A84F31",
    selection="#C9644226",
    focus="#C96442",
    success="#2F7D4F",
    warning="#9A6700",
    error="#B42318",
    info="#3B6EA8",
)
DARK = dict(
    bg="#262624",
    surface="#30302E",
    surface_2="#3A3A37",
    border="#45453F",
    text="#F5F4EE",
    text_muted="#A6A39A",
    text_disabled="#6B6A65",
    accent="#D97757",
    accent_strong="#D97757",
    accent_hover="#E08A6A",
    accent_fg="#1F1E1D",
    accent_text="#E08A6A",
    selection="#D9775733",
    focus="#D97757",
    success="#6FBF8B",
    warning="#E3B341",
    error="#F07167",
    info="#7AA7D9",
)
SERIES = {
    "light": ["#C96442", "#3B6EA8", "#2F7D4F", "#9A6700", "#7B5EA7", "#5F7A7A"],
    "dark": ["#D97757", "#7AA7D9", "#6FBF8B", "#E3B341", "#B39DDB", "#9DB8B8"],
}

_fonts_ready = False


def setup_fonts(app) -> str:
    global _fonts_ready
    if not _fonts_ready and os.name == "nt":
        folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "segoeuib.ttf", "YuGothM.ttc", "YuGothB.ttc"):
            path = folder / name
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        _fonts_ready = True
    family = "Yu Gothic UI" if "Yu Gothic UI" in QFontDatabase.families() else app.font().family()
    app.setFont(QFont(family, 10))
    return family


def resolved_mode(app, requested: str) -> str:
    if requested in {"light", "dark"}:
        return requested
    return "dark" if app.styleHints().colorScheme() == Qt.ColorScheme.Dark else "light"


def apply_theme(app, requested: str) -> tuple[str, dict]:
    family = setup_fonts(app)
    mode = resolved_mode(app, requested)
    t = DARK if mode == "dark" else LIGHT
    app.setStyle("Fusion")
    palette = QPalette()
    role = QPalette.ColorRole
    for r, key in (
        (role.Window, "bg"),
        (role.Base, "surface_2"),
        (role.AlternateBase, "surface"),
        (role.WindowText, "text"),
        (role.Text, "text"),
        (role.Button, "surface"),
        (role.ButtonText, "text"),
        (role.Highlight, "accent"),
        (role.HighlightedText, "accent_fg"),
        (role.ToolTipBase, "surface"),
        (role.ToolTipText, "text"),
        (role.PlaceholderText, "text_muted"),
        (role.Link, "accent_text"),
    ):
        palette.setColor(r, QColor(t[key]))
    for r in (role.Text, role.ButtonText, role.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, r, QColor(t["text_disabled"]))
    app.setPalette(palette)
    app.setStyleSheet(f"""
      QWidget {{ color:{t["text"]}; font-family:'{family}'; font-size:13px; }}
      QMainWindow, QDialog {{ background:{t["bg"]}; }}
      QLabel#Title {{ font-size:24px; font-weight:600; }}
      QLabel#Subtitle, QLabel[muted="true"] {{ color:{t["text_muted"]}; }}
      QLabel#Status {{ background:{t["surface"]}; border:1px solid {t["border"]};
                        border-radius:6px; padding:8px 12px; }}
      QGroupBox {{ border:1px solid {t["border"]}; border-radius:8px; margin-top:12px;
                   padding:12px 8px 8px; }}
      QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 5px; }}
      QTabWidget::pane {{ border:1px solid {t["border"]}; border-radius:6px; }}
      QTabBar::tab {{ background:{t["surface"]}; padding:9px 13px;
        border-bottom:2px solid {t["border"]}; }}
      QTabBar::tab:selected {{ background:{t["bg"]}; border-bottom:2px solid {t["accent"]}; }}
      QTabBar::tab:hover {{ background:{t["surface_2"]}; }}
      QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background:{t["surface_2"]};
        border:1px solid {t["border"]}; border-radius:4px; padding:5px 6px; min-height:20px; }}
      QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border:2px solid {t["focus"]}; }}
      QWidget:disabled {{ color:{t["text_disabled"]}; }}
      QAbstractItemView {{ background:{t["surface"]}; border:1px solid {t["border"]};
        selection-background-color:{t["accent"]}; selection-color:{t["accent_fg"]}; }}
      QPushButton {{ background:{t["surface"]}; border:1px solid {t["border"]};
        border-radius:5px; padding:7px 12px; min-height:20px; }}
      QPushButton:hover {{ background:{t["surface_2"]}; }}
      QPushButton:disabled {{ color:{t["text_disabled"]}; }}
      QPushButton[primary="true"] {{ background:{t["accent_strong"]}; color:{t["accent_fg"]};
        border:none; font-weight:600; }}
      QPushButton[primary="true"]:hover {{ background:{t["accent_hover"]}; }}
      QPushButton[primary="true"]:disabled {{ background:{t["surface_2"]};
        color:{t["text_disabled"]}; }}
      QPlainTextEdit, QTextBrowser {{ background:{t["bg"]}; border:1px solid {t["border"]};
        border-radius:5px; padding:6px; }}
      QSplitter::handle {{ background:{t["border"]}; }}
      QProgressBar {{ border:1px solid {t["border"]}; border-radius:4px; background:{t["surface"]};
        min-height:18px; text-align:center; }}
      QProgressBar::chunk {{ background:{t["accent"]}; border-radius:3px; }}
      QScrollArea {{ border:none; background:{t["bg"]}; }}
      QScrollBar:vertical {{ background:{t["surface"]}; width:10px; margin:0; }}
      QScrollBar::handle:vertical {{ background:{t["border"]};
        border-radius:4px; min-height:24px; }}
      QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
      QToolTip {{ background:{t["surface"]}; color:{t["text"]}; border:1px solid {t["border"]}; }}
    """)
    return mode, t


def style_figure(figure, mode: str) -> None:
    t = DARK if mode == "dark" else LIGHT
    figure.set_facecolor(t["bg"])
    for ax in figure.axes:
        ax.set_facecolor(t["bg"])
        for spine in ax.spines.values():
            spine.set_color(t["border"])
        ax.tick_params(colors=t["text_muted"], labelsize=9)
        ax.xaxis.label.set_color(t["text"])
        ax.yaxis.label.set_color(t["text"])
        ax.title.set_color(t["text"])
        for item in ax.texts:
            item.set_color(t["text_muted"])
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(t["surface"])
            legend.get_frame().set_edgecolor(t["border"])
            for label in legend.get_texts():
                label.set_color(t["text"])
