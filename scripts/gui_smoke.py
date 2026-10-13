"""Render both UI themes and optionally exercise its actual QProcess worker."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QToolTip

from sheath_pic.gui import MainWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("artifacts/quick"))
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    original_theme = window.theme_combo.currentData()
    errors = []
    window.error = errors.append
    window.resize(1450, 960)
    window.show()
    if args.run:
        window.quick_settings()
        window.allow_unconverged.setChecked(True)
        window.start_run("all")
        deadline = time.monotonic() + 180
        while window.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if window.process is not None:
            window.cancel_run()
            raise RuntimeError("GUI worker timeout")
        assert window.field is not None and window.particles is not None, errors
        print("GUI worker finished:", window.current_directory, flush=True)
    else:
        window.load_result_directory(args.results)
    out = Path("artifacts/gui")
    out.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        window.theme_combo.setCurrentIndex(window.theme_combo.findData(theme))
        window.phase_slider.setValue(len(window.field.phase_s) // 2)
        for index, name in (
            (0, "inputs"),
            (1, "transport"),
            (2, "field"),
            (3, "particles"),
            (4, "diagnostics"),
        ):
            window.views.setCurrentIndex(index)
            app.processEvents()
            window.grab().save(str(out / f"{theme}-{name}.png"))
        window.settings_tabs.setCurrentIndex(1)
        window.controls["density"].setEnabled(False)
        window.controls["density_expression"].setFocus()
        window.views.setCurrentIndex(0)
        app.processEvents()
        window.grab().save(str(out / f"{theme}-controls.png"))
        QToolTip.showText(window.mapToGlobal(QPoint(200, 180)), "入力単位：r,zはm / 密度はm⁻³")
        app.processEvents()
        for widget in app.topLevelWidgets():
            if widget.metaObject().className() == "QTipLabel":
                widget.grab().save(str(out / f"{theme}-tooltip.png"))
        QToolTip.hideText()
        window.controls["density"].setEnabled(True)
        window.settings_tabs.setCurrentIndex(0)
    assert not errors, errors
    window.theme_combo.setCurrentIndex(window.theme_combo.findData(original_theme))
    window.close()
    print("Screenshots:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
