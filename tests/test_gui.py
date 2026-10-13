import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))

from PySide6.QtWidgets import QApplication

from sheath_pic.config import Project
from sheath_pic.gui import CsvMappingDialog, MainWindow


def test_gui_preserves_loaded_numeric_precision_and_themes(tmp_path):
    app = QApplication.instance() or QApplication([])
    workspace = Path(__file__).resolve().parents[1]
    window = MainWindow(workspace)
    project = Project(base_dir=str(workspace))
    project.geometry.chamber_radius_m = 0.278901234567
    project.plasma.density_m3 = 1.234567891234e16
    window.project = project
    window._populate(project)
    assert window.collect_project().geometry.chamber_radius_m == project.geometry.chamber_radius_m
    assert window.collect_project().plasma.density_m3 == project.plasma.density_m3
    for theme in ("light", "dark", "system"):
        window.theme_combo.setCurrentIndex(window.theme_combo.findData(theme))
        app.processEvents()
        assert window.mode in {"light", "dark"}
    path = tmp_path / "wave.csv"
    path.write_text("time_us,voltage_kV\n0,0\n1.25,-10\n2.5,0\n")
    dialog = CsvMappingDialog(path, window)
    assert dialog.time_column.currentText() == "time_us"
    assert dialog.voltage_column.currentText() == "voltage_kV"
    dialog.close()
    window.close()
