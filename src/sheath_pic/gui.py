from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import Project
from .inputs import Waveform, read_numeric_csv
from .mesh import AxisymmetricMesh
from .storage import (
    export_distributions,
    load_field,
    load_particles,
    plasma_observables,
    write_json,
)
from .theme import SERIES, apply_theme, style_figure
from .transport import TransportTable
from .worker import make_waveform


def _canvas():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6), constrained_layout=True)
    canvas = FigureCanvasQTAgg(figure)
    return figure, canvas


class CsvMappingDialog(QDialog):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("実験波形の列と単位")
        headers, values = read_numeric_csv(path)
        layout = QVBoxLayout(self)
        label = QLabel(f"{path.name}\n{len(values):,}行・{len(headers)}列。電圧は接地基準です。")
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        self.time_column, self.voltage_column = QComboBox(), QComboBox()
        self.time_column.addItems(headers)
        self.voltage_column.addItems(headers)
        self.voltage_column.setCurrentIndex(min(1, len(headers) - 1))
        self.time_unit, self.voltage_unit = QComboBox(), QComboBox()
        for text, value in (("s", 1), ("µs", 1e-6), ("ns", 1e-9), ("ms", 1e-3)):
            self.time_unit.addItem(text, value)
        for text, value in (("V", 1), ("kV", 1000), ("mV", 0.001)):
            self.voltage_unit.addItem(text, value)
        form.addRow("時刻列", self.time_column)
        form.addRow("時刻単位", self.time_unit)
        form.addRow("電圧列", self.voltage_column)
        form.addRow("電圧単位", self.voltage_unit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self, workspace: Path | None = None):
        super().__init__()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.project = Project(base_dir=str(self.workspace))
        self.settings_path = self.workspace / ".cache" / "ui-settings.json"
        self.theme_request = "system"
        if self.settings_path.exists():
            try:
                self.theme_request = json.loads(self.settings_path.read_text()).get(
                    "theme", "system"
                )
            except (ValueError, OSError):
                pass
        self.mode = "light"
        self.table = None
        self.field = None
        self.particles = None
        self.result_directory = None
        self.field_path = None
        self.current_directory = None
        self.stdout_buffer = b""
        self.process = None
        self.controls = {}
        self.setWindowTitle("Sheath-PIC · Ar RF sheath")
        self.resize(1450, 960)
        self.setMinimumSize(1120, 760)
        self._build()
        self._populate(self.project)
        self.theme_changed()
        self.preview_inputs()
        QApplication.instance().styleHints().colorSchemeChanged.connect(self.system_theme_changed)

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 14)
        header = QHBoxLayout()
        title = QLabel("Sheath-PIC")
        title.setObjectName("Title")
        header.addWidget(title)
        subtitle = QLabel("実測電位 → 軸対称RF場 → イオンのエネルギー・入射角")
        subtitle.setObjectName("Subtitle")
        header.addWidget(subtitle)
        header.addStretch()
        self.theme_combo = QComboBox()
        for text, value in (
            ("テーマ：OSに追従", "system"),
            ("ライト", "light"),
            ("ダーク", "dark"),
        ):
            self.theme_combo.addItem(text, value)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(self.theme_request)))
        self.theme_combo.currentIndexChanged.connect(self.theme_changed)
        header.addWidget(self.theme_combo)
        layout.addLayout(header)
        toolbar = QHBoxLayout()
        for label, callback in (
            ("波形CSVを読み込む", self.import_waveform),
            ("合成デモ波形", self.use_demo),
            ("条件を保存", self.save_project),
            ("条件を開く", self.open_project),
            ("計算結果を開く", self.open_result),
            ("図を保存", self.save_figure),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.status_label = QLabel()
        self.status_label.setObjectName("Status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        self.settings_tabs = QTabWidget()
        left_layout.addWidget(self.settings_tabs)
        self._geometry_form()
        self._plasma_form()
        self._numerics_form()
        self._transport_form()
        preview = QPushButton("入力と密度分布をプレビュー")
        preview.clicked.connect(self.preview_inputs)
        left_layout.addWidget(preview)
        action_group = QGroupBox("計算")
        actions = QVBoxLayout(action_group)
        self.action_buttons = []
        for label, action in (
            ("1  電子物性（Boltzpmp DC）", "transport"),
            ("2  2次元RF場を計算", "field"),
            ("3  IEDF / IADF / IAEDF", "particles"),
            ("すべて計算", "all"),
        ):
            button = QPushButton(label)
            button.setProperty("primary", action == "all")
            button.clicked.connect(lambda checked=False, a=action: self.start_run(a))
            actions.addWidget(button)
            self.action_buttons.append(button)
        self.allow_unconverged = QCheckBox("未収束場で粒子計算を許可（探索用）")
        self.allow_unconverged.setToolTip("許可した結果には未収束場を使用したことを記録します。")
        actions.addWidget(self.allow_unconverged)
        run_controls = QHBoxLayout()
        self.stop_button = QPushButton("中断")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_run)
        self.resume_button = QPushButton("場計算を再開")
        self.resume_button.clicked.connect(self.resume_run)
        self.resume_button.setEnabled(False)
        run_controls.addWidget(self.stop_button)
        run_controls.addWidget(self.resume_button)
        actions.addLayout(run_controls)
        left_layout.addWidget(action_group)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        left_layout.addWidget(self.progress)
        self.progress_text = QLabel("計算待機中")
        self.progress_text.setWordWrap(True)
        self.progress_text.setProperty("muted", True)
        left_layout.addWidget(self.progress_text)
        splitter.addWidget(left)
        self.views = QTabWidget()
        self.figures = {}
        for label, key in (
            ("入力", "inputs"),
            ("電子物性", "transport"),
            ("電位・シース", "field"),
            ("IEDF / IADF", "particles"),
        ):
            widget = QWidget()
            box = QVBoxLayout(widget)
            box.setContentsMargins(10, 10, 10, 10)
            if key == "field":
                tools = QHBoxLayout()
                self.quantity_combo = QComboBox()
                for text, value in (
                    ("電位 [kV]", "potential_v"),
                    ("|E| [kV/m]", "electric_magnitude"),
                    ("Er [kV/m]", "electric_r_vm"),
                    ("Ez [kV/m]", "electric_z_vm"),
                    ("log10 ne [m⁻³]", "electron_density_m3"),
                    ("log10 ni [m⁻³]", "ion_density_m3"),
                    ("平均電子エネルギー [eV]", "mean_energy_ev"),
                ):
                    self.quantity_combo.addItem(text, value)
                self.quantity_combo.currentIndexChanged.connect(self.draw_field)
                tools.addWidget(self.quantity_combo)
                self.phase_slider = QSlider(Qt.Orientation.Horizontal)
                self.phase_slider.setRange(0, 0)
                self.phase_slider.valueChanged.connect(self.draw_field)
                tools.addWidget(self.phase_slider, 1)
                self.phase_label = QLabel("位相 —")
                tools.addWidget(self.phase_label)
                box.addLayout(tools)
            if key == "particles":
                tools = QHBoxLayout()
                self.radial_combo = QComboBox()
                self.radial_combo.addItem("Wafer全体")
                self.radial_combo.currentIndexChanged.connect(self.draw_particles)
                tools.addWidget(self.radial_combo)
                self.particle_summary = QLabel("場計算後にイオンを追跡します。")
                self.particle_summary.setWordWrap(True)
                tools.addWidget(self.particle_summary, 1)
                export = QPushButton("分布CSVを保存")
                export.clicked.connect(self.export_csv)
                tools.addWidget(export)
                box.addLayout(tools)
            figure, canvas = _canvas()
            self.figures[key] = (figure, canvas)
            box.addWidget(canvas, 1)
            self.views.addTab(widget, label)
        self.diagnostic_text = QTextBrowser()
        self.diagnostic_text.setPlainText(
            "場計算後、周期収束・保存則・近似の適用範囲を表示します。"
        )
        self.views.addTab(self.diagnostic_text, "診断")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.views.addTab(self.log, "ログ")
        splitter.addWidget(self.views)
        splitter.setSizes([340, 1060])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        note = QLabel(
            "物理モデルの初版です。定量利用には格子・時間刻み・粒子数の収束確認と実験照合が必要です。"
        )
        note.setProperty("muted", True)
        note.setWordWrap(True)
        layout.addWidget(note)

    def _form(self, title):
        body = QWidget()
        form = QFormLayout(body)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setVerticalSpacing(9)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        self.settings_tabs.addTab(scroll, title)
        return form

    def _number(self, form, key, label, minimum, maximum, decimals=3):
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setKeyboardTracking(False)
        control.setSingleStep(1)
        form.addRow(label, control)
        self.controls[key] = control
        return control

    def _integer(self, form, key, label, minimum, maximum):
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setKeyboardTracking(False)
        form.addRow(label, control)
        self.controls[key] = control
        return control

    def _text(self, form, key, label):
        control = QLineEdit()
        form.addRow(label, control)
        self.controls[key] = control
        return control

    def _choice(self, form, key, label, options):
        control = QComboBox()
        control.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        control.setMinimumContentsLength(8)
        for text, value in options:
            control.addItem(text, value)
        control.currentTextChanged.connect(control.setToolTip)
        form.addRow(label, control)
        self.controls[key] = control
        return control

    def _geometry_form(self):
        form = self._form("装置")
        self._number(form, "cathode_diameter", "Wafer径 [mm]", 1, 1000)
        self._number(form, "chamber_diameter", "内径 [mm]", 1, 2000)
        self._number(form, "gap", "電極間 [mm]", 0.1, 1000)
        self._number(form, "pressure", "Ar圧力 [mTorr]", 0.001, 10000)
        self._number(form, "temperature", "ガス温度 [K]", 1, 3000, 1)
        self._number(form, "frequency", "周波数 [kHz]", 0.01, 100000)
        self._number(form, "peak", "デモ負側ピーク [kV]", 0.001, 1000)
        info = QLabel(
            "対向電極・側壁：接地 / 表面：Si\n"
            "初期内径578.233 mmは、上面＋高さ50 mmの側壁で面積比5とする仮定値です。\n"
            "底面の駆動電極外周：絶縁・反射境界。"
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        form.addRow(info)

    def _plasma_form(self):
        form = self._form("密度")
        self._text(form, "density", "基準密度 [m⁻³]")
        self._choice(
            form,
            "density_profile",
            "分布",
            [("一様", "uniform"), ("数式", "expression"), ("CSV", "csv")],
        )
        self._text(form, "density_expression", "密度式")
        self.controls["density_expression"].setToolTip(
            "r,z,R,Hはm。ne0は基準密度。sin, cos, exp, sqrt, abs, minimum, maximumが使用できます。",
        )
        density_csv = self._text(form, "density_csv", "密度CSV")
        density_csv.setPlaceholderText("r_mm,z_mm,ne_m3")
        browse = QPushButton("密度CSVを選択")
        browse.clicked.connect(self.import_density)
        form.addRow(browse)
        self._choice(
            form,
            "density_mode",
            "密度の扱い",
            [
                ("自己整合：初期分布として使用", "self_consistent"),
                ("外部供給で目標分布へ緩和", "maintained"),
            ],
        )
        self._number(form, "density_relaxation", "維持緩和時間 [周期]", 0.1, 1000, 1)
        self._number(form, "electron_energy", "初期平均ε [eV]", 0.04, 10000)
        self._number(form, "source_energy", "供給平均ε [eV]", 0.04, 10000)
        self._number(form, "gamma_c", "Wafer二次電子係数", 0, 1, 4)
        self._number(form, "gamma_g", "接地面二次電子係数", 0, 1, 4)
        self._number(form, "sticking", "電子吸収率", 0.001, 1, 3)
        self._choice(
            form,
            "ion_energy_frame",
            "イオン断面積の軸",
            [
                ("重心系エネルギー（要確認）", "center_of_mass"),
                ("実験室系エネルギー", "lab"),
            ],
        )
        info = QLabel(
            "二次電子係数の0は未指定時の比較条件です。Siの測定値ではありません。\n"
            "二次電子は流体源として近似し、ビームの非局所輸送は解いていません。"
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        form.addRow(info)

    def _numerics_form(self):
        form = self._form("計算")
        for key, label, low, high in (
            ("nr", "rセル数", 4, 160),
            ("nz", "zセル数", 4, 240),
            ("min_cycles", "最小周期数", 1, 200),
            ("max_cycles", "最大周期数", 1, 200),
            ("steps", "最小時間分割 / 周期", 16, 4096),
            ("phases", "保存位相数", 8, 4096),
            ("particles", "テスト粒子数", 100, 2000000),
            ("particle_periods", "最大追跡 [周期]", 1, 1000),
            ("radial_bins", "Wafer半径ビン数", 2, 100),
            ("energy_bins", "エネルギービン数", 2, 1000),
            ("angle_bins", "角度ビン数", 2, 360),
            ("seed", "乱数seed", 0, 2147483647),
        ):
            self._integer(form, key, label, low, high)
        self._number(form, "tolerance", "周期収束許容値", 0.00001, 1, 5)
        quick = QPushButton("粗い格子の動作確認設定")
        quick.clicked.connect(self.quick_settings)
        form.addRow(quick)
        info = QLabel(
            "時間刻みはイオンCFLと加速・反応時間から自動縮小します。\n"
            "動作確認設定は精度確認用ではありません。"
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        form.addRow(info)

    def _transport_form(self):
        form = self._form("物性")
        self._text(form, "electron_xsec", "電子断面積")
        self._text(form, "ion_xsec", "イオン断面積")
        self._number(form, "en_min", "最小E/N [Td]", 0.00001, 1e6, 5)
        self._number(form, "en_max", "最大E/N [Td]", 0.0001, 1e7, 3)
        self._integer(form, "en_points", "E/N計算点数", 4, 256)
        self._number(form, "energy_max", "初期ε格子上限 [eV]", 21, 10000, 1)
        self._integer(form, "angle_cells", "電子の角度分割", 8, 180)
        info = QLabel(
            "BoltzpmpはDC計算のみ。EEDF端が十分小さくなるまで格子上限を拡張します。\n"
            "断面積範囲内で収束しない計算点はエラーにします。\n"
            "拡散・エネルギー輸送にはMaxwellモーメント近似を使用します。"
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        form.addRow(info)

    def _populate(self, project):
        g, p, n, t = project.geometry, project.plasma, project.numerics, project.transport
        values = {
            "cathode_diameter": 2000 * g.cathode_radius_m,
            "chamber_diameter": 2000 * g.chamber_radius_m,
            "gap": g.gap_m * 1000,
            "pressure": p.pressure_pa / 0.13332236842105263,
            "temperature": p.gas_temperature_k,
            "frequency": n.frequency_hz / 1000,
            "peak": n.demo_negative_peak_v / 1000,
            "density": str(p.density_m3),
            "density_profile": p.density_profile,
            "density_expression": p.density_expression,
            "density_csv": p.density_csv,
            "density_mode": p.density_mode,
            "density_relaxation": p.density_relaxation_periods,
            "electron_energy": p.initial_mean_energy_ev,
            "source_energy": p.source_mean_energy_ev,
            "gamma_c": p.cathode_secondary_yield,
            "gamma_g": p.ground_secondary_yield,
            "sticking": p.electron_sticking,
            "ion_energy_frame": p.ion_energy_frame,
            "nr": g.radial_cells,
            "nz": g.axial_cells,
            "min_cycles": n.minimum_cycles,
            "max_cycles": n.maximum_cycles,
            "steps": n.steps_per_period,
            "phases": n.stored_phases,
            "particles": n.particles,
            "particle_periods": n.particle_max_periods,
            "radial_bins": n.radial_bins,
            "energy_bins": n.energy_bins,
            "angle_bins": n.angle_bins,
            "seed": n.seed,
            "tolerance": n.periodic_tolerance,
            "electron_xsec": project.electron_xsec,
            "ion_xsec": project.ion_xsec,
            "en_min": t.en_min_td,
            "en_max": t.en_max_td,
            "en_points": t.en_points,
            "energy_max": t.energy_max_ev,
            "angle_cells": t.angle_cells,
        }
        for key, value in values.items():
            widget = self.controls[key]
            if isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findData(value))
            else:
                widget.setValue(value)
        self.original_control_values = values
        self.displayed_control_values = {
            key: widget.value()
            for key, widget in self.controls.items()
            if isinstance(widget, (QDoubleSpinBox, QSpinBox))
        }

    def collect_project(self) -> Project:
        project = Project.from_dict(self.project.to_dict())
        g, p, n, t = project.geometry, project.plasma, project.numerics, project.transport

        def val(key):
            current = self.controls[key].value()
            if current == self.displayed_control_values[key]:
                return self.original_control_values[key]
            return current

        def text(key):
            return self.controls[key].text().strip()

        def choice(key):
            return self.controls[key].currentData()

        g.cathode_radius_m = val("cathode_diameter") / 2000
        g.chamber_radius_m = val("chamber_diameter") / 2000
        g.gap_m = val("gap") / 1000
        g.radial_cells, g.axial_cells = val("nr"), val("nz")
        p.pressure_pa = val("pressure") * 0.13332236842105263
        p.gas_temperature_k = val("temperature")
        p.density_m3 = float(text("density"))
        p.density_profile, p.density_mode = choice("density_profile"), choice("density_mode")
        p.density_expression, p.density_csv = text("density_expression"), text("density_csv")
        p.density_relaxation_periods = val("density_relaxation")
        p.initial_mean_energy_ev, p.source_mean_energy_ev = (
            val("electron_energy"),
            val("source_energy"),
        )
        p.cathode_secondary_yield, p.ground_secondary_yield = val("gamma_c"), val("gamma_g")
        p.electron_sticking, p.ion_energy_frame = val("sticking"), choice("ion_energy_frame")
        n.frequency_hz, n.demo_negative_peak_v = val("frequency") * 1000, val("peak") * 1000
        for attr, key in (
            ("minimum_cycles", "min_cycles"),
            ("maximum_cycles", "max_cycles"),
            ("steps_per_period", "steps"),
            ("stored_phases", "phases"),
            ("particles", "particles"),
            ("particle_max_periods", "particle_periods"),
            ("radial_bins", "radial_bins"),
            ("energy_bins", "energy_bins"),
            ("angle_bins", "angle_bins"),
            ("seed", "seed"),
            ("periodic_tolerance", "tolerance"),
        ):
            setattr(n, attr, val(key))
        project.electron_xsec, project.ion_xsec = text("electron_xsec"), text("ion_xsec")
        t.en_min_td, t.en_max_td, t.en_points = val("en_min"), val("en_max"), val("en_points")
        t.energy_max_ev, t.angle_cells = val("energy_max"), val("angle_cells")
        project.validate()
        return project

    def theme_changed(self, *_):
        import matplotlib as mpl

        self.theme_request = self.theme_combo.currentData()
        self.mode, _ = apply_theme(QApplication.instance(), self.theme_request)
        mpl.rcParams.update(
            {
                "axes.prop_cycle": mpl.cycler(color=SERIES[self.mode]),
                "font.family": ["Yu Gothic", "Meiryo", "DejaVu Sans"],
            }
        )
        write_json(self.settings_path, {"theme": self.theme_request})
        for figure, canvas in self.figures.values():
            style_figure(figure, self.mode)
            canvas.draw_idle()
        self.draw_transport()
        self.draw_field()
        self.draw_particles()
        if self.table is not None or self.field is not None:
            self.preview_inputs()

    def system_theme_changed(self, *_):
        if self.theme_request == "system":
            self.theme_changed()

    def error(self, message):
        QMessageBox.warning(self, "入力・計算の確認", str(message))

    def preview_inputs(self):
        try:
            project = self.collect_project()
            wave = make_waveform(project)
            mesh = AxisymmetricMesh(project.geometry)
            density = mesh.initial_density(
                project.plasma,
                project.path(project.plasma.density_csv) if project.plasma.density_csv else None,
            )
            figure, canvas = self.figures["inputs"]
            figure.clear()
            a, b = figure.subplots(2, 1)
            time_s = np.linspace(0, wave.period_s, 513)
            # Endpoint is displayed from the original trace rather than wrapping to t=0.
            voltage = np.interp(time_s + wave.time_s[0], wave.time_s, wave.voltage_v)
            a.plot(time_s * 1e6, voltage / 1000, color=SERIES[self.mode][0], linewidth=1.8)
            a.set(
                xlabel="Time [µs]",
                ylabel="Wafer potential [kV]",
                title="Measured waveform"
                if project.numerics.waveform_csv
                else "Synthetic demo waveform",
            )
            image = b.pcolormesh(
                mesh.r_edges * 1000,
                mesh.z_edges * 1000,
                np.log10(np.maximum(density, 1)).T,
                cmap="viridis",
                shading="flat",
            )
            figure.colorbar(image, ax=b, label="log10 initial ne [m⁻³]")
            b.plot(
                [0, project.geometry.cathode_radius_m * 1000],
                [0, 0],
                color=SERIES[self.mode][0],
                linewidth=4,
            )
            b.set(xlabel="r [mm]", ylabel="z [mm]", title="Axisymmetric initial density")
            style_figure(figure, self.mode)
            canvas.draw_idle()
            label = (
                Path(project.numerics.waveform_csv).name
                if project.numerics.waveform_csv
                else "合成デモ波形"
            )
            self.input_summary = (
                f"Ar · {project.numerics.frequency_hz / 1000:g} kHz · "
                f"{project.plasma.pressure_pa / 0.13332236842105263:g} mTorr · "
                f"{label} / 負側 {wave.voltage_v.min() / 1000:.3g} kV / "
                f"接地面積比 {mesh.ground_area / mesh.cathode_area:.3f}"
            )
            self._update_status()
        except Exception as exc:
            self.error(exc)

    def draw_transport(self, *_):
        figure, canvas = self.figures["transport"]
        figure.clear()
        if self.table is None:
            ax = figure.add_subplot()
            ax.text(
                0.5,
                0.5,
                "Boltzpmp DC transport table",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            table = self.table
            a, b, c, d = figure.subplots(2, 2).ravel()
            a.loglog(table.en_td, table.mean_energy_ev, color=SERIES[self.mode][0])
            a.set(xlabel="E/N [Td]", ylabel="Mean energy [eV]", title="DC electron energy")
            b.loglog(table.en_td, table.mobility_m2_vs, color=SERIES[self.mode][1])
            b.set(xlabel="E/N [Td]", ylabel="Mobility [m²/(V s)]", title="Electron mobility")
            for values, name, color in (
                (table.excitation_m3_s, "Excitation", SERIES[self.mode][0]),
                (table.ionization_m3_s, "Ionization", SERIES[self.mode][1]),
            ):
                c.loglog(table.en_td, np.maximum(values, 1e-40), label=name, color=color)
            c.set(
                xlabel="E/N [Td]",
                ylabel="Rate coefficient [m³/s]",
                title="Electron-impact reactions",
            )
            c.legend(fontsize=8)
            for index in np.linspace(0, len(table.en_td) - 1, 4).astype(int):
                d.semilogy(
                    table.eedf_energy_ev,
                    np.maximum(table.eedf[index], 1e-20),
                    label=f"{table.en_td[index]:.2g} Td",
                )
            d.set(
                xlabel="Energy [eV]",
                ylabel="EEDF [eV⁻¹]",
                xlim=(0, min(150, table.eedf_energy_ev[-1])),
                ylim=(1e-12, 10),
                title="EEDF",
            )
            d.legend(fontsize=8)
        style_figure(figure, self.mode)
        canvas.draw_idle()

    def import_waveform(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "実験波形CSV", str(self.workspace), "CSV / TSV (*.csv *.tsv *.txt);;すべて (*)"
        )
        if not path:
            return
        try:
            dialog = CsvMappingDialog(Path(path), self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            project = self.collect_project()
            n = project.numerics
            n.waveform_csv = path
            n.waveform_time_column, n.waveform_voltage_column = (
                dialog.time_column.currentText(),
                dialog.voltage_column.currentText(),
            )
            n.waveform_time_scale, n.waveform_voltage_scale = (
                dialog.time_unit.currentData(),
                dialog.voltage_unit.currentData(),
            )
            make_waveform(project)
            self.project = project
            self.preview_inputs()
            self.views.setCurrentIndex(0)
        except Exception as exc:
            self.error(exc)

    def import_density(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "密度分布CSV", str(self.workspace), "CSV (*.csv)"
        )
        if path:
            self.controls["density_csv"].setText(path)
            self.controls["density_profile"].setCurrentIndex(
                self.controls["density_profile"].findData("csv"),
            )
            self.preview_inputs()

    def use_demo(self):
        self.project.numerics.waveform_csv = ""
        self.project.numerics.waveform_time_column = ""
        self.project.numerics.waveform_voltage_column = ""
        self.preview_inputs()
        self.views.setCurrentIndex(0)

    def quick_settings(self):
        for key, value in {
            "nr": 8,
            "nz": 12,
            "min_cycles": 2,
            "max_cycles": 2,
            "steps": 64,
            "phases": 16,
            "particles": 2000,
            "en_points": 12,
            "angle_cells": 16,
        }.items():
            self.controls[key].setValue(value)
        self.progress_text.setText(
            "粗い格子の動作確認設定。物理精度の検証には格子を細かくしてください。"
        )

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "条件を保存", str(self.workspace / "project.json"), "Project (*.json)"
        )
        if path:
            try:
                project = self.collect_project()
                project.save(Path(path))
                self.project = project
                self.log.appendPlainText(f"条件を保存：{path}")
            except Exception as exc:
                self.error(exc)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "条件を開く", str(self.workspace), "Project (*.json)"
        )
        if path:
            try:
                self.project = Project.load(Path(path))
                self._populate(self.project)
                self.preview_inputs()
            except Exception as exc:
                self.error(exc)

    def open_result(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "場計算結果を開く",
            str(self.workspace / "results"),
            "Field (field.npz);;NPZ (*.npz)",
        )
        if path:
            try:
                self.load_result_directory(Path(path).parent)
            except Exception as exc:
                self.error(exc)

    def load_result_directory(self, directory: Path):
        outputs = {}
        for key in ("field", "transport", "particles"):
            path = directory / f"{key}.npz"
            if path.exists():
                outputs[key] = str(path)
        if "field" not in outputs and "transport" not in outputs:
            raise ValueError("field.npzまたはtransport.npzが必要です。")
        self.load_outputs(outputs)
        self.result_directory = directory
        if self.field is not None:
            self.project = (
                Project.load(directory / "project.json")
                if (directory / "project.json").exists()
                else Project.from_dict(self.field.project)
            )
            self._populate(self.project)
            self.preview_inputs()
        self.resume_button.setEnabled((directory / "checkpoint.npz").exists())

    def load_outputs(self, output):
        if "transport" in output:
            self.table = TransportTable.load(Path(output["transport"]))
            self.draw_transport()
        if "field" in output:
            self.field_path = Path(output["field"])
            self.result_directory = self.field_path.parent
            self.field = load_field(self.field_path)
            self.particles = None
            self.phase_slider.setRange(0, len(self.field.phase_s) - 1)
            self.phase_slider.setValue(0)
            self.draw_field()
            self.draw_particles()
            self._update_status()
        if "particles" in output:
            self.particles = load_particles(Path(output["particles"]))
            self.radial_combo.blockSignals(True)
            self.radial_combo.clear()
            self.radial_combo.addItem("Wafer全体")
            for a, b in zip(
                self.particles.radius_edges_m[:-1], self.particles.radius_edges_m[1:], strict=True
            ):
                self.radial_combo.addItem(f"r = {a * 1000:.1f}–{b * 1000:.1f} mm")
            self.radial_combo.blockSignals(False)
            self.draw_particles()
            self._update_status()
            self.views.setCurrentIndex(3)
        elif "field" in output:
            self.views.setCurrentIndex(2)
        elif "transport" in output:
            self.views.setCurrentIndex(1)
        self.draw_diagnostics()

    def _update_status(self):
        base = getattr(self, "input_summary", "Ar CCP")
        if self.field is not None:
            status = "周期収束" if self.field.converged else "周期未収束・探索結果"
            self.status_label.setText(f"{base}\n表示結果：{status} / 物理精度は未検証")
        else:
            self.status_label.setText(base)

    def draw_diagnostics(self):
        lines = []
        if self.field is not None:
            d = self.field.diagnostics
            lines.extend(
                [
                    "RF場の数値診断",
                    "",
                    f"周期収束：{'許容値を満たす' if d['periodic_converged'] else '未収束'}",
                    f"停止理由：{d['stop_reason']}",
                    f"完了周期：{d['completed_cycles']} / 最終周期誤差：{d['periodic_error']:.4g}",
                    f"イオン数収支の相対誤差：{d['ion_balance_relative_error']:.3g}",
                    f"電子数収支の相対誤差：{d['electron_balance_relative_error']:.3g}",
                    "電子エネルギーの数値収支誤差："
                    f"{d.get('electron_energy_balance_relative_error', float('nan')):.3g}",
                    "格子間隔 r/z："
                    f"{d['radial_spacing_m'] * 1000:.3g} / {d['axial_spacing_m'] * 1000:.3g} mm",
                    f"最小Debye長：{(d.get('minimum_debye_m') or float('nan')) * 1000:.3g} mm",
                    f"物性範囲外の最大電子数割合：{d['maximum_table_outside_mass_fraction']:.2%}",
                    "電子仕事の線形化差："
                    f"{d.get('work_linearization_relative_difference', float('nan')):.3g}",
                    "",
                    "適用範囲・近似",
                    *[f"・{message}" for message in d["warnings"]],
                    "",
                    "再現性",
                    f"Boltzpmp：{d['boltzpmp_version']} / DC",
                    f"電子断面積 SHA-256：{d['electron_xsec_sha256']}",
                    f"イオン断面積 SHA-256：{d['ion_xsec_sha256']}",
                    f"波形：{d['source_waveform']}",
                ]
            )
        if self.particles is not None:
            d = self.particles.diagnostics
            lines.extend(
                [
                    "",
                    "テスト粒子",
                    f"注入：{d['injected']:,} / Wafer到達：{d['arrived']:,}",
                    f"接地面損失：{d['lost_ground']:,} / 打切り：{d['timed_out']:,}",
                    f"外部対除去による損失：{d.get('lost_external', 0):,}",
                    f"平均エネルギー：{d['mean_energy_ev']:.4g} eV",
                    f"平均角度：{d['mean_angle_deg']:.4g}°",
                    f"平均衝突数：{d['mean_collision_count']:.4g} / seed：{d['seed']}",
                    f"供給源：{d['source']}",
                    *[f"・{message}" for message in d["warnings"]],
                ]
            )
        if self.table is not None:
            lines.extend(
                [
                    "",
                    "電子物性",
                    f"DC計算点：{len(self.table.en_td)}",
                    f"平均電子エネルギー範囲：{self.table.energy_range}",
                    *self.table.metadata.get("warnings", []),
                ]
            )
        self.diagnostic_text.setPlainText("\n".join(lines))

    def save_figure(self):
        index = self.views.currentIndex()
        if index >= 4:
            return
        key = ("inputs", "transport", "field", "particles")[index]
        path, _ = QFileDialog.getSaveFileName(
            self, "図を保存", str(self.workspace / f"{key}.png"), "PNG (*.png)"
        )
        if path:
            self.figures[key][0].savefig(path, dpi=180)

    def export_csv(self):
        if self.particles is None:
            self.error("粒子計算結果が必要です。")
            return
        path = QFileDialog.getExistingDirectory(
            self, "分布CSVの保存先", str(self.workspace / "results")
        )
        if path:
            export_distributions(Path(path), self.particles)
            self.log.appendPlainText(f"分布CSVを保存：{path}")

    def start_run(self, action, resume=None):
        if self.process is not None:
            return
        try:
            project = self.collect_project()
            make_waveform(project)
            if action == "particles" and self.field_path is None:
                raise ValueError("先に場を計算するか、保存済みの場を開いてください。")
            stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
            directory = self.workspace / "results" / f"{stamp}_{uuid.uuid4().hex[:6]}"
            manifest = {
                "project": project.to_dict(),
                "directory": str(directory),
                "action": action,
                "allow_unconverged": self.allow_unconverged.isChecked(),
            }
            if action == "particles":
                manifest["field"] = str(self.field_path)
            if resume:
                manifest["resume"] = str(resume)
            write_json(directory / "manifest.json", manifest)
            self.current_directory = directory
            self.project = project
            self.stdout_buffer = b""
            self.process = QProcess(self)
            self.process.setWorkingDirectory(str(self.workspace))
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONUNBUFFERED", "1")
            environment.insert("PYTHONIOENCODING", "utf-8")
            self.process.setProcessEnvironment(environment)
            self.process.readyReadStandardOutput.connect(self.read_stdout)
            self.process.readyReadStandardError.connect(self.read_stderr)
            self.process.finished.connect(self.process_finished)
            self.process.errorOccurred.connect(self.process_error)
            self.process.start(
                sys.executable, ["-m", "sheath_pic.worker", str(directory / "manifest.json")]
            )
            self.settings_tabs.setEnabled(False)
            for button in self.action_buttons:
                button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.progress.setValue(0)
            self.progress_text.setText("計算プロセスを開始しています…")
            self.log.appendPlainText(f"実行：{action} / {directory.name}")
        except Exception as exc:
            self.error(exc)

    def read_stdout(self):
        if self.process is None:
            return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                self.log.appendPlainText(line)
                continue
            kind = event.get("event")
            if kind == "progress":
                self.progress.setValue(round(event["value"] * 1000))
                self.progress_text.setText(event["message"])
            elif kind == "stage":
                self.log.appendPlainText(f"開始：{event['name']}")
            elif kind in {"result", "complete"}:
                try:
                    self.load_outputs(event)
                    if kind == "complete":
                        self.progress.setValue(1000)
                        self.progress_text.setText("計算完了。結果・診断・入力条件を保存しました。")
                        self.log.appendPlainText("完了")
                except Exception as exc:
                    self.error(exc)
            elif kind == "error":
                self.progress_text.setText(event["message"])
                self.log.appendPlainText(event.get("details", event["message"]))
                self.error(event["message"])
            elif kind == "cancelled":
                self.progress_text.setText(event["message"])
                self.log.appendPlainText(event["message"])

    def read_stderr(self):
        if self.process is not None:
            message = (
                bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
            )
            if message:
                self.log.appendPlainText(message)

    def process_finished(self, code, *_):
        self.read_stdout()
        self.read_stderr()
        process, self.process = self.process, None
        if process is not None:
            process.deleteLater()
        self.settings_tabs.setEnabled(True)
        for button in self.action_buttons:
            button.setEnabled(True)
        self.stop_button.setEnabled(False)
        checkpoint = self.current_directory / "checkpoint.npz" if self.current_directory else None
        self.resume_button.setEnabled(bool(checkpoint and checkpoint.exists()))
        if code not in {0, 2}:
            self.log.appendPlainText(f"プロセス終了コード：{code}")

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.progress_text.setText(
                "計算プロセスを開始できませんでした。Python環境を確認してください。"
            )
            self.process_finished(1)

    def cancel_run(self):
        if self.process is not None and self.current_directory is not None:
            (self.current_directory / "cancel.request").touch()
            self.progress_text.setText(
                "中断を要求しました。現在の計算処理が完了するまで待っています。"
            )
            self.stop_button.setEnabled(False)

    def resume_run(self):
        directory = self.current_directory or self.result_directory
        if directory is None or not (directory / "checkpoint.npz").exists():
            self.error("再開可能なチェックポイントがありません。")
            return
        self.start_run("field", directory / "checkpoint.npz")

    def closeEvent(self, event):
        if self.process is not None:
            self.cancel_run()
            self.progress_text.setText("中断後にウィンドウを閉じてください。")
            event.ignore()
        else:
            event.accept()

    def draw_field(self, *_):
        figure, canvas = self.figures["field"]
        figure.clear()
        if self.field is None:
            ax = figure.add_subplot()
            ax.text(
                0.5,
                0.5,
                "RF potential / density / sheath",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            field = self.field
            project = Project.from_dict(field.project)
            mesh = AxisymmetricMesh(project.geometry)
            index = min(self.phase_slider.value(), len(field.phase_s) - 1)
            quantity = self.quantity_combo.currentData()
            energy_mask = field.electron_density_m3[index] < max(
                float(field.electron_density_m3[index].max()) * 1e-3,
                1,
            )
            visible_energy = np.ma.masked_where(energy_mask, field.mean_energy_ev[index])
            if quantity == "electric_magnitude":
                values = np.hypot(field.electric_r_vm[index], field.electric_z_vm[index]) / 1000
            else:
                values = getattr(field, quantity)[index]
                if quantity.endswith("density_m3"):
                    values = np.log10(np.maximum(values, 1))
                elif quantity == "mean_energy_ev":
                    values = visible_energy
                elif quantity != "mean_energy_ev":
                    values = values / 1000
            a, b, c, d = figure.subplots(2, 2).ravel()
            image = a.pcolormesh(
                mesh.r_edges * 1000, mesh.z_edges * 1000, values.T, cmap="viridis", shading="flat"
            )
            figure.colorbar(image, ax=a)
            a.set(xlabel="r [mm]", ylabel="z [mm]", title=self.quantity_combo.currentText())
            obs = plasma_observables(field, mesh)
            a.plot(
                mesh.r * 1000,
                obs["sheath_edge_m"][index] * 1000,
                color=SERIES[self.mode][3],
                linewidth=1.4,
            )
            image_b = b.pcolormesh(
                mesh.r_edges * 1000,
                mesh.z_edges * 1000,
                visible_energy.T,
                cmap="cividis",
                shading="flat",
            )
            figure.colorbar(image_b, ax=b, label="eV")
            b.set(xlabel="r [mm]", ylabel="z [mm]", title="Mean energy (ne ≥ 0.1% peak)")
            wave = Waveform(
                field.waveform_time_s,
                field.waveform_voltage_v,
                1 / project.numerics.frequency_hz,
                "saved",
            )
            c.plot(
                field.phase_s * 1e6,
                wave.at(field.phase_s) / 1000,
                color=SERIES[self.mode][0],
                label="Wafer [kV]",
            )
            c.set(xlabel="Phase time [µs]", ylabel="Wafer [kV]", title="Wafer / plasma potential")
            plasma_axis = c.twinx()
            plasma_axis.plot(
                field.phase_s * 1e6,
                obs["plasma_potential_v"],
                color=SERIES[self.mode][1],
                label="Plasma [V]",
            )
            plasma_axis.set_ylabel("Plasma [V]")
            c.axvline(field.phase_s[index] * 1e6, color=SERIES[self.mode][3], linewidth=1)
            rows = mesh.cathode_rows
            vs = (
                float(wave.at(field.phase_s[index])) - obs["radial_plasma_potential_v"][index, rows]
            )
            d.plot(mesh.r[rows] * 1000, vs / 1000, color=SERIES[self.mode][0])
            d.set(
                xlabel="Wafer radius [mm]",
                ylabel="Vmetal − Vplasma [kV]",
                title="Signed sheath voltage",
            )
            d.ticklabel_format(axis="y", style="plain", useOffset=False)
            self.phase_label.setText(
                f"{field.phase_s[index] * 1e6:.3f} µs / "
                + ("バルクあり" if obs["bulk_present"][index] else "準中性バルクなし")
            )
        style_figure(figure, self.mode)
        canvas.draw_idle()

    def draw_particles(self, *_):
        figure, canvas = self.figures["particles"]
        figure.clear()
        if self.particles is None:
            ax = figure.add_subplot()
            ax.text(
                0.5,
                0.5,
                "Ion energy / angle distributions",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            result = self.particles
            selected = self.radial_combo.currentIndex() - 1
            if selected >= 0:
                joint = result.radial_iaedf[selected]
                counts = result.counts[selected]
            else:
                joint = result.iaedf
                counts = result.counts.sum(axis=0)
            de, da = np.diff(result.energy_edges_ev), np.diff(result.angle_edges_deg)
            energy = 0.5 * (result.energy_edges_ev[1:] + result.energy_edges_ev[:-1])
            angle = 0.5 * (result.angle_edges_deg[1:] + result.angle_edges_deg[:-1])
            iedf, iadf = np.sum(joint * da[None, :], axis=1), np.sum(joint * de[:, None], axis=0)
            a, b, c, d = figure.subplots(2, 2).ravel()
            a.plot(energy, iedf, color=SERIES[self.mode][0])
            total = counts.sum()
            if total:
                probability = counts.sum(axis=1) / total
                error = np.sqrt(probability * (1 - probability) / total) / de
                a.fill_between(
                    energy,
                    np.maximum(iedf - error, 0),
                    iedf + error,
                    color=SERIES[self.mode][0],
                    alpha=0.2,
                )
            a.set(xlabel="Energy [eV]", ylabel="PDF [eV⁻¹]", title="IEDF")
            b.plot(angle, iadf, color=SERIES[self.mode][1])
            b.set(xlabel="Angle from wafer normal [deg]", ylabel="PDF [deg⁻¹]", title="IADF")
            image = c.pcolormesh(
                result.energy_edges_ev,
                result.angle_edges_deg,
                joint.T,
                cmap="viridis",
                shading="flat",
            )
            figure.colorbar(image, ax=c, label="PDF [eV⁻¹ deg⁻¹]")
            c.set(xlabel="Energy [eV]", ylabel="Angle [deg]", title="IAEDF")
            radius = 0.5 * (result.radius_edges_m[1:] + result.radius_edges_m[:-1])
            if np.isfinite(result.radial_flux_m2_s).any():
                d.plot(
                    radius * 1000, result.radial_flux_m2_s, color=SERIES[self.mode][2], marker="o"
                )
                d.set_ylabel("Ion flux [m⁻² s⁻¹]")
            else:
                d.bar(
                    radius * 1000,
                    result.counts.sum(axis=(1, 2)),
                    width=np.diff(result.radius_edges_m) * 1000,
                    color=SERIES[self.mode][2],
                )
                d.set_ylabel("Arrived particles")
            d.set(xlabel="Wafer radius [mm]", title="Radial arrival flux")
            stats = result.diagnostics
            self.particle_summary.setText(
                ("探索結果 · " if not stats["field_periodic_converged"] else "")
                + f"到達 {stats['arrived']:,} / {stats['injected']:,} · "
                f"平均 {stats['mean_energy_ev']:.1f} eV · "
                f"{stats['mean_angle_deg']:.2f}°"
            )
        style_figure(figure, self.mode)
        canvas.draw_idle()


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Sheath-PIC")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
