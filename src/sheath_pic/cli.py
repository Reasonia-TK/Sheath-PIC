from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Project
from .worker import execute


def main():
    parser = argparse.ArgumentParser(
        description="Sheath-PIC GUIと同じ計算核をコマンドラインから実行"
    )
    parser.add_argument("action", choices=["transport", "field", "particles", "all"])
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--field", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--quick", action="store_true", help="粗い格子・2周期の数値動作確認")
    parser.add_argument("--allow-unconverged", action="store_true")
    args = parser.parse_args()
    project = Project.load(args.project) if args.project else Project(base_dir=str(Path.cwd()))
    if args.quick:
        project.geometry.radial_cells = 8
        project.geometry.axial_cells = 12
        project.numerics.maximum_cycles = 2
        project.numerics.minimum_cycles = 2
        project.numerics.steps_per_period = 64
        project.numerics.stored_phases = 16
        project.numerics.particles = 2000
        project.transport.en_points = 12
        project.transport.angle_cells = 16
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    directory = args.output or Path("results") / stamp
    manifest = {
        "project": project.to_dict(),
        "directory": str(directory.resolve()),
        "action": args.action,
        "allow_unconverged": args.allow_unconverged,
    }
    if args.field:
        manifest["field"] = str(args.field.resolve())
    if args.resume:
        manifest["resume"] = str(args.resume.resolve())
    result = execute(manifest)
    print(result)


if __name__ == "__main__":
    main()
