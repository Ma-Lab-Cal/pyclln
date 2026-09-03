#!/usr/bin/env python3
"""Build Supplemental Figure 1: XOR update time and memory vs edge count."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REPO_ROOT = Path(__file__).resolve().parents[3]
def _rel(p):
    try: return str(Path(p).resolve().relative_to(REPO_ROOT.resolve()))
    except Exception: return Path(p).name
IMPORTED_TIMING_CSV = REPO_ROOT / "xor" / "results" / "xor_time_benchmark_multiside_import" / "summary" / "multiside_timing_summary.csv"
TIMING_CSV = DATA_DIR / "suppl1_xor_timing_summary.csv"
MEMORY_CSV = DATA_DIR / "suppl1_xor_memory_summary.csv"
METHODOLOGY_JSON = DATA_DIR / "suppl1_methodology.json"
XOR_NXN_SRC = REPO_ROOT / "xor" / "timing_benchmark_import" / "source_snapshot" / "xor_nxn"
SHARED_LVL1_TRAINER = XOR_NXN_SRC / "xor_nxn_alter_trainer_lvl1.py"
SUBPROCESS_LVL1_TRAINER = XOR_NXN_SRC / "xor_nxn_alter_trainer_subproc_lvl1.py"

BACKEND_LABEL = {
    "shared": "PyCLLN",
    "subprocess": "Naive Spice",
}
SOLVER_LABEL = {
    "klu": "KLU",
    "sparse": "Sparse",
}
SERIES_ORDER = [
    ("shared", "klu"),
    ("shared", "sparse"),
    ("subprocess", "klu"),
    ("subprocess", "sparse"),
]
TIMING_SERIES_ORDER = [
    ("shared", "klu"),
    ("subprocess", "klu"),
]
SELECTED_SIDES = [3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
STYLE = {
    ("shared", "klu"): {"color": "#1F77B4", "marker": "o", "linestyle": "-"},
    ("shared", "sparse"): {"color": "#1F77B4", "marker": "s", "linestyle": "--"},
    ("subprocess", "klu"): {"color": "#C0392B", "marker": "o", "linestyle": "-"},
    ("subprocess", "sparse"): {"color": "#C0392B", "marker": "s", "linestyle": "--"},
}

# Figure sizing follows the existing supplement panel convention.
PANEL_WIDTH_IN = 3.2
PANEL_HEIGHT_IN = 2.52
COMBINED_WIDTH_IN = 6.7
COMBINED_HEIGHT_IN = 2.52
PNG_DPI = 600
FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
PRIMARY_FONT = "Open Sans"
FONT_SIZE = 6.0
LEGEND_FONT_SIZE = 5.0
AXIS_COLOR = "#34383D"
LINEWIDTH = 1.1
MARKER_SIZE = 3.0
CURVE_ALPHA = 1.0
SPINE_LINEWIDTH = 0.55
TICK_WIDTH = 0.45
TICK_LENGTH = 2.0


@dataclass
class TimingRow:
    side: int
    edge_count: int
    backend: str
    solver: str
    per_sample_wall_mean_s: float


@dataclass
class MemoryRow:
    side: int
    edge_count: int
    backend: str
    solver: str
    status: str
    baseline_rss_mb: float
    peak_rss_mb: float
    peak_extra_rss_mb: float
    elapsed_s: float
    note: str = ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Supplemental Figure 1 XOR scaling panels")
    p.add_argument("--timing-csv", type=Path, default=TIMING_CSV)
    p.add_argument("--imported-timing-csv", type=Path, default=IMPORTED_TIMING_CSV)
    p.add_argument("--memory-csv", type=Path, default=MEMORY_CSV)
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--panel-prefix", type=str, default="suppl1")
    p.add_argument("--measure-timing", action="store_true", help="Measure timing rows and merge them into --timing-csv")
    p.add_argument(
        "--timing-sides",
        type=int,
        nargs="+",
        default=[3, 5, 7, 15],
        help="Grid sides to time when --measure-timing is set",
    )
    p.add_argument("--timing-epochs", type=int, default=50)
    p.add_argument("--timing-timeout-s", type=float, default=120.0)
    p.add_argument("--measure-memory", action="store_true", help="Regenerate memory CSV before plotting")
    p.add_argument(
        "--memory-sides",
        type=int,
        nargs="+",
        default=[3, 5, 7, 15],
        help="Grid sides to measure when --measure-memory is set",
    )
    p.add_argument("--memory-timeout-s", type=float, default=240.0)
    p.add_argument("--sample-interval-s", type=float, default=0.02)
    p.add_argument("--ngspice-bin", type=str, default="ngspice")
    return p.parse_args()


def _configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "font.weight": "normal",
            "axes.labelsize": FONT_SIZE,
            "axes.labelweight": "normal",
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        pad=1.5,
        colors=AXIS_COLOR,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        width=0.35,
        length=1.2,
        pad=1.5,
        colors=AXIS_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
        spine.set_edgecolor(AXIS_COLOR)
    ax.xaxis.label.set_fontfamily(PRIMARY_FONT)
    ax.xaxis.label.set_fontsize(FONT_SIZE)
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontfamily(PRIMARY_FONT)
    ax.yaxis.label.set_fontsize(FONT_SIZE)
    ax.yaxis.label.set_fontweight("normal")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(PRIMARY_FONT)
        label.set_fontsize(FONT_SIZE)
        label.set_fontweight("normal")


def _series_label(backend: str, solver: str) -> str:
    return f"{BACKEND_LABEL[backend]} {SOLVER_LABEL[solver]}"


def _backend_label(backend: str, solver: str) -> str:
    return BACKEND_LABEL[backend]


def _edge_count(side: int) -> int:
    return 4 if side == 2 else 2 * side * side


def load_timing_rows(path: Path) -> list[TimingRow]:
    rows: list[TimingRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        imported_format = "available" in (reader.fieldnames or []) or "device_level" in (reader.fieldnames or [])
        for row in reader:
            if imported_format:
                if row.get("available") != "True":
                    continue
                if row.get("device_level") != "lvl1":
                    continue
            backend = str(row["backend"]).strip().lower()
            solver = str(row["solver"]).strip().lower()
            if (backend, solver) not in SERIES_ORDER:
                continue
            value = row.get("per_sample_wall_mean_s", "")
            if value == "":
                continue
            rows.append(
                TimingRow(
                    side=int(row["side"]),
                    edge_count=int(row["edge_count"]),
                    backend=backend,
                    solver=solver,
                    per_sample_wall_mean_s=float(value),
                )
            )
    rows.sort(key=lambda r: (r.edge_count, r.backend, r.solver))
    if not rows:
        raise FileNotFoundError(f"No timing rows found in {path}")
    return rows


def merge_timing_rows(*row_sets: list[TimingRow]) -> list[TimingRow]:
    merged: dict[tuple[int, str, str], TimingRow] = {}
    for rows in row_sets:
        for row in rows:
            merged[(row.side, row.backend, row.solver)] = row
    return sorted(merged.values(), key=lambda r: (r.edge_count, r.backend, r.solver))


def filter_timing_rows_for_plot(rows: list[TimingRow]) -> list[TimingRow]:
    keep = set(SELECTED_SIDES)
    return [row for row in rows if row.side in keep]


def load_memory_rows(path: Path) -> list[MemoryRow]:
    rows: list[MemoryRow] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                MemoryRow(
                    side=int(row["side"]),
                    edge_count=int(row["edge_count"]),
                    backend=str(row["backend"]).strip().lower(),
                    solver=str(row["solver"]).strip().lower(),
                    status=str(row["status"]).strip(),
                    baseline_rss_mb=float(row["baseline_rss_mb"]),
                    peak_rss_mb=float(row["peak_rss_mb"]),
                    peak_extra_rss_mb=float(row["peak_extra_rss_mb"]),
                    elapsed_s=float(row["elapsed_s"]),
                    note=str(row.get("note", "")),
                )
            )
    rows.sort(key=lambda r: (r.edge_count, r.backend, r.solver))
    return rows


def write_timing_source(rows: list[TimingRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        fieldnames = list(TimingRow.__dataclass_fields__)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _invoke_timing_run(
    side: int,
    backend: str,
    solver: str,
    epochs: int,
    timeout_s: float,
    ngspice_bin: str,
) -> TimingRow:
    if backend == "shared":
        script = SHARED_LVL1_TRAINER
    elif backend == "subprocess":
        script = SUBPROCESS_LVL1_TRAINER
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if backend == "subprocess" and shutil.which(ngspice_bin) is None:
        raise FileNotFoundError(f"ngspice executable not found: {ngspice_bin}")

    with tempfile.TemporaryDirectory(prefix=f"suppl1_xor_time_s{side}_{backend}_{solver}_") as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            "--side",
            str(side),
            "--epochs",
            str(epochs),
            "--seed",
            "0",
            "--solver",
            solver,
        ]
        if backend == "subprocess":
            cmd.extend(["--ngspice-bin", ngspice_bin])

        env = os.environ.copy()
        env["RUN_DIR"] = str(run_dir)

        try:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout_tail = "\n".join((exc.stdout or "").splitlines()[-8:])
            stderr_tail = "\n".join((exc.stderr or "").splitlines()[-8:])
            raise RuntimeError(
                f"timing run failed for side={side} {backend}/{solver}\n"
                f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
            ) from exc

        epoch_times = np.load(run_dir / "0_epoch_times.npy").astype(float)
        if epoch_times.size == 0:
            raise RuntimeError(f"No epoch timing data for side={side} {backend}/{solver}")
        return TimingRow(
            side=side,
            edge_count=_edge_count(side),
            backend=backend,
            solver=solver,
            per_sample_wall_mean_s=float(np.mean(epoch_times / 4.0)),
        )


def measure_timing_rows(
    sides: Iterable[int],
    epochs: int,
    timeout_s: float,
    ngspice_bin: str,
) -> list[TimingRow]:
    rows: list[TimingRow] = []
    for side in sides:
        for backend, solver in SERIES_ORDER:
            row = _invoke_timing_run(
                side=side,
                backend=backend,
                solver=solver,
                epochs=epochs,
                timeout_s=timeout_s,
                ngspice_bin=ngspice_bin,
            )
            rows.append(row)
            print(
                f"[timing] side={side} edges={row.edge_count} {backend}/{solver} "
                f"per_sample={row.per_sample_wall_mean_s:.6g}s",
                flush=True,
            )
    return rows


def write_memory_rows(rows: list[MemoryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        fieldnames = list(MemoryRow.__dataclass_fields__)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def merge_memory_rows(*row_sets: list[MemoryRow]) -> list[MemoryRow]:
    merged: dict[tuple[int, str, str], MemoryRow] = {}
    for rows in row_sets:
        for row in rows:
            merged[(row.side, row.backend, row.solver)] = row
    return sorted(merged.values(), key=lambda r: (r.edge_count, r.backend, r.solver))


def filter_memory_rows_for_plot(rows: list[MemoryRow]) -> list[MemoryRow]:
    keep = set(SELECTED_SIDES)
    return [row for row in rows if row.side in keep]


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1])
    except OSError:
        return 0
    return 0


def _process_ppids() -> dict[int, int]:
    out: dict[int, int] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            text = Path(entry.path, "stat").read_text()
            close = text.rfind(")")
            if close == -1:
                continue
            parts = text[close + 2 :].split()
            if len(parts) >= 2:
                out[pid] = int(parts[1])
        except OSError:
            continue
    return out


def _tree_rss_kb(root_pid: int) -> int:
    ppids = _process_ppids()
    children: dict[int, list[int]] = {}
    for pid, ppid in ppids.items():
        children.setdefault(ppid, []).append(pid)

    total = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _rss_kb(pid)
        stack.extend(children.get(pid, []))
    return total


class RssMonitor:
    def __init__(self, root_pid: int, interval_s: float) -> None:
        self.root_pid = int(root_pid)
        self.interval_s = float(interval_s)
        self.baseline_kb = 0
        self.peak_kb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "RssMonitor":
        self.baseline_kb = _tree_rss_kb(self.root_pid)
        self.peak_kb = self.baseline_kb
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.peak_kb = max(self.peak_kb, _tree_rss_kb(self.root_pid))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_kb = max(self.peak_kb, _tree_rss_kb(self.root_pid))
            time.sleep(self.interval_s)

    @property
    def baseline_mb(self) -> float:
        return self.baseline_kb / 1024.0

    @property
    def peak_mb(self) -> float:
        return self.peak_kb / 1024.0

    @property
    def peak_extra_mb(self) -> float:
        return max(0.0, (self.peak_kb - self.baseline_kb) / 1024.0)


def _import_xor_modules() -> tuple[object, object]:
    if str(XOR_NXN_SRC) not in sys.path:
        sys.path.insert(0, str(XOR_NXN_SRC))
    import xor_nxn_alter_trainer as base  # type: ignore
    import xor_nxn_alter_trainer_lvl1 as lvl1  # type: ignore

    return base, lvl1


def _build_lvl1_problem(side: int, solver: str):
    base, lvl1 = _import_xor_modules()
    x_data, y_data = base.xor_dataset()
    graph, input_nodes, output_nodes, negref_idx, posref_idx = base._build_training_graph(side)
    edge_list = list(graph.edges())
    rng = np.random.default_rng(0)
    gate_voltages = rng.uniform(0.5, 3.0, size=len(edge_list)).astype(float)
    max_node = max(graph.nodes())
    netlist = lvl1.mk_switch_netlist(
        edge_list=edge_list,
        weights=gate_voltages,
        max_node=max_node,
        I_pos=0.45,
        I_neg=0.0,
        i_idxs=input_nodes,
        o_idxs=output_nodes,
        negref_idx=negref_idx,
        posref_idx=posref_idx,
        solver=solver,
    )
    nodes = np.asarray(sorted(graph.nodes()), dtype=int)
    index_of = np.full(nodes.max() + 1, -1, dtype=int)
    index_of[nodes] = np.arange(nodes.size, dtype=int)
    e1 = np.asarray([a for (a, _) in edge_list], dtype=int)
    e2 = np.asarray([b for (_, b) in edge_list], dtype=int)
    return {
        "base": base,
        "x": x_data,
        "y": y_data,
        "graph": graph,
        "edge_list": edge_list,
        "gate_voltages": gate_voltages,
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "negref_idx": negref_idx,
        "posref_idx": posref_idx,
        "max_node": max_node,
        "netlist": netlist,
        "nodes": nodes,
        "index_of": index_of,
        "e1": e1,
        "e2": e2,
    }


def _run_shared_update(side: int, solver: str) -> None:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    p = _build_lvl1_problem(side, solver)
    base = p["base"]
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(p["netlist"])

    x_row = p["x"][0]
    y_target = float(p["y"][0, 0])
    eta = 0.5
    gamma = 0.3

    base.mk_free(ng, len(p["output_nodes"]))
    base.alter_inputs(ng, x_row, base_idx=7)
    ng.run()
    free_out = base.get_voltages(ng, p["output_nodes"])
    free_nodes = base.get_voltages(ng, p["nodes"])
    free_y = float(free_out[0]) if free_out.size else 0.0
    try:
        ng.exec_command("destroy all")
    except Exception:
        pass

    v_target = eta * y_target + (1.0 - eta) * free_y
    base.alter_outputs(ng, np.array([v_target], dtype=float), vout_base_idx=7 + len(p["input_nodes"]))
    ng.run()
    clamped_nodes = base.get_voltages(ng, p["nodes"])
    try:
        ng.exec_command("destroy all")
    except Exception:
        pass

    free_e1 = free_nodes[p["index_of"][p["e1"]]]
    free_e2 = free_nodes[p["index_of"][p["e2"]]]
    clamped_e1 = clamped_nodes[p["index_of"][p["e1"]]]
    clamped_e2 = clamped_nodes[p["index_of"][p["e2"]]]
    update = -gamma * ((clamped_e1 - clamped_e2) ** 2 - (free_e1 - free_e2) ** 2)
    if np.any(update != 0.0):
        cmds = []
        gate_voltages = p["gate_voltages"]
        for k, du in enumerate(update):
            nv = float(np.clip(gate_voltages[k] + float(du), 0.4, 10.0))
            gate_voltages[k] = nv
            cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}")
        base._exec_chunked(ng, cmds)


def _inject_control(netlist: str, control_lines: Iterable[str]) -> str:
    lines = netlist.strip().splitlines()
    if not lines or lines[-1].strip().lower() != ".end":
        raise RuntimeError("Expected netlist to end with .end")
    return "\n".join(lines[:-1] + [".control", *control_lines, "quit", ".endc", lines[-1]]) + "\n"


def _parse_allv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    try:
        text = path.read_text()
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("v("):
            continue
        try:
            key, value = line.split(" = ")
            out[int(key[2:-1])] = float(value)
        except Exception:
            continue
    return out


def _voltages_from_map(values: dict[int, float], nodes: np.ndarray) -> np.ndarray:
    return np.asarray([values.get(int(n), np.nan) for n in nodes], dtype=float)


def _run_ngspice(cwd: Path, cir_name: str, ngspice_bin: str) -> None:
    subprocess.run(
        [ngspice_bin, "-b", cir_name],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def _run_subprocess_update(side: int, solver: str, ngspice_bin: str) -> None:
    p = _build_lvl1_problem(side, solver)
    x_row = np.asarray(p["x"][0], dtype=float).reshape(-1)
    y_target = float(p["y"][0, 0])
    eta = 0.5
    gamma = 0.3
    out_idx = int(p["output_nodes"][0])

    with tempfile.TemporaryDirectory(prefix="suppl1_xor_mem_") as tmp:
        tmpdir = Path(tmp)
        free_name = "free_allv.txt"
        free_control = [f"alter rs{i} 1e12" for i in range(1, len(p["output_nodes"]) + 1)]
        free_control += [f"alter v{7 + i} dc = {float(v):.16f}" for i, v in enumerate(x_row)]
        free_control += ["run", f"print allv > {free_name}"]
        (tmpdir / "free.cir").write_text(_inject_control(p["netlist"], free_control))
        _run_ngspice(tmpdir, "free.cir", ngspice_bin)

        free_map = _parse_allv(tmpdir / free_name)
        free_nodes = _voltages_from_map(free_map, p["nodes"])
        free_y = float(free_map.get(out_idx, 0.0))
        v_target = eta * y_target + (1.0 - eta) * free_y

        clamped_name = "clamped_allv.txt"
        clamp_control = [f"alter rs{i} 1.0" for i in range(1, len(p["output_nodes"]) + 1)]
        clamp_control += [f"alter v{7 + i} dc = {float(v):.16f}" for i, v in enumerate(x_row)]
        clamp_control.append(f"alter v{7 + len(p['input_nodes'])} dc = {float(v_target):.16f}")
        clamp_control += ["run", f"print allv > {clamped_name}"]
        (tmpdir / "clamped.cir").write_text(_inject_control(p["netlist"], clamp_control))
        _run_ngspice(tmpdir, "clamped.cir", ngspice_bin)

        clamped_map = _parse_allv(tmpdir / clamped_name)
        clamped_nodes = _voltages_from_map(clamped_map, p["nodes"])

    free_e1 = free_nodes[p["index_of"][p["e1"]]]
    free_e2 = free_nodes[p["index_of"][p["e2"]]]
    clamped_e1 = clamped_nodes[p["index_of"][p["e1"]]]
    clamped_e2 = clamped_nodes[p["index_of"][p["e2"]]]
    update = -gamma * ((clamped_e1 - clamped_e2) ** 2 - (free_e1 - free_e2) ** 2)
    p["gate_voltages"][:] = np.clip(p["gate_voltages"] + np.nan_to_num(update), 0.4, 10.0)


def _memory_worker(side: int, backend: str, solver: str, interval_s: float, ngspice_bin: str) -> dict[str, object]:
    t0 = time.perf_counter()
    with RssMonitor(os.getpid(), interval_s) as monitor:
        if backend == "shared":
            _run_shared_update(side, solver)
        elif backend == "subprocess":
            _run_subprocess_update(side, solver, ngspice_bin)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return {
        "status": "ok",
        "baseline_rss_mb": monitor.baseline_mb,
        "peak_rss_mb": monitor.peak_mb,
        "peak_extra_rss_mb": monitor.peak_extra_mb,
        "elapsed_s": time.perf_counter() - t0,
        "note": "peak tree RSS minus baseline tree RSS during one free+clamped XOR update",
    }


def _invoke_memory_worker(
    side: int,
    backend: str,
    solver: str,
    interval_s: float,
    timeout_s: float,
    ngspice_bin: str,
) -> MemoryRow:
    cmd = [
        sys.executable,
        _rel(Path(__file__)),
        "--_memory-worker",
        "--side",
        str(side),
        "--backend",
        backend,
        "--solver",
        solver,
        "--sample-interval-s",
        str(interval_s),
        "--ngspice-bin",
        ngspice_bin,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=True)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return MemoryRow(
            side=side,
            edge_count=_edge_count(side),
            backend=backend,
            solver=solver,
            status=str(payload["status"]),
            baseline_rss_mb=float(payload["baseline_rss_mb"]),
            peak_rss_mb=float(payload["peak_rss_mb"]),
            peak_extra_rss_mb=float(payload["peak_extra_rss_mb"]),
            elapsed_s=float(payload["elapsed_s"]),
            note=str(payload.get("note", "")),
        )
    except subprocess.TimeoutExpired:
        return MemoryRow(
            side=side,
            edge_count=_edge_count(side),
            backend=backend,
            solver=solver,
            status="timeout",
            baseline_rss_mb=0.0,
            peak_rss_mb=0.0,
            peak_extra_rss_mb=0.0,
            elapsed_s=time.perf_counter() - started,
            note=f"timed out after {timeout_s:.1f}s",
        )
    except Exception as exc:
        return MemoryRow(
            side=side,
            edge_count=_edge_count(side),
            backend=backend,
            solver=solver,
            status="error",
            baseline_rss_mb=0.0,
            peak_rss_mb=0.0,
            peak_extra_rss_mb=0.0,
            elapsed_s=time.perf_counter() - started,
            note=f"{type(exc).__name__}: {exc}",
        )


def measure_memory_rows(
    sides: Iterable[int],
    interval_s: float,
    timeout_s: float,
    ngspice_bin: str,
) -> list[MemoryRow]:
    if shutil.which(ngspice_bin) is None:
        raise FileNotFoundError(f"ngspice executable not found: {ngspice_bin}")
    rows: list[MemoryRow] = []
    for side in sides:
        for backend, solver in SERIES_ORDER:
            row = _invoke_memory_worker(side, backend, solver, interval_s, timeout_s, ngspice_bin)
            rows.append(row)
            print(
                f"[memory] side={side} edges={row.edge_count} {backend}/{solver} "
                f"status={row.status} peak_extra={row.peak_extra_rss_mb:.2f} MB "
                f"elapsed={row.elapsed_s:.2f}s",
                flush=True,
            )
    return rows


def _group_rows(rows):
    grouped = {key: [] for key in SERIES_ORDER}
    for row in rows:
        key = (row.backend, row.solver)
        if key in grouped:
            grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: r.edge_count)
    return grouped


def draw_timing_panel(
    ax: plt.Axes,
    rows: list[TimingRow],
    show_legend: bool = True,
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    grouped = _group_rows(rows)
    for key in TIMING_SERIES_ORDER:
        pts = grouped[key]
        if not pts:
            continue
        style = STYLE[key]
        ax.plot(
            [p.edge_count for p in pts],
            [p.per_sample_wall_mean_s for p in pts],
            label=_backend_label(*key),
            linewidth=LINEWIDTH,
            markersize=MARKER_SIZE,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            alpha=CURVE_ALPHA,
        )
    ax.set_xlabel("Edge Count")
    ax.set_ylabel("Time Per Sample Update (s)")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xscale == "linear":
        ax.set_xlim(left=0)
    if yscale == "linear":
        ax.set_ylim(bottom=0)
    _style_axis(ax)
    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper left",
            ncol=1,
            handlelength=3.2,
            numpoints=2,
            prop={"family": PRIMARY_FONT, "size": LEGEND_FONT_SIZE},
        )


def draw_memory_panel(ax: plt.Axes, rows: list[MemoryRow], show_legend: bool = True) -> None:
    ok_rows = [row for row in rows if row.status == "ok"]
    grouped = _group_rows(ok_rows)
    for key in SERIES_ORDER:
        pts = grouped[key]
        if not pts:
            continue
        style = STYLE[key]
        ax.plot(
            [p.edge_count for p in pts],
            [p.peak_rss_mb for p in pts],
            label=_series_label(*key),
            linewidth=LINEWIDTH,
            markersize=MARKER_SIZE,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            alpha=CURVE_ALPHA,
        )
    ax.set_xlabel("Edge Count")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_xlim(left=0)
    ymax = max((row.peak_rss_mb for row in ok_rows), default=0.0)
    upper = max(50.0, float(np.ceil(ymax / 50.0) * 50.0))
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + 1e-9, 50.0))
    _style_axis(ax)
    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper left",
            ncol=1,
            handlelength=3.2,
            numpoints=2,
            prop={"family": PRIMARY_FONT, "size": LEGEND_FONT_SIZE},
        )


def _save(fig: plt.Figure, out_prefix: Path) -> list[Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    return [png]


def make_figures(timing_rows: list[TimingRow], memory_rows: list[MemoryRow], out_dir: Path, prefix: str) -> list[Path]:
    _configure_fonts()
    paths: list[Path] = []

    # Linear-axis timing variant (not used in the manuscript composite; kept for reference).
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.16, top=0.98)
    draw_timing_panel(ax, timing_rows, show_legend=True)
    paths.extend(_save(fig, out_dir / f"{prefix}_linear_timing"))

    # Log-log timing: this single panel IS the manuscript Supplementary Figure 1 (suppl1.png).
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.16, top=0.98)
    draw_timing_panel(ax, timing_rows, show_legend=True, xscale="log", yscale="log")
    paths.extend(_save(fig, out_dir / f"{prefix}"))

    # Memory panel (not used in the manuscript composite; kept for reference).
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.16, top=0.98)
    draw_memory_panel(ax, memory_rows, show_legend=False)
    paths.extend(_save(fig, out_dir / f"{prefix}_memory"))

    fig, axes = plt.subplots(1, 2, figsize=(COMBINED_WIDTH_IN, COMBINED_HEIGHT_IN))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.17, top=0.82, wspace=0.30)
    draw_timing_panel(axes[0], timing_rows, show_legend=False)
    draw_memory_panel(axes[1], memory_rows, show_legend=False)
    axes[0].text(0.01, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=7)
    axes[1].text(0.01, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.52, 0.995),
        handlelength=3.2,
        numpoints=2,
        columnspacing=1.0,
        prop={"family": PRIMARY_FONT, "size": LEGEND_FONT_SIZE},
    )
    # Legacy two-panel combined composite (a: linear timing, b: memory); not used in the manuscript.
    paths.extend(_save(fig, out_dir / f"{prefix}_memory_timing_combined"))

    return paths


def write_methodology(args: argparse.Namespace, timing_rows: list[TimingRow], memory_rows: list[MemoryRow]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "selected_sides": SELECTED_SIDES,
        "timing": {
            "source_csv": _rel(args.timing_csv),
            "imported_timing_csv": _rel(args.imported_timing_csv),
            "metric": "50-epoch mean wall-clock time per sample update",
            "definition": "mean(0_epoch_times.npy / 4 XOR truth-table samples), XOR alter benchmark",
            "edge_count": "2 * side^2 for side >= 3",
            "timing_epochs_for_measured_rows": int(args.timing_epochs),
            "rows": len(timing_rows),
        },
        "memory": {
            "source_csv": _rel(args.memory_csv),
            "metric": "peak resident set size",
            "definition": "peak process-tree RSS during one XOR free+clamped update",
            "sampling_interval_s": float(args.sample_interval_s),
            "rows": len(memory_rows),
            "ok_rows": sum(1 for row in memory_rows if row.status == "ok"),
        },
        "series": [
            {"backend": backend, "solver": solver, "label": _series_label(backend, solver)}
            for backend, solver in SERIES_ORDER
        ],
    }
    METHODOLOGY_JSON.write_text(json.dumps(payload, indent=2))


def main() -> None:
    if "--_memory-worker" in sys.argv:
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--_memory-worker", action="store_true")
        worker_parser.add_argument("--side", type=int, required=True)
        worker_parser.add_argument("--backend", choices=["shared", "subprocess"], required=True)
        worker_parser.add_argument("--solver", choices=["klu", "sparse"], required=True)
        worker_parser.add_argument("--sample-interval-s", type=float, default=0.02)
        worker_parser.add_argument("--ngspice-bin", type=str, default="ngspice")
        worker_args = worker_parser.parse_args()
        result = _memory_worker(
            side=worker_args.side,
            backend=worker_args.backend,
            solver=worker_args.solver,
            interval_s=worker_args.sample_interval_s,
            ngspice_bin=worker_args.ngspice_bin,
        )
        print(json.dumps(result))
        return

    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    timing_source = args.timing_csv if args.timing_csv.exists() else args.imported_timing_csv
    timing_rows = load_timing_rows(timing_source)
    if args.measure_timing:
        measured_timing_rows = measure_timing_rows(
            sides=args.timing_sides,
            epochs=args.timing_epochs,
            timeout_s=args.timing_timeout_s,
            ngspice_bin=args.ngspice_bin,
        )
        timing_rows = merge_timing_rows(timing_rows, measured_timing_rows)
    timing_rows = filter_timing_rows_for_plot(merge_timing_rows(timing_rows))
    write_timing_source(timing_rows, args.timing_csv)

    memory_rows = load_memory_rows(args.memory_csv) if args.memory_csv.exists() else []
    if args.measure_memory or not args.memory_csv.exists():
        measured_memory_rows = measure_memory_rows(
            sides=args.memory_sides,
            interval_s=args.sample_interval_s,
            timeout_s=args.memory_timeout_s,
            ngspice_bin=args.ngspice_bin,
        )
        memory_rows = merge_memory_rows(memory_rows, measured_memory_rows)
    memory_rows = filter_memory_rows_for_plot(merge_memory_rows(memory_rows))
    write_memory_rows(memory_rows, args.memory_csv)

    paths = make_figures(timing_rows, memory_rows, args.out_dir, args.panel_prefix)
    write_methodology(args, timing_rows, memory_rows)
    print("wrote:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
