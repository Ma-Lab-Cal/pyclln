#!/usr/bin/env python3
"""Assemble Supplementary Figure 4 from the Fig. 3 all-test sensitivity panel.

S4 uses the signed all-test sensitivity,

    E_test[d(O_c+ - O_c-) / dx_p],

not the evidence-weighted attribution E_test[x_p J].
"""

from __future__ import annotations

from pathlib import Path
_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())  # robust paper_release root (self-contained)

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIG3_DIR = _PR / "final_figures" / "main_figures" / "fig3"

DATA_DIR = HERE / "data"
SIGNED_MAP = DATA_DIR / "suppl4_signed_map.png"
SIGNED_COLORBAR = DATA_DIR / "suppl4_signed_colorbar.png"
SIGNED_DATA = FIG3_DIR / "data" / "fig3_scikit_all_test_sensitivity.npz"

OUT_MAP = HERE / "suppl4.png"
OUT_DATA = DATA_DIR / "suppl4.npz"

BACKGROUND = (255, 255, 255, 255)
GAP_PX = 48
PAD_PX = 24


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    for path in (SIGNED_MAP, SIGNED_COLORBAR, SIGNED_DATA):
        _require(path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    map_img = Image.open(SIGNED_MAP).convert("RGBA")
    cbar_img = Image.open(SIGNED_COLORBAR).convert("RGBA")

    height = max(map_img.height, cbar_img.height) + 2 * PAD_PX
    width = map_img.width + GAP_PX + cbar_img.width + 2 * PAD_PX
    canvas = Image.new("RGBA", (width, height), BACKGROUND)
    canvas.alpha_composite(map_img, (PAD_PX, PAD_PX + (height - 2 * PAD_PX - map_img.height) // 2))
    canvas.alpha_composite(
        cbar_img,
        (
            PAD_PX + map_img.width + GAP_PX,
            PAD_PX + (height - 2 * PAD_PX - cbar_img.height) // 2,
        ),
    )
    canvas.convert("RGB").save(OUT_MAP, dpi=(600, 600))

    source_data = np.load(SIGNED_DATA, allow_pickle=True)
    np.savez(
        OUT_DATA,
        signed_sensitivity=source_data["signed_sensitivity"],
        signed_vlim=source_data["signed_vlim"],
        test_labels=source_data["test_labels"],
        n_test=source_data["n_test"],
        class_counts=source_data["class_counts"],
        mean_output_voltage=source_data["mean_output_voltage"],
        gates=source_data["gates"],
        surface=source_data["surface"],
        source_npz=str(SIGNED_DATA),
        definition=(
            "unweighted all-test averages for each output: signed=E_test[J]; "
            "units are V differential output per V input"
        ),
    )

    print(OUT_MAP)
    print(OUT_DATA)


if __name__ == "__main__":
    main()
