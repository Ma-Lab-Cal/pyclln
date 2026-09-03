# Figure Dimensions

Dimensions are measured from the current generated PNG files in this folder.
PNG files are 600 dpi exports. Panel `fig1_c.png` is maximally trimmed after
generation. Panels `fig1_a.png`, `fig1_b.png`, and `fig1_d.png` are trimmed first,
then centered on a shared canvas so they have identical dimensions. Panel
`fig1_e.png` is cropped to its content and placed back onto the prescribed
2.36 x 3.57 in canvas.

## Main Figure

| File | Size (in) | Pixels |
|---|---:|---:|
| `fig1.png` | 6.96 x 4.82 | 4176 x 2891 px |

## Individual Panels

| File | Size (in) | Pixels | Notes |
|---|---:|---:|---|
| `fig1_a.png` | 1.13 x 1.06 | 678 x 639 px | shared a/b/d canvas |
| `fig1_b.png` | 1.13 x 1.06 | 678 x 639 px | shared a/b/d canvas |
| `fig1_c.png` | 0.85 x 0.87 | 512 x 523 px | trimmed |
| `fig1_d.png` | 1.13 x 1.06 | 678 x 639 px | shared a/b/d canvas |
| `fig1_e.png` | 2.36 x 3.57 | 1416 x 2142 px | content-trimmed exact canvas |

## MOSFET Legend

The MOSFET-network legend is rendered at the prescribed 2.00 x 0.30 in canvas
with explicit 6 pt Open Sans text. Keep this PNG at 2.00 x 0.30 in when placing
it; cropping and then scaling it back to 2.00 x 0.30 in changes the apparent
font size.

| File | Size (in) | Pixels | Font |
|---|---:|---:|---|
| `fig1d_legend.png` | 2.00 x 0.30 | 1200 x 180 px | 6 pt Open Sans |

## IV Curve Panels

Additional IV-curve panels for comparing linear resistor and source/body-tied
NMOS edge behavior. PNG files are 600 dpi exports.

| File | Size (in) | Pixels |
|---|---:|---:|
| `fig1_c_resistor.png` | 1.16 x 1.23 | 696 x 738 px |
| `fig1_c_nmos.png` | 1.16 x 1.23 | 696 x 738 px |

## Source

Regenerate with:

```bash
python figures_nmi/fig1_framework/make_fig1.py
```

Regenerate the IV-curve panels with:

```bash
python paper_release/final_figures/main_figures/fig1/plot_fig1_iv_curves.py
```

The script defines the exact panel canvases near the bottom of
`make_fig1.py`: panels `a`-`d` use 2.00 x 2.00 in, panel `e` uses
2.36 x 3.57 in, and the MOSFET legend uses 2.00 x 0.30 in.
