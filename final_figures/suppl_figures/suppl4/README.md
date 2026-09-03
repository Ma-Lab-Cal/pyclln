# Supplemental Figure 4

Unweighted all-test local sensitivity maps for the final scikit-digits
network. This is the signed `all_sensitivity` quantity from Fig. 3:

`E_test[d(O_c+ - O_c-) / dx_p]`

It is not the input-voltage-weighted evidence attribution
`E_test[x_p d(O_c+ - O_c-) / dx_p]`.

Colorbar units are V/V: volts of differential output change per volt of input
pixel change. The displayed signed range is `-0.037` to `+0.037` V/V; green is
negative and purple is positive.

Outputs:

- `suppl4.png`: assembled S4 panel with the
  signed all-test sensitivity maps and colorbar.
- `data/suppl4.npz`: signed/unweighted sensitivity
  values and provenance.

The signed map + colorbar live in data/ (written here by the Fig. 3 sensitivity script from the shared fig3 `.npz`). Reassemble the panel with:

```bash
python final_figures/suppl_figures/suppl4/plot_suppl4_all_test_sensitivity.py
```
