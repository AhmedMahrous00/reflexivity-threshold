# Figure Manifest

| Paper asset | Output path | Source script |
| --- | --- | --- |
| Iteration-complexity threshold panel | `figures/fig_iteration_complexity.pdf` | `plot_iteration_complexity.py` |
| Iteration-complexity threshold panel, PNG | `figures/fig_iteration_complexity.png` | `plot_iteration_complexity.py` |
| Iteration-complexity source data | `figures/fig_iteration_complexity.csv` | `plot_iteration_complexity.py` |
| Rho-sweep trajectory panel | `figures/fig_rho_sweep_five_lines.pdf` | `plot_rho_sweep_five_lines.py` |
| Rho-sweep trajectory panel, PNG | `figures/fig_rho_sweep_five_lines.png` | `plot_rho_sweep_five_lines.py` |
| Rho-sweep mean curves | `figures/fig_rho_sweep_five_lines.csv` | `plot_rho_sweep_five_lines.py` |
| Rho-sweep raw simulation arrays | `figures/fig_rho_sweep_five_lines.npz` | `plot_rho_sweep_five_lines.py` |

Use `python synthetic_simulation_artifact/reproduce.py --clean` from the
repository root to regenerate all listed outputs.
