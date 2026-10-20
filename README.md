# The Reflexivity Threshold

Simulation code for **The Reflexivity Threshold: A Phase Transition for Multi-Agent Performative Prediction**, Ahmed Mahrous and Roberto Di Pietro, NeurIPS 2026.

[Paper](https://openreview.net/forum?id=JykL1GuF6G)

## Reproduce the figures

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Outputs are saved in `figures/`. The scripts reproduce the iteration-complexity and spectral-radius sweep panels; seeds are recorded in `SEEDS.json`. Small numerical differences may occur across library versions.

## Citation

```bibtex
@inproceedings{mahrous2026reflexivity,
  title = {The Reflexivity Threshold: A Phase Transition for Multi-Agent Performative Prediction},
  author = {Mahrous, Ahmed and Di Pietro, Roberto},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://openreview.net/forum?id=JykL1GuF6G}
}
```

Code is released under the [MIT License](LICENSE).
