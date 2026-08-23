# History-Start Sensitivity in Retail Forecast Evaluation

Reproducibility repository for **“When Does History Start Matter? Model-Specific Sensitivity and Global Robustness in Retail Forecast Evaluation.”**

Repository: https://github.com/hjy075/history-start-sensitivity


## What this repository contains

- `notebooks/01_visuelle_main.ipynb` — Visuelle 2.0 main experiment.
- `notebooks/02_freshretail_data_gate.ipynb` — FreshRetailNet-50K data gate.
- `notebooks/03_freshretail_main.ipynb` — FreshRetailNet-50K main experiment.
- `notebooks/04_final_analysis.ipynb` — corrected series-weighted product-cluster bootstrap and paper outputs.
- `results/` — compact derived summaries used in the paper.
- `paper/` — LaTeX source, bibliography, and paper figures.

## Data

Raw third-party datasets are **not included**. See [`DATA_SOURCES.md`](DATA_SOURCES.md).

### Visuelle 2.0
Obtain the dataset through the official project page and access form:
https://humaticslab.github.io/forecasting/visuelle

The public notebook intentionally does not embed or bypass the provider’s access route. Set `VISUELLE_SOURCE` to your downloaded archive, `sales.csv`, or a directory containing `sales.csv`.

### FreshRetailNet-50K
The notebooks load the official Dingdong-Inc dataset from Hugging Face:
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

The dataset card lists version 1.0 and CC BY 4.0.

## Recommended execution order

1. `01_visuelle_main.ipynb`
2. `02_freshretail_data_gate.ipynb`
3. `03_freshretail_main.ipynb`
4. `04_final_analysis.ipynb`

All notebooks are designed for Google Colab and store caches/output under `MyDrive/history_start_sensitivity/`.

## Environment

The forecasting experiments use `statsforecast==2.1.1`. See `requirements.txt` for the lightweight Python dependencies.

## Citation

If you use the code, please cite the accompanying preprint. Dataset-specific citations remain required; see `DATA_SOURCES.md`.

## License

Code and notebooks are released under the MIT License. Third-party datasets are excluded and remain governed by their original providers’ terms. The manuscript and figures are not third-party data.
