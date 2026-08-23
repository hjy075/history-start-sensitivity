# Data sources and provenance

This project uses **third-party public research datasets**. Raw dataset files are not committed to this repository.

## Visuelle 2.0

- Official project page: https://humaticslab.github.io/forecasting/visuelle
- Official code repository: https://github.com/HumaticsLAB/visuelle2.0-code
- Paper: G. Skenderi et al., “The Multi-Modal Universe of Fast-Fashion: The Visuelle 2.0 Benchmark,” CVPR Workshops, 2022. DOI: `10.1109/CVPRW56347.2022.00245`.
- Access: the official page describes Visuelle 2.0 as publicly available and provides the download link after completion of a short form.
- Fields used here: **only the released `sales.csv` table** (weekly product-store sales). Images, customer-level purchase data, weather, Google Trends, prices, discounts, and other modalities are not used.
- Redistribution: **not included here**. The official project page requests citation of the benchmark paper; an explicit dataset license is not displayed on the project page as of 2026-08-23, so users should obtain the data through the original access route.

## FreshRetailNet-50K

- Official dataset: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- Paper: Y. Wang et al., “FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail,” arXiv:2505.16319.
- Version used: official public dataset release, **version 1.0**.
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)** according to the official dataset card.
- Split used here: **train** (`4,500,000` rows; 50,000 store-product series × 90 days).
- Columns used: `store_id`, `product_id`, `dt`, `sale_amount`, `stock_hour6_22_cnt`.
- Redistribution: not included here; the notebook loads the official Hugging Face repository directly.

### Metadata note

The current FreshRetailNet-50K arXiv paper (v5, June 2026) states **863 perishable SKUs**, while the current Hugging Face dataset card states **865**. This total SKU count is not used by our estimands or sample construction, so the manuscript intentionally omits the overall SKU total and reports only quantities directly relevant to our analysis.

## Citation responsibility

If you reproduce or extend this project, cite both the research paper/preprint and the original dataset papers/pages above. Do not treat this repository as the source of either raw dataset.
