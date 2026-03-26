![PCLAI Banner](./figures/pclai_banner.png)

*Quick links:*
**Learn about the PCLAI HPRC v2 BED files:** [PCLAI on HPRC Release 2 samples](https://github.com/AI-sandbox/hprc-pclai) | **Preprint**: [biorxiv](https://www.biorxiv.org/content/10.64898/2026.03.23.713813v1)

## Point Cloud Local Ancestry Inference (PCLAI)

Point cloud local ancestry inference (PCLAI) is a deep learning-based approach for inferring continuous population genetic structure along the genome. Instead of assigning each genomic window to a discrete ancestry label, PCLAI predicts a **continuous coordinate** (e.g., a point in PC1–PC2 space) for every window, together with a **per-window confidence score**. For more technical details, we recommend reading [our corresponding preprint](https://www.biorxiv.org/content/10.64898/2026.03.23.713813v1).

## Usage

### Training PCLAI on custom datasets
Soon!

### Inferring PCLAI coordinates
Soon!

### Demo
Soon!

### License
**NOTICE**: This software is available for use free of charge for academic research use only. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to that effect. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" should contact the authors. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice.


## Cite

When using the [PCLAI method](https://github.com/AI-sandbox/pclai) or [PCLAI outputs](https://github.com/AI-sandbox/hprc-pclai), please cite the following paper:

```{tex}
@article{geleta_pclai_2026,
    author = {Geleta, Margarita and Mas Montserrat, Daniel and Ioannidis, Nilah M. and Ioannidis, Alexander G.},
    title = {{Point cloud local ancestry inference (PCLAI): coordinate-based ancestry along the genome}},
    year = {2026},
    journal = {biorxiv},
    doi={10.64898/2026.03.23.713813}
}
```
