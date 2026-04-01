![PCLAI Banner](./figures/pclai_banner.png)

*Quick links:*
**Learn about the PCLAI HPRC v2 BED files:** [PCLAI on HPRC Release 2 samples](https://github.com/AI-sandbox/hprc-pclai) | **Preprint**: [TBD](https://github.com/AI-sandbox/pclai)

## Point Cloud Local Ancestry Inference (PCLAI)

Point cloud local ancestry inference (PCLAI) is a deep learning-based approach for inferring continuous population genetic structure along the genome. Instead of assigning each genomic window to a discrete ancestry label, PCLAI predicts a **continuous coordinate** (e.g., a point in PC1–PC2 space) for every window, together with a **per-window confidence score**. For more technical details, we recommend reading [our corresponding preprint](https://github.com/AI-sandbox/pclai).

## Usage

### Using pre-trained PCLAI models

#### Pretrained bundles

Currently, we provide two pretrained bundles:

- `pclai_1kg`: trained on 1000 Genomes
- `pclai_1kg+hgdp`: trained on 1000 Genomes + Human Genome Diversity Project (HGDP)

These bundles have different numbers of input SNPs, so you may want to try both and compare which one has better SNP coverage for your input VCF.

Each bundle follows this structure:

```text
pclai_bundle/
  manifest.json
  snp_manifests/
    chr1_s01.snps.tsv
    chr1_s02.snps.tsv
    ...
    chr13.snps.tsv
    ...
  models/
    pclainet_chm1_s01.pt2
    pclainet_chm1_s02.pt2
    ...
```

The `snp_manifests/` directory contains the exact ordered SNP list expected by each model artifact. Each `.snps.tsv` file has this format:
```
chrom   pos     rsid    ref     alt
chr21   5034221 .       G       A
chr21   5034230 .       C       T
chr21   5034244 .       C       T
```
The `models/` directory contains exported PyTorch artifacts (`.pt2`).

##### Input requirements

Your input VCF must be harmonized to the model/reference space before running inference. In practice, that means:

- GRCh38 coordinates
- Phased haplotypes
- Imputed against the reference panel
- Harmonized to the SNP order expected by the bundle

##### Harmonized output format

After harmonization, you should have a folder like:

```
my_vcfs/
  chr1.vcf.gz
  chr2.vcf.gz
  ...
  chr22.vcf.gz
```
These per-chromosome VCFs are in the exact SNP order expected by the bundle and can be passed directly to the inference runner.

##### Harmonization workflow

If your input VCF is already:

- GRCh38
- Phased
- Imputed
- And already matches the SNP order expected by the bundle

Then you can skip harmonization and run inference directly.

Otherwise, run `pipeline.py` first.

```
python3 ./pipeline.py \
  --input-vcf "/path/to/input.vcf.gz" \
  --workdir "/path/to/workdir" \
  --bundle-dir "/path/to/pclai_bundle" \
  --reference-split-template "/path/to/reference.chr{chrom}.vcf.gz" \
  --impute-engine beagle \
  --beagle-jar "/path/to/beagle.jar" \
  --threads 16 \
  --reference-fasta "/path/to/GRCh38.fa" \
  --auto-normalize-on-qc-fail \
  --log-level DEBUG
```

This produces harmonized chromosome VCFs under:
```
/path/to/workdir/
  chr1/final.for_model.chr1.vcf.gz
  chr2/final.for_model.chr2.vcf.gz
  ...
```

We provide `pipeline.py` to do this harmonization.

We need a folder with harmonized input VCFs that looks like this:

```
my_vcfs/
  chr1.vcf.gz
  chr2.vcf.gz
  ...
  chr22.vcf.gz
```
##### TL;DR

If you have an arbitrary input VCF and want to run our pretrained models:

1. Choose a bundle (`pclai_1kg_cpu`, `pclai_1kg_cuda`, `pclai_1kg+hgdp_cpu` or `pclai_1kg+hgdp_cuda`)
2. Harmonize your input VCF with `pipeline.py`
3. Collect the harmonized per-chromosome VCFs into a folder
4. Run the inference script on that folder


```

```



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
    journal = {https://github.com/AI-sandbox/pclai},
}
```
