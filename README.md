# Evaluating calibration for WSI classification

## Download data
- The TCGA colorectal cancer slides from TCGA-COAD and TCGA-READ studies can be downloaded from <https://portal.gdc.cancer.gov/>
- Access to the MCO slides can be requested at <https://www.sredhconsortium.org/sredh-datasets/mco-study-whole-slide-image-dataset>


## Setup directories and generate patch locations

1. Set environment variables. This can be done by adding the following lines to your `.bashrc` file:
```bash
export TCGA_ROOT_DIR=path/to/tcga/slides
export MCO_ROOT_DIR=path/to/mco/slides
export SLIDE_PROCESS_DIR=path/to/local/storage
```

2. Create output directories for TCGA and MCO datasets

3. Patch locations are computed using our preprocessing repository at <https://github.com/DBO-DKFZ/wsi_preprocessing-crc>