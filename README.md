# MambaCpG
# MambaCpG

## Performance comparison between MambaCpG and other models on different scale datasets
| Dataset       | Ncell | (AUC) DeepCpG | (AUC) CpG Transformer | (AUC) GraphCpG | (AUC) MambaCpG |
|---------------|-------|---------------|-----------------------|----------------|----------------|
| 2i            |  12   |     84.64     |       **85.81**       |      75.81     |      85.08     |
| Ser           |  20   |     89.97     |         90.91         |      79.28     |    **91.42**   |
| HCC           |  25   |     96.73     |       **97.85**       |      97.14     |      97.81     |
| MBL           |  30   |     87.96     |       **92.43**       |      89.55     |      92.30     |
| Hemato        | 122   |     88.73     |         90.43         |      89.68     |    **91.01**   |
| Neuron-Mouse  | 690   |     89.06     |         91.49         |      91.63     |    **92.48**   |
| Neuron-Homo   | 780   |     90.34     |         92.95         |      93.25     |    **94.20**   |


## Construction process of methylation matrix and structure of MambaCpG.
![Image text]()


## Installation
We recommend adhering to the following instructions for local deployment and leveraging GPU resources for model training:

```bash
git clone https://github.com/
cd MambaCpG
conda create --name MambaCpG python=3.10
source activate MambaCpG
# Manually download the following two libraries, upload them to the project, and install them using `pip install` with the file paths.
[mamba-ssm](https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
pip install -r requirements.txt
```

## Usage
```bash
# train a model
python trainmodel.py X.npz y.npz pos.npz
# impute your dataset
python impute.py
```

## Citation

## License
