# Multi-granularity Correlation Refinement for Semantic Correspondence
This repository is the implementation of the paper: 

Zhen Liang, Enyu Che, Guoqiang Xiao, and Jingwei Qu. [Multi-granularity Correlation Refinement for Semantic Correspondence](https://jingweiqu.github.io/project/MIAM/index.html). *ICME*, 2024.

It contains the training and evaluation procedures in the paper.

## Environment Settings
```
git clone https://github.com/2000LZZ/MIAM
cd MIAM
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U scikit-image
pip install git+https://github.com/albumentations-team/albumentations
pip install tensorboardX termcolor timm tqdm requests pandas
```

## Citation
```text
@inproceedings{liang2024multi,
 title={Multi-granularity Correlation Refinement for Semantic Correspondence,
 author={Liang, Zhen and Che, Enyu and Xiao, Guoqiang and Qu, Jingwei},
 booktitle={Proceedings of the IEEE International Conference on Multimedia & Expo},
 year={2024},
 publisher={IEEE},
 address={New York},
 doi={10.1109/ICME57554.2024.10687853}
}
```