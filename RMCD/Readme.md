# Relation-Aware Graph Learning with Mixture-of-Experts Prediction for Cognitive Diagnosis

This repository is the implementation of the paper:

Jingwei Qu, Mingze Zhang, Pingshun Zhang, Li Tao, Ying Wang, Zhaofang Yang, and Haibin Ling.  
**Relation-Aware Graph Learning with Mixture-of-Experts Prediction for Cognitive Diagnosis**.

It contains the dataset configuration, graph construction, training, and evaluation procedures in the paper.

## Environment Settings

```bash
git clone https://github.com/swu-qjw-lab/code.git
cd code/RMCD

conda create -n RMCD python=3.8
conda activate RMCD

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Dataset Configuration

The dataset information is controlled by `dataset_config.py`.  
Before running the code, please modify this file to select the dataset and set the corresponding dataset statistics and file paths.

## Model Directory

Before training, please create the `model` folder to save the trained model files.

```bash
cd RMCD
mkdir model
```

## Graph Construction

Before training and evaluation, please build the graph files:

```bash
python build_graph_files.py
```

## Training and Evaluation

Run the following command to train and evaluate RMCD:

```bash
python main.py
```

## Citation

```text
@inproceedings{qu2026relation,
  title={Relation-Aware Graph Learning with Mixture-of-Experts Prediction for Cognitive Diagnosis},
  author={Qu, Jingwei and Zhang, Mingze and Zhang, Pingshun and Tao, Li and Wang, Ying and Yang, Zhaofang and Ling, Haibin},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence},
  year={2026},
  publisher={IJCAI}
}
```
## Pretrained Weights

Due to the large size of the trained weights, we do not include them in this repository.
You can download the pretrained weights of RMCD from the following link:
[RMCD Pretrained Weights](https://higa.teracloud.jp/share/11e23dc7de43f528)
