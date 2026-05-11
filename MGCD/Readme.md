conda create -n MGCD python=3.8
conda activate MGCD
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt 


python build_graph_files.py
python main.py
