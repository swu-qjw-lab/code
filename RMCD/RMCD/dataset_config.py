# dataset_config.py

DATA_NUM = 2  # 1: Junyi, 2: ASSIST17, 3: ASSIST09

DATASET_CONFIGS = {
    1: {
        "name": "junyi",
        "train_file": "../data/junyi/train_set.json",
        "test_file": "../data/junyi/test_set.json",
        "graph_dir": "../data/junyi/graph",
        "config_file": "config.txt",
        "student_n": 10000,
        "exer_n": 835,
        "knowledge_n": 835,
    },
    2: {
        "name": "ASSIST17",
        "train_file": "../data/ASSIST17/train_set.json",
        "test_file": "../data/ASSIST17/test_set.json",
        "graph_dir": "../data/ASSIST17/graph",
        "config_file": "config2.txt",
        "student_n": 1709,
        "exer_n": 3162,
        "knowledge_n": 102,
    },
    3: {
        "name": "ASSIST09",
        "train_file": "../data/ASSIST09/train_set.json",
        "test_file": "../data/ASSIST09/test_set.json",
        "graph_dir": "../data/ASSIST09/graph",
        "config_file": "config3.txt",
        "student_n": 2493,
        "exer_n": 17746,
        "knowledge_n": 123,
    },
}


def get_dataset_config():
    if DATA_NUM not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported DATA_NUM: {DATA_NUM}")
    return DATASET_CONFIGS[DATA_NUM]