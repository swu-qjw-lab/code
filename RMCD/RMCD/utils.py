import argparse
from build_graph import build_hetero_graph_pyg
from dataset_config import get_dataset_config

DATA_CFG = get_dataset_config()

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=DATA_CFG["exer_n"],
                        help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=DATA_CFG["knowledge_n"],
                        help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=DATA_CFG["student_n"],
                        help='The number for student.')
        self.add_argument('--gpu', type=int, default=1,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=30,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')
        self.add_argument('--emb_dim', type=int, default=256,
                  help='Embedding dimension.')

def construct_local_map(args):
    data_file = DATA_CFG["graph_dir"]
    node_counts = {
        'user': args.student_n,     
        'exercise': args.exer_n,
        'knowledge': args.knowledge_n
    }
    local_map = build_hetero_graph_pyg(node_counts, data_file)

    return local_map


