#build_graph.py
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph_pyg(node_counts, data_folder='../data/junyi/graph'):
    num_user = node_counts['user']
    num_exer = node_counts['exercise']
    num_know = node_counts['knowledge']
    data = HeteroData()

    data['user'].num_nodes = num_user
    data['exercise'].num_nodes = num_exer
    data['knowledge'].num_nodes = num_know

    def _read_edge_file(path):
        edges = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split('\t')
                edges.append((int(a), int(b)))
        if len(edges) == 0:
            return torch.empty((2,0), dtype=torch.long)
        src, dst = tuple(zip(*edges))
        return torch.tensor([src,dst], dtype=torch.long)

    def remap_edge(edge_index, src_offset=0, dst_offset=0):
        edge_index[0] = edge_index[0] - src_offset
        edge_index[1] = edge_index[1] - dst_offset
        return edge_index

    u_e_edge = _read_edge_file(f"{data_folder}/u_e.txt")
    # if data_num == 1:
    #     u_e_edge = remap_edge(u_e_edge, src_offset=0, dst_offset=835)
    # elif data_num == 2:
    #     u_e_edge = remap_edge(u_e_edge, src_offset=0, dst_offset=3162)
    # elif data_num == 3:
    #     u_e_edge = remap_edge(u_e_edge, src_offset=0, dst_offset=17746)
    u_e_edge = remap_edge(u_e_edge, src_offset=0, dst_offset=num_exer)

    data['exercise', 'e_u', 'user'].edge_index = u_e_edge
    data['user', 'u_e', 'exercise'].edge_index = u_e_edge.flip(0) 

    u_k_edge = _read_edge_file(f"{data_folder}/u_k.txt")
    u_k_edge[0] = u_k_edge[0] - 1
    u_k_edge[1] = u_k_edge[1] - 1

    data['user', 'u_k', 'knowledge'].edge_index = u_k_edge
    data['knowledge', 'k_u', 'user'].edge_index = u_k_edge.flip(0)


    e_k_edge = _read_edge_file(f"{data_folder}/e_k.txt")
    e_k_edge = remap_edge(e_k_edge, src_offset=num_exer, dst_offset=0)
    # if data_num==1:
    #     e_k_edge = remap_edge(e_k_edge, src_offset=835, dst_offset=0) #junyi
    # elif data_num == 2:
    #     e_k_edge = remap_edge(e_k_edge, src_offset=3162, dst_offset=0) #17
    # elif data_num == 3:  
    #     e_k_edge = remap_edge(e_k_edge, src_offset=17746, dst_offset=0) #09

    data['knowledge', 'k_e', 'exercise'].edge_index = e_k_edge
    data['exercise', 'e_k', 'knowledge'].edge_index = e_k_edge.flip(0)

    data['user'].num_nodes = num_user
    data['exercise'].num_nodes = num_exer
    data['knowledge'].num_nodes = num_know



    return data

