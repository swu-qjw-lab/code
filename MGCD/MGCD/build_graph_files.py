import json
import os


def build_graph_files():
    # dataset = "junyi"
    # train_file = "../data/junyi/train_set.json"
    # q_file = "../data/junyi/log_data.json"
    # out_dir = "../data/junyi/graph"
    # config_file = "config.txt"

    dataset = "ASSIST17"
    train_file = "../data/ASSIST17/train_set.json"
    q_file = "../data/ASSIST17/log_data.json"
    out_dir = "../data/ASSIST17/graph"
    config_file = "config2.txt"

    # dataset = "ASSIST09"
    # train_file = "../data/ASSIST09/train_set.json"
    # q_file = "../data/ASSIST09/log_data_all.json"
    # out_dir = "../data/ASSIST09/graph"
    # config_file = "config3.txt"

    os.makedirs(out_dir, exist_ok=True)

    with open(config_file) as f:
        f.readline()
        student_n, exer_n, knowledge_n = map(int, f.readline().split(","))

    with open(train_file, "r", encoding="utf-8") as f:
        train_logs = json.load(f)

    e_u_edges = set() 
    u_k_edges = set()  

    for record in train_logs:
        user_id = record["user_id"] - 1
        exer_id = record["exer_id"] - 1

        e_u_edges.add((exer_id, user_id + exer_n))

        for k in record["knowledge_code"]:
            k_id = k - 1
            u_k_edges.add((user_id + 1, k_id + 1))

    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    k_e_edges = set() 
    for user_data in q_data:
        for log in user_data["logs"]:
            exer_id = log["exer_id"] - 1
            for k in log["knowledge_code"]:
                k_id = k - 1

                k_e_edges.add((k_id + exer_n, exer_id))

    with open(os.path.join(out_dir, "u_e.txt"), "w") as f:
        for e, u in sorted(e_u_edges):
            f.write(f"{e}\t{u}\n")

    with open(os.path.join(out_dir, "u_k.txt"), "w") as f:
        for u, k in sorted(u_k_edges):
            f.write(f"{u}\t{k}\n")

    with open(os.path.join(out_dir, "e_k.txt"), "w") as f:
        for k, e in sorted(k_e_edges):
            f.write(f"{k}\t{e}\n")

    print("边文件已生成到:", out_dir)
    print("dataset:", dataset)
    print("exercise-user edges:", len(e_u_edges))
    print("user-knowledge edges:", len(u_k_edges))
    print("knowledge-exercise edges:", len(k_e_edges))


if __name__ == "__main__":
    build_graph_files()