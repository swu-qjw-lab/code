
#main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
import os
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net 
from utils import CommonArgParser, construct_local_map
import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
# import time
def train(args,local_map):
    data_loader = TrainDataLoader()
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = Net(args, local_map)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.00005)
    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('Total parameter number: {}'.format(total_params))
    print('training model...')
    loss_function = nn.NLLLoss()
    epoch_num = args.epoch_n
    start=0
    flag = False
    for epoch in range(args.epoch_n):
        s_weight = 1
        net.train()
        data_loader.reset()
        # time_start = time.time()
        running_loss = 0.0
        running_moe_loss=0.0
        batch_count = 0
        model_path = f'model/model_epoch{epoch+1}'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path,weights_only=True)
            net.load_state_dict(checkpoint)
            #torch.cuda.empty_cache()
            rmse, auc = predict(args, net, epoch)
            continue
        total_samples = len(data_loader)
        batch_size = data_loader.batch_size
        expected_batches = (total_samples + batch_size - 1) // batch_size  
        print(f"当前epoch数据总量: {total_samples} 条，理论最大batch数: {expected_batches}")
        progress_bar = tqdm.tqdm(total=expected_batches, desc=f"Epoch {epoch+1}")
        start += 1
        correct_count,exer_count=0,0


        while not data_loader.is_end():
            batch_count += 1
            progress_bar.update(1)
            current_progress = progress_bar.n / expected_batches * 100
            progress_bar.set_postfix_str(f"处理进度: {current_progress:.1f}%")

            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()

            output_1,moe_loss= net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,return_loss=True)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            main_loss = loss_function(torch.log(output+1e-10), labels)
            total_loss= main_loss + 0.3*moe_loss
            total_loss.backward()
            optimizer.step()
            net.apply_clipper()
            running_loss += main_loss.item()
            running_moe_loss+=moe_loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f moeloss: %.3f ' % (epoch + 1, batch_count + 1, running_loss / 200,moe_loss / 200))
                running_loss = 0.0
                running_moe_loss=0.0
        progress_bar.close()
        # time_end = time.time()
        # print('epoch time ', time_end - time_start, ' second')
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))
        torch.cuda.empty_cache()
        rmse, auc = predict(args, net, epoch)
        flag = True

def predict(args, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader('predict')  
    print('predicting model...')
    data_loader.reset()
    net.eval()
    pred_count=0
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    total_samples = len(data_loader)  
    batch_size = data_loader.batch_size
    print(f"总样本为{total_samples}，规模为{batch_size}")
    loss_function = nn.NLLLoss()
    expected_batches = (total_samples + batch_size - 1) // batch_size  
    progress_bar = tqdm.tqdm(total=expected_batches, desc=f"Predict Epoch {epoch+1}")
    val_loss =0.0
    tmp_loss =0.0
    with torch.no_grad():  
        while not data_loader.is_end():
            progress_bar.update(1) 
            current_progress = progress_bar.n / expected_batches * 100
            progress_bar.set_postfix_str(f"处理进度: {current_progress:.1f}%") 
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            pred_count += 1
            if input_stu_ids is None:
                break
            input_stu_ids = input_stu_ids.to(device)
            input_exer_ids = input_exer_ids.to(device)
            input_knowledge_embs = input_knowledge_embs.to(device)
            labels = labels.to(device)
            output= net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_2 = torch.ones(output.size()).to(device) - output
            output_3 = torch.cat((output_2, output), 1)
            loss = loss_function(torch.log(output_3+1e-10), labels)
            val_loss +=loss.item()
            output = output.view(-1)            
            pred_labels = (output > 0.5).long()
            correct_count += (pred_labels == labels).sum().item()
            exer_count += labels.size(0)
            if pred_count % 200 == 199:
                print('loss: %.3f' % (val_loss/200))
                tmp_loss = val_loss/200
                val_loss=0.0
            pred_all.extend(output.cpu().numpy())
            label_all.extend(labels.cpu().numpy())
    print(f"正确数量{correct_count},实际数量{exer_count}")
    progress_bar.close()
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    pred_labels=(pred_all>0.5).astype(int)
    precision=precision_score(label_all,pred_labels,zero_division=1)
    recall=recall_score(label_all,pred_labels,zero_division=1)
    f1=f1_score(label_all,pred_labels,zero_division=1)

    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f, precision= %f, recall= %f, f1 score= %f' % (epoch+1, accuracy, rmse, auc, precision, recall, f1))
    with open('result/rmcd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, precision= %f, recall= %f, f1 score= %f,test loss= %f\n' % (epoch+1, accuracy, rmse, auc, precision, recall, f1,tmp_loss))

    return rmse, auc



def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args(args=[])
    args.gpu = 0
    local_map = construct_local_map(args)
    train(args,local_map)
