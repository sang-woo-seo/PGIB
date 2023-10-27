import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args, mcts_args
from my_mcts import mcts
from tqdm import tqdm
from proto_join import join_prototypes_by_activations
from utils import PlotUtils
from torch_geometric.utils import to_networkx
from itertools import accumulate
from torch_geometric.datasets import MoleculeNet
import pdb
import random


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info):
    f = open('./log/hyper_search.txt', 'a')
    f.write(info)
    f.write('\n')
    f.close()


# train for graph classification
def train_GC(model_type):
    
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    dataloader = get_dataloader(dataset, data_args.dataset_name, train_args.batch_size, data_split_ratio=data_args.data_split_ratio) # train, val, test dataloader 나눔

    print('start training model==================')

    gnnNets = GnnNets(input_dim, output_dim, model_args) 
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]

    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print("Dataset : ", data_args.dataset_name)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    data_size = len(dataset)

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    early_stop_count = 0
    data_indices = dataloader['train'].dataset.indices 

    best_acc = 0.0

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            gnnNets.eval()

            # prototype projection
            for i in range( gnnNets.model.prototype_vectors.shape[0] ): 
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                for j in range(i*10, len(data_indices)): 
                    data = dataset[data_indices[j]] 
                    if data.y == label: 
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i]) 
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print('Projection of prototype completed')
                        break


            # prototype merge
            share = True
            if train_args.share: 
                if gnnNets.model.prototype_vectors.shape[0] > round(output_dim * model_args.num_prototypes_per_class * (1-train_args.merge_p)) :  
                    join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile,  dataloader['train'], optimizer)

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for i, batch in enumerate(dataloader['train']):
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(batch) 

            loss = criterion(logits, batch.y)

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device) 
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            #diversity loss
            prototype_numbers = []
            for i in range(gnnNets.model.prototype_class_identity.shape[1]):
                prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[: ,i])))
            prototype_numbers = accumulate(prototype_numbers)
            n = 0
            ld = 0

            for k in prototype_numbers:    
                p = gnnNets.model.prototype_vectors[n : k]
                n = k
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3 
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device) 
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2)) 

            if model_args.cont:
                loss = loss + train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 
            else:
                loss = loss + train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())


        # report train msg
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
              f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()))


        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record("Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))

        test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")           

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)


    print(f"The best validation accuracy is {best_acc}.")
    
    # report test msg
    gnnNets = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_{model_type}_{model_args.readout}_best.pth')) # .to_device()
    gnnNets.to_device()
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test | Dataset: {data_args.dataset_name:s} | model: {model_args.model_name:s}_{model_type:s} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record("loss: {:.3f}, acc: {:.3f}".format(test_state['loss'], test_state['acc']))

    return test_state['acc']





def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _, _, _, _ = gnnNets(batch)
            if data_args.dataset_name == 'clintox':
                batch.y = torch.tensor([torch.argmax(i).item() for i in batch.y]).to(model_args.device)
            loss = criterion(logits, batch.y)


            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state




def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            logits, probs, active_node_index, _, _, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)
            
            # # test_subgraph extraction          
            # save_dir = os.path.join('./masking_interpretation_results',
            #                         f"{mcts_args.dataset_name}_"
            #                         f"{model_args.readout}_"
            #                         f"{model_args.model_name}_")
            # if not os.path.isdir(save_dir):
            #     os.mkdir(save_dir)
            # plotutils = PlotUtils(dataset_name=data_args.dataset_name)

            # for i, index in enumerate(test_dataloader.dataset.indices[batch_index * train_args.batch_size: (batch_index+1) * train_args.batch_size]):
            #     data = test_dataloader.dataset.dataset[index]
            #     graph = to_networkx(data, to_undirected=True)
            #     if type(active_node_index[i]) == int:
            #         active_node_index[i] = [active_node_index[i]]
            #     print(active_node_index[i])
            #     plotutils.plot(graph, active_node_index[i], x=data.x,
            #                 figname=os.path.join(save_dir, f"example_{i}.png"))
    

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean())}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    # print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }

    pth_name = f"{model_name}_{model_type}_{model_args.readout}_latest.pth"
    best_pth_name = f'{model_name}_{model_type}_{model_args.readout}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(gnnNets, os.path.join(ckpt_dir, best_pth_name) )
    gnnNets.to(model_args.device)



if __name__ == '__main__':
    if os.path.isfile("./log/hyper_search.txt"):
        os.remove("./log/hyper_search.txt")

    if model_args.cont:
        model_type = 'cont'
    else:
        model_type = 'var'

    accuracy = train_GC(model_type)