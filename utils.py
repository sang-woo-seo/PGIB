import torch
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import matplotlib
from torch_geometric.utils.num_nodes import maybe_num_nodes
from textwrap import wrap
import random
import pdb

def graph_dataset_generate(args, save_path):

    class_list = ["house", "cycle", "grid", "diamond"]
    settings_dict = {"ba": {"width_basis": args.node_num ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    class_num = class_list.__len__()
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}

    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("create shape:{}".format(shape))
        for i in tqdm(range(args.data_num)):
            tr_g, label1 = creat_one_pyg_graph(context="tree", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            ba_g, label2 = creat_one_pyg_graph(context="ba", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += "/syn_dataset.pt"
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset

def dataset_bias_split(dataset, args, bias=None, split=None, total=20000):
    
    class_list = ["house", "cycle", "grid", "diamond"]
    bias_dict = {"house": bias, "cycle": 1 - bias, "grid": 1 - bias, "diamond": 1 - bias}
    
    ba_dataset = dataset['ba']
    tr_dataset = dataset['tree']
    
    train_split, val_split, test_split = float(split[0]) / 10, float(split[1]) / 10, float(split[2]) / 10
    assert train_split + val_split + test_split == 1
    train_num, val_num, test_num = total * train_split, total * val_split, total * test_split
    # blance class
    class_num = args.num_classes
    train_class_num, val_class_num, test_class_num = train_num / class_num, val_num / class_num, test_num / class_num
    train_list, val_list, test_list  = [], [], []
    edges_num = 0
    
    for shape in class_list:
        bias = bias_dict[shape]
        train_tr_num = int(train_class_num * bias)
        train_ba_num = int(train_class_num * (1 - bias))
        val_tr_num = int(val_class_num * bias)
        val_ba_num = int(val_class_num * (1 - bias))
        test_tr_num = int(test_class_num * 0.5)
        test_ba_num = int(test_class_num * 0.5)
        train_list += tr_dataset[shape][:train_tr_num] + ba_dataset[shape][:train_ba_num]
        val_list += tr_dataset[shape][train_tr_num:train_tr_num + val_tr_num] + ba_dataset[shape][train_ba_num:train_ba_num + val_ba_num]
        test_list += tr_dataset[shape][train_tr_num + val_tr_num:train_tr_num + val_tr_num + test_tr_num] + ba_dataset[shape][train_ba_num + val_ba_num:train_ba_num + val_ba_num + test_ba_num]
        _, e1 = print_graph_info(tr_dataset[shape][0], "Tree", shape)
        _, e2 = print_graph_info(ba_dataset[shape][0], "BA", shape)
        
        edges_num += e1 + e2
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    the = float(edges_num) / (class_num * 2)
    return train_list, val_list, test_list, the

def print_dataset_info(train_set, val_set, test_set, the):
    class_list = ["house", "cycle", "grid", "diamond"]
    dataset_group_dict = {}
    dataset_group_dict["Train"] = dataset_context_object_info(train_set, "Train", class_list, the)
    dataset_group_dict["Val"] = dataset_context_object_info(val_set, "Val   ", class_list, the)
    dataset_group_dict["Test"] = dataset_context_object_info(test_set, "Test  ", class_list, the)
    return dataset_group_dict

def print_graph_info(G, c, o):
    print('-' * 100)
    print("| graph: {}-{} | nodes num:{} | edges num:{} |".format(c, o, G.num_nodes, G.num_edges))
    print('-' * 100)
    return G.num_nodes, G.num_edges

def dataset_context_object_info(dataset, title, class_list, the):

    class_num = len(class_list)
    tr_list = [0] * class_num
    ba_list = [0] * class_num
    for g in dataset:
        if g.num_edges > the: # ba
            ba_list[g.y.item()] += 1
        else: # tree
            tr_list[g.y.item()] += 1
    total = sum(tr_list) + sum(ba_list)
    info = "{} Total:{}\n| Tree: House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BA  : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| All : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BIAS: House:{:.1f}%, Cycle:{:.1f}%, Grids:{:.1f}%, Diams:{:.1f}%"
    print("-" * 150)
    print(info.format(title, total, tr_list[0], tr_list[1], tr_list[2], tr_list[3],
                                    ba_list[0], ba_list[1], ba_list[2], ba_list[3],
                                    tr_list[0] +  ba_list[0],    
                                    tr_list[1] +  ba_list[1], 
                                    tr_list[2] +  ba_list[2], 
                                    tr_list[3] +  ba_list[3],
                                    100 *float(tr_list[0]) / (tr_list[0] +  ba_list[0]),
                                    100 *float(tr_list[1]) / (tr_list[1] +  ba_list[1]),
                                    100 *float(tr_list[2]) / (tr_list[2] +  ba_list[2]),
                                    100 *float(tr_list[3]) / (tr_list[3] +  ba_list[3]),
                     ))
    print("-" * 150)
    total_list = ba_list + tr_list
    group_counts = torch.tensor(total_list).float()
    return group_counts



def find_closest_node_result(results, max_nodes):
    """ return the highest reward graph node constraining to the subgraph size """
    results = sorted(results, key=lambda x: x.P, reverse=True)
    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node


class PlotUtils():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def plot(self, graph, nodelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() == 'BA_2motifs'.lower():
            self.plot_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['bbbp', 'mutag', 'clintox']:
            x = kwargs.get('x')
            # print('*************************************************************')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        elif self.dataset_name.lower() == 'ba_shapes':
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() in ['grt_sst2_BERT_Identity'.lower(), 'graph_sst2', 'graph_twitter', 'reddit-binary']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, figname=figname)
        else:
            raise NotImplementedError

    def plot_subgraph(self, graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                    edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        # pdb.set_trace()
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        # nx.draw_networkx_nodes(graph, pos_nodelist,
        #                         nodelist=nodelist,
        #                         node_color=colors,
        #                         node_size=600)                                            

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        
        plt.show() #####################
        
        plt.close('all')

    def plot_subgraph_with_nodes(self, graph, nodelist, node_idx, colors='#FFA500', labels=None, edge_color='gray',
                                edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_ba2motifs(self, graph, nodelist, edgelist=None, figname=None):
        return self.plot_subgraph(graph, nodelist, edgelist=edgelist, figname=figname)

    def plot_molecule(self, graph, nodelist, x, edgelist=None, figname=None):
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name == 'bbbp' or 'clintox':
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                           edgelist=edgelist, edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=None, figname=figname)

    def plot_sentence(self, graph, nodelist, words, edgelist=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='darkorange',
                                   node_shape='o',
                                   node_size=800)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
            nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='darkorange')

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_color='powderblue', node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey')
        nx.draw_networkx_labels(graph, pos, words_dict, font_size=16)

        plt.axis('off')
        matplotlib.rcParams['figure.figsize'] = 6, 6
        plt.title('\n'.join(wrap(' '.join(words), width=40)), fontsize=17)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close('all')

    def plot_bashapes(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                           subgraph_edge_color='black')

    def get_topk_edges_subgraph(self, edge_index, edge_mask, top_k, un_directed=False):
        if un_directed:
            top_k = 2 * top_k
        edge_mask = edge_mask.reshape(-1)
        thres_index = max(edge_mask.shape[0] - top_k, 0)
        threshold = float(edge_mask.reshape(-1).sort().values[thres_index])
        hard_edge_mask = (edge_mask >= threshold)
        selected_edge_idx = np.where(hard_edge_mask == 1)[0].tolist()
        nodelist = []
        edgelist = []
        for edge_idx in selected_edge_idx:
            edges = edge_index[:, edge_idx].tolist()
            nodelist += [int(edges[0]), int(edges[1])]
            edgelist.append((edges[0], edges[1]))
        nodelist = list(set(nodelist))
        return nodelist, edgelist

    def plot_soft_edge_mask(self, graph, edge_index, edge_mask, top_k, un_directed, figname, **kwargs):
        #edge_index = torch.tensor(list(graph.edges())).T
        edge_mask = torch.tensor(edge_mask)
        if self.dataset_name.lower() == 'BA_2motifs'.lower():
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_ba2motifs(graph, nodelist, edgelist, figname=figname)

        elif self.dataset_name.lower() in ['bbbp', 'mutag']:
            x = kwargs.get('x')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_molecule(graph, nodelist, x, edgelist, figname=figname)

        elif self.dataset_name.lower() == 'ba_shapes':
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_bashapes(graph, nodelist, y, node_idx, edgelist, figname=figname)

        elif self.dataset_name.lower() in ['Graph_SST2'.lower(), 'graph_twitter']:
            words = kwargs.get('words')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_sentence(graph, nodelist, words=words, edgelist=edgelist, figname=figname)
        else:
            raise NotImplementedError

