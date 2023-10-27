import os
import torch
from typing import List


class DataParser():
    def __init__(self):
        super().__init__()
        self.dataset_name ='MUTAG'
        self.dataset_dir = './datasets'
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1, 0.1]   


class ModelParser():
    def __init__(self):
        super().__init__()
        self.device: int = 0
        self.model_name: str = 'gin' 
        self.checkpoint: str = './checkpoint'
        self.concate: bool = False                     
        self.latent_dim: List[int] = [128,128,128]  
        self.readout: 'str' = 'max' 
        self.mlp_hidden: List[int] = []               
        self.gnn_dropout: float = 0.0                  
        self.dropout: float = 0.5                   
        self.adj_normlize: bool = True                 
        self.emb_normlize: bool = False                
        self.enable_prot = True                        

        self.num_prototypes_per_class = 7            
        self.gat_dropout = 0.6  
        self.gat_heads = 10  
        self.gat_hidden = 10 
        self.gat_concate = True  
        self.num_gat_layer = 3
        self.con_weight = 5
        self.cont = True 

    def process_args(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass



class MCTSParser(DataParser, ModelParser):
    rollout: int = 10                         
    high2low: bool = False                    
    c_puct: float = 5                         
    min_atoms: int = 5                       
    max_atoms: int = 10                        
    expand_atoms: int = 10                    

    def process_args(self) -> None:
        self.explain_model_path = os.path.join(self.checkpoint,
                                               self.dataset_name,
                                               f"{self.model_name}_best.pth")


class RewardParser():
    def __init__(self):
        super().__init__()
        self.reward_method: str = 'mc_l_shapley'                         
        self.local_raduis: int = 4                                     
        self.subgraph_building_method: str = 'zero_filling'
        self.sample_num: int = 100                                    


class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005 #0.005 
        self.batch_size = 24
        self.weight_decay = 0.0
        self.max_epochs = 300 
        self.save_epoch = 10
        self.early_stopping = 10000 
        self.last_layer_optimizer_lr = 1e-4           
        self.joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}      
        self.global_epochs = 500 
        self.warm_epochs = 10                      
        self.proj_epochs = 50 
        self.sampling_epochs = 100                     
        self.nearest_graphs = 10                       
        self.proto_percnetile = 0.1 
        self.merge_p = 0.3 
        self.count = 1
        self.share = True
        self.alpha1 = 0.0001
        self.alpha2 = 0.01


class SynParser():
    def __init__(self):
        super().__init__()
        self.bias = 0.5
        self.data_num = 200 #2000
        self.num_classes = 4
        self.feature_dim = -1
        self.max_degree = 10
        self.batch_size = 128
        self.learning_rate = 0.005 #0.002
        self.weight_decay = 0.0
        self.max_epochs = 800 





data_args = DataParser()
model_args = ModelParser()

mcts_args = MCTSParser()
reward_args = RewardParser()
train_args = TrainParser()
syn_args = SynParser()

