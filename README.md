# PGIB
Implementation of Interpretable Prototype-based Graph Information Bottleneck

![architecture2_page-0001](./image_architecture.jpg)

The success of Graph Neural Networks (GNNs) has led to a need for understanding their decision-making process and providing explanations for their predictions, which has given rise to explainable AI (XAI) that offers transparent explanations for black-box models. Recently, the use of prototypes has successfully improved the explainability of models by learning prototypes to imply training graphs that affect the prediction. However, these approaches tend to provide prototypes with excessive information from the entire graph, leading to the exclusion of key substructures or the inclusion of irrelevant substructures, which can limit both the interpretability and the performance of the model in downstream tasks. In this work, we propose a novel framework of explainable GNNs, called interpretable Prototype-based Graph Information Bottleneck (PGIB) that incorporates prototype learning within the information bottleneck framework to provide prototypes with the key subgraph from the input graph that is important for the model prediction. This is the first work that incorporates prototype learning into the process of identifying the key subgraphs that have a critical impact on the prediction performance. Extensive experiments, including qualitative analysis, demonstrate that PGIB outperforms state-of-the-art methods in terms of both prediction performance and explainability.


## Requirements
```
pytorch                   1.11.0             
torch-geometric           2.0.4
torch-scatter             2.0.9
torch-sparse              0.6.13
```
## Dataset
* Download the datasets for graph classification in this link https://chrsmrrs.github.io/datasets/
* Download the datasets for graph interpretation in this link https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main/graph-interpretation/input

## Run

```
python -m models.train_gnns
```
