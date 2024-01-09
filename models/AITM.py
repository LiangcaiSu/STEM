import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import numpy as np 

'''
Reference Code: https://github.com/easezyc/Multitask-Recommendation-Library/blob/main/models/aitm.py
'''

class AITM(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="AITM",
                 gpu=-1,
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 learning_rate=1e-3,
                 embedding_dim=10,
                 bottom_hidden_units=[64, 64, 64],
                 tower_hidden_units=[64, ],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AITM, self).__init__(feature_map,
                                           task=task,
                                           loss_weight=loss_weight,
                                           num_tasks=num_tasks,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.hidden_dim = bottom_hidden_units[-1]
        self.g = torch.nn.ModuleList([torch.nn.Linear(bottom_hidden_units[-1], bottom_hidden_units[-1]) for i in range(num_tasks - 1)])
        self.h1 = torch.nn.Linear(bottom_hidden_units[-1], bottom_hidden_units[-1])
        self.h2 = torch.nn.Linear(bottom_hidden_units[-1], bottom_hidden_units[-1])
        self.h3 = torch.nn.Linear(bottom_hidden_units[-1], bottom_hidden_units[-1])

        self.bottom = nn.ModuleList([MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                                hidden_units=bottom_hidden_units,
                                hidden_activations=hidden_activations,
                                output_activation=None,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm) for _ in range(num_tasks)])
        self.tower = nn.ModuleList([MLP_Block(input_dim=bottom_hidden_units[-1],
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=hidden_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout,
                                              batch_norm=batch_norm)
                                    for _ in range(num_tasks)])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        bottom_output = [self.bottom[i](feature_emb.flatten(start_dim=1)) for i in range(self.num_tasks)] # [(?, bottom_hidden_units[-1])]
        for i in range(1, self.num_tasks):
            p = self.g[i - 1](bottom_output[i - 1]).unsqueeze(1)
            q = bottom_output[i].unsqueeze(1)
            x = torch.cat([p, q], dim = 1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            bottom_output[i] = torch.sum(torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim), dim=1) * V, 1)
        tower_output = [self.tower[i](bottom_output[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
