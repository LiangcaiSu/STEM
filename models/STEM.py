import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
import numpy as np 
import logging 
class STEM_Layer(nn.Module):
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(STEM_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts 
        self.num_specific_experts = num_specific_experts 
        self.num_tasks = num_tasks 
        self.shared_experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_specific_experts)]) for _ in range(num_tasks)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             output_dim=num_specific_experts*num_tasks+num_shared_experts,
                                             hidden_units=gate_hidden_units,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for i in range(self.num_tasks+1)])
        self.gate_activation = get_activation('softmax')
    def forward(self, x, return_gate=False):
        """
        x: list, len(x)==num_tasks+1
        """
        specific_expert_outputs = [] 
        shared_expert_outputs = []
        # specific experts
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
        # shared experts 
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        
        # gate 
        stem_outputs = [] 
        stem_gates = []
        for i in range(self.num_tasks+1):
            if i < self.num_tasks:
                # for specific experts
                gate_input = [] 
                for j in range(self.num_tasks):
                    if j == i:
                        gate_input.extend(specific_expert_outputs[j])
                    else: 
                        specific_expert_outputs_j = specific_expert_outputs[j]
                        specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                        gate_input.extend(specific_expert_outputs_j)
                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input, dim=1) # (?, num_specific_experts*num_tasks+num_shared_experts, dim)
                gate = self.gate_activation(self.gate[i](x[i]+x[-1])) # (?, num_specific_experts*num_tasks+num_shared_experts)
                if return_gate:
                    specific_gate = gate[:,:self.num_specific_experts*self.num_tasks].mean(0)
                    task_gate = torch.chunk(specific_gate, chunks=self.num_tasks)
                    specific_gate_list = [] 
                    for tg in task_gate:
                        specific_gate_list.append(torch.sum(tg))
                    shared_gate = gate[:,-self.num_shared_experts:].mean(0).sum()
                    target_task_gate = torch.stack(specific_gate_list+[shared_gate],dim=0).view(-1) # (num_task+1,1)
                    assert len(target_task_gate) == self.num_tasks+1 
                    stem_gates.append(target_task_gate)
                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1) # (?, dim)
                stem_outputs.append(stem_output)
            else: 
                # for shared experts 
                gate_input = [] 
                for j in range(self.num_tasks):
                    gate_input.extend(specific_expert_outputs[j])
                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input, dim=1) # (?, num_specific_experts*num_tasks+num_shared_experts, dim)
                gate = self.gate_activation(self.gate[i](x[-1])) # (?, num_specific_experts*num_tasks+num_shared_experts)
                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1) # (?, dim)
                stem_outputs.append(stem_output)

        if return_gate:
            return stem_outputs, stem_gates
        else:
            return stem_outputs


class STEM(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="STEM",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=1,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(STEM, self).__init__(feature_map,
                                   task=task,
                                   num_tasks=num_tasks,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim * (self.num_tasks+1))
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.stem_layers = nn.ModuleList([STEM_Layer(num_shared_experts,
                                                   num_specific_experts,
                                                   num_tasks,
                                                   input_dim= self.embedding_dim * feature_map.num_fields if i==0 else expert_hidden_units[-1],
                                                   expert_hidden_units= expert_hidden_units,
                                                   gate_hidden_units=gate_hidden_units,
                                                   hidden_activations=hidden_activations,
                                                   net_dropout=net_dropout,
                                                   batch_norm=batch_norm) for i in range(self.num_layers)])
        self.tower = nn.ModuleList([MLP_Block(input_dim=expert_hidden_units[-1],
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
        feature_emb = self.embedding_layer(X) # (?, num_field, D)
        feature_embs = feature_emb.split(self.embedding_dim, dim=2)
        stem_inputs = [feature_embs[i].flatten(start_dim=1) for i in range(self.num_tasks+1)]
        for i in range(self.num_layers):
            stem_outputs = self.stem_layers[i](stem_inputs)
            stem_inputs = stem_outputs
        tower_output = [self.tower[i](stem_outputs[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict