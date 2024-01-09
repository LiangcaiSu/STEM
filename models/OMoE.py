import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class OMoE_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(OMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = MLP_Block(input_dim=input_dim,
                            hidden_units=gate_hidden_units,
                            output_dim=num_experts,
                            hidden_activations=hidden_activations,
                            output_activation=None,
                            dropout_rates=net_dropout,
                            batch_norm=batch_norm)
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        omoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate(x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            omoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return omoe_output


class OMoE(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="OMoE",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_experts=4,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(OMoE, self).__init__(feature_map,
                                   task=task,
                                   num_tasks=num_tasks,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.omoe_layer = OMoE_Layer(num_experts=num_experts,
                                     num_tasks=self.num_tasks,
                                     input_dim=embedding_dim * feature_map.num_fields,
                                     expert_hidden_units=expert_hidden_units,
                                     gate_hidden_units=gate_hidden_units,
                                     hidden_activations=hidden_activations,
                                     net_dropout=net_dropout,
                                     batch_norm=batch_norm)
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
        feature_emb = self.embedding_layer(X)
        expert_output = self.omoe_layer(feature_emb.flatten(start_dim=1))
        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
