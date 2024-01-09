import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class ESMM(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="ESMM",
                 gpu=-1,
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 learning_rate=1e-3,
                 embedding_dim=10,
                 tower_hidden_units=[64, ],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(ESMM, self).__init__(feature_map,
                                           task=task,
                                           loss_weight=loss_weight,
                                           num_tasks=num_tasks,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        if num_tasks != 2:
            raise ValueError("the number of tasks must be equal to 2!")
        self.tower = nn.ModuleList([MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
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
        feature_emb = self.embedding_layer(X).flatten(start_dim=1)
        tower_output = [self.tower[i](feature_emb) for i in range(self.num_tasks)]
        cvr_pred = self.output_activation[0](tower_output[0])
        ctr_pred = self.output_activation[1](tower_output[1])
        ctcvr_pred =  ctr_pred * cvr_pred
        y_pred = [ctr_pred, ctcvr_pred]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
