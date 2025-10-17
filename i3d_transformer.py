import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.i3d import InceptionI3d
from models.transformer import Transformer, TransformerEncoder
from models.mlp import MLP


__all__ = ["I3dTransformer"]


class I3dTransformer(nn.Module):
    def __init__(self, num_classes=67, d_model=64, transformer_config={}, mlp_config={}):
        super(I3dTransformer, self).__init__()
        self.d_model = d_model
        self.i3d = InceptionI3d(num_classes=num_classes, spatiotemporal_squeeze=True, final_endpoint="Logits", name="inception_i3d", in_channels=3, dropout_keep_prob=0.5, num_in_frames=64, include_embds=True)
        self.transformer = Transformer(d_model=d_model, **transformer_config)
        self.mlp = MLP(input_dim=1024, **mlp_config)
        self.class_query = nn.Parameter(torch.rand(16, d_model))
        # self.class_query = nn.Parameter(torch.rand(1, d_model))

    def forward(self, x):
        outputs = self.i3d(x)
        logits = outputs["logits"]  # [batch_size, 2000]
        x = outputs["embds"].squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)  # [batch_size, 1024, 1, 1, 1] --> [8, 1024]  
        # print("LOGITS SHAPE:", logits.shape)
        # print("EMBDS OUTPUT SHAPE:", x.shape)
    
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.d_model)  # [batch_size, 16, 64]
        class_query = self.class_query.unsqueeze(0).expand(batch_size, 16, self.d_model)
        # print("EMBDS RESHAPED:", x.shape)
        # print("CLASS QUERY SHAPE:", class_query.shape)
    
        x = self.transformer(x, class_query)  # [batch_size, 16, 64]
        # print("TRANSFORMER OUTPUT SHAPE:", x.shape)
    
        x = x.view(batch_size, -1)  # [batch_size, 1024]
        # print("TRANSFORMER FLATTENED SHAPE:", x.shape)
    
        embds = self.mlp(x)["embds"] # [batch_size, 2000]
        # print("MLP OUTPUT SHAPE:", embds.shape)
    
        return {"logits": logits, "embds": embds}




if __name__ == "__main__":
    inputs = torch.rand(1,3, 60, 224, 224)
    net = I3dTransformer(num_classes=4, d_model=64, transformer_config={'d_ff':32, 'num_heads':8, 'dropout':0,'num_layers':2})
    c = net(inputs)
    print(net)

'''
Peng Y, Lee J, Watanabe S. I3D: Transformer architectures with input-dependent dynamic depth for speech recognition[C]//ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023: 1-5.
'''









