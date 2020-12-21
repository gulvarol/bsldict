import torch

from models.i3d import InceptionI3d
from models.mlp import Mlp


class i3d_mlp(torch.nn.Module):
    def __init__(self, i3d_pretrained=None, mlp_pretrained=None):
        super(i3d_mlp, self).__init__()
        if i3d_pretrained:
            self.i3d = i3d_pretrained
        else:
            self.i3d = InceptionI3d(
                num_classes=1064,
                spatiotemporal_squeeze=True,
                final_endpoint="Logits",
                name="inception_i3d",
                in_channels=3,
                dropout_keep_prob=0.5,
                num_in_frames=16,
                include_embds=True,
            )
        if mlp_pretrained:
            self.mlp = mlp_pretrained
        else:
            self.mlp = Mlp()

    def forward(self, x):
        i3d_outputs = self.i3d(x)
        # logits from i3d
        logits = i3d_outputs["logits"]
        # [B, 1024, 1, 1, 1] => [B, 1024]
        x = i3d_outputs["embds"].squeeze(2).squeeze(2).squeeze(2)
        # Get embds from mlp (unused logits from mlp)
        embds = self.mlp(x)["embds"]
        return {"logits": logits, "embds": embds}
