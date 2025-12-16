"""Set model_in and model_out, then run this to add a zero layer to µ-Net to
create a µr-Net ready to train.
"""

import os

import torch


os.makedirs("checkpoints", exists_ok=True)
# The path for the model to bootstrap from. This should be a basic MicroNet.
model_in  = "trained_models/u-net_1600.pt"
model_out = "checkpoints/ur-net_bootstrap.pt"

# A new zero layer that outputs four features.
out4 = torch.zeros((4, 1, 3, 3, 3), dtype=torch.float32)

m = torch.load(model_in)
m["down1.1.weight"] = torch.cat((out4, m["down1.1.weight"]), 1)
m["up0.0.weight"] = torch.cat((out4, m["up0.0.weight"]), 1)
torch.save(m, model_out)


# Parameter labels by layer.
# L1:
#   'down0.0.weight', 'down0.0.bias'
#   'down0.1.weight', 'down0.1.bias'
# L2:
#   'down1.1.weight', 'down1.1.bias'
#   'down1.2.weight', 'down1.2.bias'
# L3:
#   'bottom.1.weight', 'bottom.1.bias'
#   'bottom.2.weight', 'bottom.2.bias'
# L4:
#   'bottom.4.weight', 'bottom.4.bias'
#   'bottom.5.weight', 'bottom.5.bias'
# L5:
#   'up1.0.weight', 'up1.0.bias'
#   'up1.1.weight', 'up1.1.bias'
# L6:
#   'up1.3.weight', 'up1.3.bias'
#   'up1.4.weight', 'up1.4.bias'
# L7:
#   'up0.0.weight', 'up0.0.bias'
#   'up0.1.weight', 'up0.1.bias'
# L8:
#   'up0.3.weight', 'up0.3.bias'
#   'up0.4.weight', 'up0.4.bias'
