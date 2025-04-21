import torch
from ldm.util import instantiate_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

first_stage_model = instantiate_from_config('./VQ4_mir.yaml')
