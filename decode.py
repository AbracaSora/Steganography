from PIL import Image

import torch
from omegaconf import OmegaConf
from torchvision.transforms import transforms
from torchvision.utils import save_image

from ldm.util import instantiate_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 加载模型
config = OmegaConf.load("./model/VQ4_mir.yaml")
second_stage_model = instantiate_from_config(config)

# model文件夹下有两个文件Encoder.pth和Decoder.pth
# second_stage_model对应Decoder.pth
second_stage_model.init_from_ckpt("model/Decoder.pth")

second_stage_model.to(device)

# 加载损失函数

# 加载隐写图像和封面图像
stego_image = Image.open("stego_image.png").convert("RGB")

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

stego_image = transformer(stego_image).unsqueeze(0).to(device)


with torch.no_grad():
    output_latent = second_stage_model.encode(stego_image)
    output = second_stage_model.decode(output_latent)

# 保存隐写图像和输出图像
save_image(output, f"output.png")
