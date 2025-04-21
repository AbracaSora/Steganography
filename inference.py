import numpy as np
import torchmetrics
from PIL import Image
from loss import ImageReconstructionLoss
import torch
from omegaconf import OmegaConf
from torchvision.transforms import transforms
from torchvision.utils import save_image

from ldm.util import instantiate_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 加载模型
config = OmegaConf.load("./model/VQ4_mir.yaml")
first_stage_model = instantiate_from_config(config)
second_stage_model = instantiate_from_config(config)

# model文件夹下有两个文件Encoder.pth和Decoder.pth
# first_stage_model对应Encoder.pth
# second_stage_model对应Decoder.pth
first_stage_model.init_from_ckpt("model/Encoder.pth")
second_stage_model.init_from_ckpt("model/Decoder.pth")

first_stage_model.to(device)
second_stage_model.to(device)

# 加载损失函数
l1_loss = torch.nn.L1Loss().to(device)
ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
loss_fn = ImageReconstructionLoss().to(device)
# 加载隐写图像和封面图像
secret = Image.open("secret.png").convert("RGB")
cover = Image.open("cover.png").convert("RGB")

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

secret = transformer(secret).unsqueeze(0).to(device)
cover = transformer(cover).unsqueeze(0).to(device)

with torch.no_grad():
    # 生成隐写图像
    secret_latent = first_stage_model.encode(secret)
    cover_latent = first_stage_model.encode(cover)

    stego_latent = cover_latent * 0.5 + secret_latent * 0.5
    stego_image = first_stage_model.decode(stego_latent)

    output_latent = second_stage_model.encode(stego_image)
    output = second_stage_model.decode(output_latent)

# 保存隐写图像和输出图像
loss_dict = loss_fn(output, secret, 0)[1]
print("SSIM:", 1 - ssim(output, secret).item())
print("L1 Loss:", l1_loss(output, secret).item())
print("Reconstruction Loss:", loss_dict["recon_loss"])
print("Perceptual Loss:", loss_dict["perceptual_loss"])
save_image(stego_image, "stego_image.png")
save_image(cover, "origin.png")
save_image(output, f"output.png")
