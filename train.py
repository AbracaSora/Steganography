from pathlib import Path
from ldm.models.autoencoder import VQModelInterface
import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torchvision import transforms
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import datetime
import torchmetrics.image
from Logger import TrainingLogger
from loss import ImageReconstructionLoss as loss_fn

# 指定数据集路径
dataset_path = Path("../images")
secrets_path = Path("./images/images_100")

# 加载配置文件
# first_process:VQModelInterface = instantiate_from_config(config)
# first_process.init_from_ckpt("model/model.ckpt")
config = OmegaConf.load("./VQ4_mir.yaml")
second_process: VQModelInterface = instantiate_from_config(config)
second_process.init_from_ckpt("./models/first_stage_models/vq-f4/model.ckpt")
output_process: VQModelInterface = instantiate_from_config(config)
output_process.init_from_ckpt("./models/first_stage_models/vq-f4/model.ckpt")

# 数据预处理
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def list_files(path):
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            files.extend(list_files(full_path))
        else:
            files.append(full_path)
    return files


images = list_files(dataset_path)
secrets = list_files(secrets_path)


def generate_secret():
    return random.choice(secrets)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


timestr = format(datetime.datetime.now(), "%Y%m%d_%H_%M_%S")
logger = TrainingLogger(log_file=f"train_log_{timestr}.log")
# 清除缓存
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# loss = torch.nn.MSELoss().cuda()
l1_loss = torch.nn.L1Loss().cuda()
# perceptual_loss = LPIPS().eval().cuda()
ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

loss = loss_fn().cuda()
param = list(second_process.parameters()) + list(output_process.parameters())
optimizer = torch.optim.Adam(param, lr=1e-5)
# first_process.to(device)
# first_process.to('cpu')
second_process.to(device)
output_process.to(device)

epochs = 10
batch_size = 1

# 创建数据集和数据加载器
dataset = ImageDataset(images, transform=transformer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=torch.utils.data.RandomSampler(dataset, replacement=True,
                                                                                num_samples=1000))
loss_list = []

for epoch in range(epochs):
    second_process.train()
    output_process.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    for i, image in enumerate(pbar):
        image = image.to(device)
        secret_path = generate_secret()
        secret = Image.open(secret_path).convert("RGB")
        secret = transformer(secret).unsqueeze(0).to(device)
        secret = secret.repeat(batch_size, 1, 1, 1).to(device)

        # 将图像和秘密图像映射到潜在空间
        image_latent = second_process.encode(image)
        secret_latent = second_process.encode(secret)

        # 进行隐写处理
        latent_object = 0.5 * image_latent + 0.5 * secret_latent

        # 重建图像
        recon_image = second_process.decode(latent_object)

        # 解码图像中的秘密
        recon_image_latent = output_process.encode(recon_image)
        output = output_process.decode(recon_image_latent)

        # 计算损失
        # mseLoss = loss(recon_image, image)
        # perceptualLoss = perceptual_loss(recon_image, image).mean(dim=[1,2,3])
        # l1Loss = l1_loss(recon_image, image)
        # decoderLoss = 1 - ssim(output, secret) + l1_loss(output, secret)
        # ssim_loss = 1 - ssim(recon_image, image)
        # loss_value = 0.5 * mseLoss + alpha * decoderLoss + beta * perceptualLoss + gamma * l1Loss + delta * ssim_loss
        # total = epochs * len(dataloader)
        global_step = epoch * len(dataloader) + i
        # alpha = global_step /total * 0.6 + 0.2
        l1Loss = l1_loss(output, secret)
        ssimLoss = (1 - ssim(output, secret))
        loss_dict = loss(recon_image, image, global_step)
        loss_value = 0.1 * l1Loss + 0.1 * ssimLoss + 0.8 * loss_dict[0]

        # 记录损失
        logger.log_metrics(i, ssimLoss.item(), l1Loss.item(), loss_dict[1]['perceptual_loss'],
                           loss_dict[1]['recon_loss'], loss_dict[0].item())

        optimizer.zero_grad()
        total_loss += loss_value.item()
        loss_value.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss_value.item())
        if i % 10 == 0:
            save_image(recon_image[0], f"output_1/reconstructed_{epoch}_{i}.png")
            save_image(image[0], f"output_1/original_{epoch}_{i}.png")
            save_image(output[0], f"output_1/output_{epoch}_{i}.png")

    logger.log_epoch(epoch,len(dataloader))
    loss_list.append(total_loss / len(dataloader))

logger.save_loss_plot(filename=f"loss_plot_{timestr}.png")
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_plot.png')

torch.save(second_process.state_dict(), f"second_process_{timestr}.pth")
torch.save(output_process.state_dict(), f"output_process_{timestr}.pth")