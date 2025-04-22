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
first_stage_model: VQModelInterface = instantiate_from_config(config)
first_stage_model.init_from_ckpt("./models/first_stage_models/vq-f4/model.ckpt")
second_stage_model: VQModelInterface = instantiate_from_config(config)
second_stage_model.init_from_ckpt("./models/first_stage_models/vq-f4/model.ckpt")

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
train_logger = TrainingLogger(log_file=f"train_log_{timestr}.log")
test_logger = TrainingLogger(log_file=f"test_log_{timestr}.log")
# 清除缓存
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l1_loss = torch.nn.L1Loss().cuda()
ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

loss = loss_fn().cuda()
param = list(first_stage_model.parameters()) + list(second_stage_model.parameters())
optimizer = torch.optim.Adam(param, lr=1e-5)
first_stage_model.to(device)
second_stage_model.to(device)

epochs = 10
batch_size = 1

# 创建数据集和数据加载器
dataset = ImageDataset(images, transform=transformer)
train_size = len(dataset) * 0.8
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_size), int(test_size)])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

loss_list = []

for epoch in range(epochs):
    first_stage_model.train()
    second_stage_model.train()
    total_loss = 0
    # 训练
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    for i, image in enumerate(pbar):
        image = image.to(device)
        secret_path = generate_secret()
        secret = Image.open(secret_path).convert("RGB")
        secret = transformer(secret).unsqueeze(0).to(device)
        secret = secret.repeat(batch_size, 1, 1, 1).to(device)

        # 将图像和秘密图像映射到潜在空间
        image_latent = first_stage_model.encode(image)
        secret_latent = first_stage_model.encode(secret)

        # 进行隐写处理
        latent_object = 0.5 * image_latent + 0.5 * secret_latent

        # 重建图像
        recon_image = first_stage_model.decode(latent_object)

        # 解码图像中的秘密
        recon_image_latent = second_stage_model.encode(recon_image)
        output = second_stage_model.decode(recon_image_latent)

        # 计算损失
        # mseLoss = loss(recon_image, image)
        # perceptualLoss = perceptual_loss(recon_image, image).mean(dim=[1,2,3])
        # l1Loss = l1_loss(recon_image, image)
        # decoderLoss = 1 - ssim(output, secret) + l1_loss(output, secret)
        # ssim_loss = 1 - ssim(recon_image, image)
        # loss_value = 0.5 * mseLoss + alpha * decoderLoss + beta * perceptualLoss + gamma * l1Loss + delta * ssim_loss
        # total = epochs * len(dataloader)
        global_step = epoch * len(pbar) + i
        # alpha = global_step /total * 0.6 + 0.2
        l1Loss = l1_loss(output, secret)
        ssimLoss = (1 - ssim(output, secret))
        loss_dict = loss(recon_image, image, global_step)
        loss_value = 0.1 * l1Loss + 0.1 * ssimLoss + 0.8 * loss_dict[0]

        # 记录损失
        train_logger.log_metrics(i, ssimLoss.item(), l1Loss.item(), loss_dict[1]['perceptual_loss'],
                                 loss_dict[1]['recon_loss'], loss_dict[0].item())

        optimizer.zero_grad()
        total_loss += loss_value.item()
        loss_value.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss_value.item())
        if i % 10 == 0:
            save_image(recon_image[0], f"output/reconstructed_{epoch}_{i}.png")
            save_image(image[0], f"output/original_{epoch}_{i}.png")
            save_image(output[0], f"output/output_{epoch}_{i}.png")

    train_logger.log_epoch(epoch)
    loss_list.append(total_loss / len(pbar))

    # 测试
    first_stage_model.eval()
    second_stage_model.eval()

    with torch.no_grad():
        total_test_loss = 0
        test_bar = tqdm(test_dataloader, desc=f"Testing Epoch {epoch + 1}/{epochs}", leave=False)
        for i, image in enumerate(test_bar):
            image = image.to(device)
            secret_path = generate_secret()

            secret = Image.open(secret_path).convert("RGB")
            secret = transformer(secret).unsqueeze(0).to(device)
            secret = secret.repeat(batch_size, 1, 1, 1).to(device)
            image_latent = first_stage_model.encode(image)
            secret_latent = first_stage_model.encode(secret)
            latent_object = 0.5 * image_latent + 0.5 * secret_latent
            recon_image = first_stage_model.decode(latent_object)
            recon_image_latent = second_stage_model.encode(recon_image)
            output = second_stage_model.decode(recon_image_latent)

            l1Loss = l1_loss(output, secret)
            ssimLoss = (1 - ssim(output, secret))
            global_step = (epoch + 1) * len(pbar)
            loss_dict = loss(recon_image, image, global_step)

            loss_value = 0.1 * l1Loss + 0.1 * ssimLoss + 0.8 * loss_dict[0]

            test_logger.log_metrics(i, ssimLoss.item(), l1Loss.item(), loss_dict[1]['perceptual_loss'],
                                    loss_dict[1]['recon_loss'], loss_dict[0].item())

            total_test_loss += loss_value.item()
            test_bar.set_postfix(loss=loss_value.item())
            if i % 10 == 0:
                save_image(recon_image[0], f"output/test_reconstructed_{epoch}_{i}.png")
                save_image(image[0], f"output/test_original_{epoch}_{i}.png")
                save_image(output[0], f"output/test_output_{epoch}_{i}.png")

        test_logger.log_epoch(epoch)
        loss_list.append(total_test_loss / len(test_bar))

train_logger.save_loss_plot(filename=f"loss_plot_{timestr}.png")
test_logger.save_loss_plot(filename=f"test_loss_plot_{timestr}.png")
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_plot.png')

torch.save(first_stage_model.state_dict(), f"second_process_{timestr}.pth")
torch.save(second_stage_model.state_dict(), f"output_process_{timestr}.pth")
