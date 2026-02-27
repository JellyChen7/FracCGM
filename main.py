import copy
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from modules import UNet_conditional
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn.functional as F


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        # Returns the cumulative product of elements of input in the dimension dim.
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alphas_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_train_data(args)
    model = UNet_conditional(1, 3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    l = len(dataloader)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("./output/logs", datetime.now().strftime('%Y%m%d_%H%M')
                                        + '_ndata{}'.format(args.train_number)))  # /output/

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # if np.random.random() < 0.1:
            #    labels = None
            predicted_noise = model(x_t, t, labels)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join("./models", args.run_name, "ckpt{}.pt".format(epoch + 1)))
            # torch.save(optimizer.state_dict(), os.path.join("./models", args.run_name, "optim{}.pt".format(epoch + 1)))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 64
    args.epochs = 600  # 300
    args.image_size = 128
    args.train_number = 1000
    args.device = "cpu"
    args.lr = 2e-4
    args.mse = True
    args.run_name = 'DDIM_length' + datetime.now().strftime('%Y%m%d_%H%M') + '_ntrain{}'.format(args.train_number)
    train(args)


if __name__ == '__main__':
    launch()