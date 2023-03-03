import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from utils import plot_images, save_images, get_data, setup_logging
from modules import UNet

logging.basicConfig(format="%(asctime)s  - %(levelname)s - %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # we will use the linear beta schedule for simplicity here
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return beta

    def noise_images(self, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # in this case sef.alpha_hat[t] is a scalar, so we expand it to the same shape as x for element-wise multiplication
        sqrt_alpha_hat: torch.Tensor = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat: torch.Tensor = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e: torch.Tensor = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n: int):
        # We generate random integers from 1 to noise_steps.
        # The size of the tensor is (n,) which means we will generate a 1-dimensional tensor of size n.
        # This tensor will cointain 'n' random sampled time steps, each representing the index of the noise level for
        # each image in the batch.
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        We sample n images.
        This algorith can be found in the DDPM paper : https://arxiv.org/pdf/2006.11239.pdf page 4, algorithm 2

        :param model:
        :param n:
        :return:
        """
        # In the DDPM paper (algorithm 2) we can find this algorithm
        logging.info(f"Sampling {n} images...")
        model.eval()
        with torch.no_grad():
            # we sample a random image (noise) for each image in the batch.
            # 3 is the number of channels for the images (RGB)
            # our images are square, so we use the same size for height and width
            # now x contains a random noise tensor of size (n, 3, img_size, img_size)
            # randn is normally distributed (torch.randn) with mean 0 and variance 1
            # this is what the "N" in the paper stands for. Normal distribution.
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)  # these are our initial images

            # this is in reverse order, so we start from the last timestep. We want to denoise the images and
            # the x represent complete noised images. We want to go back from noised to denoised images.
            # When we see the "steps" when generating an image, we are talking about noise_steps.
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # t is the step in the algorithm. It is a tensor of size (n,), because torch.ones(n)
                # returns a tensor of size (n,). We need to convert it to long because we don't want to use float for
                # indexing discrete steps.
                t = (torch.ones(n) * i).long().to(self.device)
                # the model predict the noise level for each image in the batch
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    # The last step we don't want more noise, so we set the noise to zero
                    noise = torch.zeros_like(x)
                # we denoise the images using the formula from the paper
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        # we clamp the values of x to [-1,1]. Anything bellow -1 will be -1 and above 1 will be 1.
        # then we sum +1 and the range will be [0,2] and divide by 2 and the range will be [0,1]
        x = (x.clamp(-1, 1) + 1) / 2
        # We need to return an image so the values should be from 0 to 256
        x = (x * 256).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    lendl = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch} / {args.epochs}:  ")
        # This is the algorithm 1 from the DDPM paper
        pbar = tqdm(dataloader)
        # the dataloader will return a batch of images
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            # we sample random time steps for each image in the batch
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # we denoise the images using the formula from the paper
            x_t, noise = diffusion.noise_images(images, t)
            # we predict the noise level for each image in the batch
            predicted_noise = model(x_t, t)
            # we calculate the loss
            loss = mse(noise, predicted_noise)
            # we backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # we log the loss
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * lendl + i)

        # we sample some images
        sampled_images = diffusion.sample(model, n=images.shape[0])
        # we log the images
        save_images(sampled_images, os.path.join("results", args.run_name, f"epoch_{epoch}.png"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500
    args.lr = 3e-4
    args.batch_size = 12
    args.img_size = 64
    args.device = "cuda"
    args.dataset_path = "e:\datasets\landscape_dataset"
    args.num_workers = 4
    train(args)


if __name__ == '__main__':
    launch()