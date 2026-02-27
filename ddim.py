import torch
from tqdm import tqdm
import logging
import numpy as np
import torch.nn.functional as F


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device="cuda"):
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
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def sample_ddim(self, model, batch_size, labels, ddim_timesteps=200,
                    ddim_discr_method="uniform", ddim_eta=0.0, clip_denoised=True):
        logging.info(f"Sampling {batch_size} new images....")

        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.noise_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.noise_steps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.noise_steps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            ddim_timestep_seq = ddim_timestep_seq + 1
            # previous sequence
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
            # start from pure noise (for each example in the batch)
            x = torch.randn((batch_size, 1, self.img_size, self.img_size)).to(self.device)
            x_array = torch.empty((batch_size, ddim_timesteps, self.img_size, self.img_size)).to(self.device)
            j = 0
            # x_array[:,j,:,:] = x[:,0,:,:]
            for i in tqdm(reversed(range(0, ddim_timesteps)), position=0):
                # k
                t = torch.full((batch_size,), ddim_timestep_seq[i]).long().to(self.device)
                # s
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i]).long().to(self.device)

                # 1. get current and previous alpha_cumprod
                # alpha_k
                alpha_cumprod_t = self._extract(self.alpha_hat, t, x.shape)
                # alpha_s
                alpha_cumprod_t_prev = self._extract(self.alphas_hat_prev, prev_t, x.shape)

                # 2. predict noise using model
                predicted_noise = model(x, t, labels)

                # 3. get the predicted x_0
                pred_x0 = (x-torch.sqrt((1.-alpha_cumprod_t))*predicted_noise)/torch.sqrt(alpha_cumprod_t)

                if clip_denoised:
                    # Clamps all elements in input into the range
                    pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

                # 4. compute variance: "sigma_t(η)"
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (
                                1 - alpha_cumprod_t / alpha_cumprod_t_prev))

                # 5. compute "direction pointing to x_t"
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * predicted_noise

                # 6. compute x_{t-1}
                # torch.randn_like Returns a tensor with the same size as input that is filled with random numbers
                # from a normal distribution with mean 0 and variance 1.
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)
                x_array[:,j,:,:] = x[:,0,:,:]
                j = j + 1
            x_array = x_array[:,np.newaxis,:,:,:]
        return x, x_array
