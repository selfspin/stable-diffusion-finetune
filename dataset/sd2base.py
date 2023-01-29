from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import cv2
import numpy as np
import os


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusionv2base(nn.Module):
    def __init__(self, device, opt):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(
                f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.opt = opt
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.001)
        self.max_step = int(self.num_train_timesteps * 1)

        print(f'[INFO] loading stable diffusion...')
        model_key = "stabilityai/stable-diffusion-2-base"

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae",
                                                 use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer",
                                                       use_auth_token=self.token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key,
                                                          subfolder="text_encoder",
                                                          use_auth_token=self.token).to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",
                                                         use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler")
        # self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        #                         num_train_timesteps=self.num_train_timesteps,
        #                         clip_sample=False, set_alpha_to_one=False, steps_offset=1,
        #                         trained_betas=None)

        # self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        #                                num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def forward(self, image, label, guidance_scale=100):
        B = image.size(0)
        text_embeds = self.get_text_embeds(label, '')
        # interp to 512x512 to be fed into vae.

        # timestep ~ U(0.001, 1)
        t = torch.randint(self.min_step, self.max_step, [B], dtype=torch.long, device=self.device)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(image)

        # predict the noise residual with unet
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # latents_0 = self.scheduler.add_noise(torch.zeros_like(latents), torch.ones_like(noise), t)
        # for uncond/cond
        latent_model_input = torch.cat([latents_noisy] * 2)
        t_ = torch.cat([t] * 2, dim=0)
        # pred noise
        noise_pred = self.unet(latent_model_input, t_, encoder_hidden_states=text_embeds).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])[:, None, None, None]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        # grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        # latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return w, noise_pred, noise

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    device = torch.device('cuda')
    view_list = [ 'left', 'right', 'front left', 'front right', 'back left', 'back right', 'front', 'back']#['side', 'front', 'back', 'overhead']['side rear']#
    sd = StableDiffusionv2base(device, opt)
    file = 'stable-diffusion-finetune/checkpoints'
    for view in view_list:
        # text = opt.prompt+', '+view+' view'
        # text = 'the '+view+' side of '+opt.prompt
        # text = 'the '+view+'-hand side of '+opt.prompt
        # text = 'the '+view+' view of '+opt.prompt+', '+view+' view'
        # text = 'the '+view+' view of '+opt.prompt+' from the '+view+' view'
        text = 'the '+view+' hand side of '+opt.prompt+' from the '+view+' view'

        print(text)
        for lr in ['1e-08','2e-08','1.5000000000000002e-08']:# ['1e-08','5e-08', '8e-08', '1e-07', '5e-07']:
            # for epoch in ['gt_image']:
            for epoch in os.listdir(os.path.join(file, lr)):
                sd.load_state_dict(torch.load(os.path.join(file, lr, epoch)))
                for index in range(20):
                    seed_everything(opt.seed + index)

                    imgs = sd.prompt_to_img(text, opt.negative, opt.H, opt.W, opt.steps)

                    # visualize image
                    os.makedirs(os.path.join('2dbase', str('hand side  of '+opt.prompt+' from the '), str(view), lr, epoch[:6]), exist_ok=True)
                    plt.imsave(os.path.join('2dbase', str('hand side  of '+opt.prompt+' from the '), str(view), lr, epoch[:6], str(index) + '.png'), imgs[0])
                    # plt.imshow(imgs[0])
                    # plt.show()
