import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
load_dotenv()
import re
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import gc

from io import BytesIO
import numpy as np
from PIL import Image
import PIL
import inspect
from typing import List, Optional, Union

from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import requests

class StableDiffusionImg2ImgPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: torch.FloatTensor,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
    ):

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
          raise ValueError(
              f'The value of strength should in [0.0, 1.0] but is {strength}')

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(
            self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # encode the init image into latents and scale the latents
        init_latents = self.vae.encode(init_image.to(self.device)).sample()
        init_latents = 0.18215 * init_latents

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size, dtype=torch.long, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape,
                            generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

def preprocess(image):
    w, h = image.size
    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

app = App(token=os.getenv('SLACK_BOT_TOKEN'))

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.getenv('YOUR_TOKEN')
GENERATED_FILEPATH = "./results/generated.png"
INIT_FILEPATH = "./results/init.png"
WIDTH = 384
HEIGHT = 512

usingUser = None
pipe = None
pipeI2I = None

@app.message(re.compile(r"^!img ([ a-zA-Z0-9!-/:-@¥[-`'{-~]+)$"))
def message_img(client, message, say, context):
    global usingUser
    global pipe
    global pipeI2I
    try:
        if usingUser is not None:
            say(f"<@{usingUser}> さんが画像を生成中ですのでしばらくお待ちください。")
        
        else:
            if pipe is None:
                del pipeI2I
                pipeI2I = None
                gc.collect()

                say(f"text-to-imageのモデルのローディングを行います。")
                print('Model(T2I) loading start.')
                pipe = StableDiffusionPipeline.from_pretrained(
                    MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
                pipe.to(DEVICE)
                print('Model(T2I) loading finished.')

            usingUser = message['user']
            prompt = context['matches'][0]
            say(f"<@{message['user']}> さんのプロンプト `{prompt}` で画像を生成します。1分程度お待ちください。")

            with autocast(DEVICE):
                print(f'Generating start. ')
                image = pipe(prompt, guidance_scale=7.5,
                            height=HEIGHT,
                            width=WIDTH,
                            num_inference_steps=100)["sample"][0]

                os.makedirs('./results', exist_ok=True)
                image.save(GENERATED_FILEPATH)
                print(f'Generating finished.')

            client.files_upload(
                channels=message['channel'],
                file=GENERATED_FILEPATH,
                title=prompt
            )

            say(f"<@{message['user']}> さんのプロンプト `{prompt}` の画像の生成が終わりました。")
            usingUser = None
    except Exception as e:
        usingUser = None
        print(e)
        say(f"エラーが発生しました。やり方を変えて試してみてください。 Error: {e}")


@app.message(re.compile(r"^!img-i <(https?://[\w/:%#\$&\?\(\)~\.=\+\-]+)> ([0-9\.]+) ([ a-zA-Z0-9!-/:-@¥[-`'{-~]+)$"))
def message_i2i(client, message, say, context):
    global usingUser
    global pipe
    global pipeI2I
    try:
        if usingUser is not None:
            say(f"<@{usingUser}> さんが画像を生成中ですのでしばらくお待ちください。")

        else:
            if pipeI2I is None:
                del pipe
                pipe = None
                gc.collect()

                say(f"image-to-imageのモデルのローディングを行います。")
                print('Model(I2I) loading start.')
                scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
                pipeI2I = StableDiffusionImg2ImgPipeline.from_pretrained(
                    MODEL_ID,
                    scheduler=scheduler,
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=YOUR_TOKEN
                ).to(DEVICE)
                print('Model(I2I) loading finished.')

            url = context['matches'][0] # Slack上ではURLは <URL> の形式になっている
            say(f"指定された画像のダウンロードを行います。")

            SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
            urlData = requests.get(url,
                allow_redirects=True,
                headers={'Authorization': f"Bearer {SLACK_BOT_TOKEN}"},
                stream=True
            ).content
            with open(INIT_FILEPATH, mode='wb') as f:
                f.write(urlData)

            usingUser = message['user']

            strength = float(context['matches'][1])
            if strength > 1.0:
                strength = 1.0
            elif strength < 0.0:
                strength = 0.0

            prompt = context['matches'][2]

            say(f"<@{message['user']}> さんのプロンプト `{prompt}` で強度 `{strength}` で元画像から画像を生成します。1分程度お待ちください。")

            with autocast(DEVICE):
                with open(INIT_FILEPATH, "rb") as fh:
                    init_img = Image.open(BytesIO(fh.read())).convert("RGB")
                    init_img = init_img.resize((WIDTH, HEIGHT))
                    init_image = preprocess(init_img)

                    print(f'Generating start. ')
                    image = pipeI2I(prompt,
                            guidance_scale=7.5,
                            init_image=init_image,
                            strength=strength,
                            num_inference_steps=100)["sample"][0]

                    os.makedirs('./results', exist_ok=True)
                    image.save(GENERATED_FILEPATH)
                    print(f'Generating finished.')

            client.files_upload(
                channels=message['channel'],
                file=GENERATED_FILEPATH,
                title=prompt
            )

            say(f"<@{message['user']}> さんのプロンプト `{prompt}` の元画像からの画像の生成が終わりました。")
            usingUser = None
    except Exception as e:
        usingUser = None
        print(e)
        say(f"エラーが発生しました。やり方を変えて試してみてください。 Error: {e}")

@app.message(re.compile(r"^!img-help$"))
def message_help(client, message, say, context):
    say("`!img [半角英数字記号で構成されるプロンプト]` の形式で画像の生成ができます。" + 
    "元画像指定する場合には、 `!img-i [Slack内の画像のURL] [0.0～1.0までの強度] [プロンプト]` としてください。\n" + 
    "生成には1分程度の時間がかかります。" + 
    "また、誰かが生成している際には実行できません。内部的にはStable Diffusionというモデルを利用しています。" +
     "そのため生成した画像のライセンスはCC0 1.0 Universal Public Domain Dedicationとなり誰にも著作権は発生しません。" + 
     "またプロンプト探しには https://lexica.art/ をご利用ください。")


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# アプリを起動します
if __name__ == "__main__":

    SocketModeHandler(app, os.getenv('SLACK_APP_TOKEN')).start()