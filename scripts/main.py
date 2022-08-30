import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
load_dotenv()
import re
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

app = App(token=os.getenv('SLACK_BOT_TOKEN'))

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.getenv('YOUR_TOKEN')
GENERATED_FILEPATH = "./results/generated.png"

print('Model loading start.')
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)
print('Model loading finished.')


usingUser = None


@app.message(re.compile(r"^!img ([ a-zA-Z0-9!-/:-@¥[-`{-~]+)$"))
def message_img(client, message, say, context):
    global usingUser

    if usingUser is not None:
        say(f"<@{usingUser}> さんが画像を生成中ですのでしばらくお待ちください。")
    
    else:
        usingUser = message['user']
        prompt = context['matches'][0]
        say(f"<@{message['user']}> さんのプロンプト `{prompt}` で画像を生成します。1分程度お待ちください。")

        with autocast(DEVICE):
            print(f'Generating start. ')
            image = pipe(prompt, guidance_scale=7.5,
                        height=512,
                        width=512,
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


@app.message(re.compile(r"^!img-help$"))
def message_help(client, message, say, context):
    say(f"`!img [半角英数字記号で構成されるプロンプト]` の形式で画像の生成ができます。1分程度の時間がかかります。また誰かが生成している際には実行できません。内部的にはStable Diffusionというモデルを利用しています。そのため生成した画像のライセンスはCC0 1.0 Universal Public Domain Dedicationとなり誰にも著作権は発生しません。またプロンプト探しには https://lexica.art/ をご利用ください。")


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# アプリを起動します
if __name__ == "__main__":

    SocketModeHandler(app, os.getenv('SLACK_APP_TOKEN')).start()