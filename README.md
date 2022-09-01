# Stable Diffusion Bot
日本語でプロンプトを与えるとStable Diffusionの画像を投稿してくれるSlack Bot  
RTX2070 8GB環境とGTX2060 6GB環境で操作確認済み。

## 環境導入と実行
Stable Diffusionの公式版が動く環境の整備をした前提で以下を実行。([構築方法参考](https://zenn.dev/koyoarai_/articles/02f3ed864c6127bb2049))

```
conda env create -f environment.yaml
conda activate sdbot
```
以上を行って `.env` ファイルにHuggingfaceのトークンと、SlackのAppトークンとBotトークンを設定。

```
YOUR_TOKEN=hf_xxxxxxxxxxxxxx
SLACK_BOT_TOKEN=xoxb-999999999999999999999999
SLACK_APP_TOKEN=xapp-999999999999999999999999
```

設定後、実行は以下のコマンドの通り。

```
python3 script/main.py
```
## Stable Diffusion Botの使い方

- 画像生成: !img [プロンプト]
- ヘルプ表示: !img-help

## Slackのアプリに必要な権限
アプリの作り方は[このドキュメント](https://slack.dev/bolt-python/ja-jp/tutorial/getting-started)に準拠。
必要権限は以下。

### OAuth & Permissions - Bot Token Scopes
- chat:write
- files:write

### Event Subscriptions
- message.channels
- message.groups
- message.im
- message.mpim 

## トラブルシューティング

以下のエラーが出る。

```
RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

ビデオメモリが足りません。 `main.py` のコード内の画像のサイズを指定しているところを512ではなく8の倍数の384か256にして対応してください。

## 参考ドキュメント
- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
- [image-2-image using diffusers](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb#scrollTo=V24njWQBC8eC)
