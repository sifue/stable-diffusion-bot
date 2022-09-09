import requests
import re
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
load_dotenv()

print("SLACK_BOT_TOKEN")
print(os.getenv('SLACK_BOT_TOKEN'))
# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.getenv('SLACK_BOT_TOKEN'))

# 'hello' を含むメッセージをリッスンします
# 指定可能なリスナーのメソッド引数の一覧は以下のモジュールドキュメントを参考にしてください：
# https://slack.dev/bolt-python/api-docs/slack_bolt/kwargs_injection/args.html


@app.message(re.compile("(hi|hello|hey)"))
def message_hello(client, message, say):
    # イベントがトリガーされたチャンネルへ say() でメッセージを送信します
    say(f"Hey there <@{message['user']}>!")

    client.files_upload(
        channels=message['channel'],
        file="results/generated.png",
        title=message['text']
    )


@app.message(re.compile(r"^dl <(https?://[\w/:%#\$&\?\(\)~\.=\+\-]+)>$"))
def message_dl(client, message, say, context):
    url = context['matches'][0]  # Slack上ではURLは <URL> の形式になっている
    print(url)

    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')

    say(f"Download! by <@{message['user']}>!")

    urlData = requests.get(url,
                           allow_redirects=True,
                           headers={
                               'Authorization': f"Bearer {SLACK_BOT_TOKEN}"},
                           stream=True
                           ).content
    with open("./results/init.png", mode='wb') as f:
        f.write(urlData)


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# アプリを起動します
if __name__ == "__main__":
    print("SLACK_APP_TOKEN")
    print(os.getenv('SLACK_APP_TOKEN'))
    SocketModeHandler(app, os.getenv('SLACK_APP_TOKEN')).start()
