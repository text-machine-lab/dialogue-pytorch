"""Sends Telegram alerts using Telegram bot with given authorization token, and sends it to client
specified with client id"""
import os
import sys
from telegram import Bot

class TelegramAlert:
    def __init__(self, config=None):
        """Using auth token and chat id found in the config file,
        send a message to user.

        config: a file containing two lines. first line is auth token, second is client id
        """
        if config is None:
            config = os.path.join(os.environ['HOME'], '.tg-config')

        self.bot_token = None
        self.chat_id = None

        # send message upon exceptionv
        self.prev_hook = sys.excepthook
        sys.excepthook = self.alerthook

        if os.path.exists(config):
            with open(config, 'r') as f:
                self.bot_token = next(f).strip()
                self.chat_id = next(f).strip()

    def write(self, text):
        if self.bot_token is not None and self.chat_id is not None:
            bot = Bot(token=self.bot_token)
            bot.send_message(chat_id=self.chat_id, text=text)

    def alerthook(self, type, value, traceback):
        if type is not KeyboardInterrupt and type is not SyntaxError:
            self.write(__file__ + ': ' + repr(value))
        self.prev_hook(type, value, traceback)

if __name__ == '__main__':
    # Test code
    alert = TelegramAlert()
    alert.write('Telegram alert!')

    raise RuntimeError