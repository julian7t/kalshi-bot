# Kalshi Trading Bot

## Requirements
Python 3.11+

## Install dependencies
```
pip install requests cryptography python-dotenv
```

## Configure secrets
Create a file called .env in this folder:

```
KALSHI_API_KEY_ID=your-key-id-here
KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=5061063023
LIVE_MODE=true
```

IMPORTANT: The private key must be the full PEM block.
If your key is in a .pem file, paste the entire contents as one value,
replacing actual newlines with \n.

## Run
```
python3 bot.py
```

## Run 24/7 in background (Linux/Mac)
```
nohup python3 bot.py >> logs/bot.log 2>&1 &
```

## Keep alive with screen (recommended)
```
screen -S kalshi
python3 bot.py
# Ctrl+A then D to detach — bot keeps running
# screen -r kalshi to re-attach
```
