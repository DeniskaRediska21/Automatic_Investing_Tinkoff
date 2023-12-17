# Automatic Tinkoff Investing

Investing bot that works through the python Tinkoff sdk. Bot trains a 1DConvolution
neural network on the provided FIGI hysrorical data and tryes to predict entry
points for the short trade.

Bot uses for entry point prediction such metrics as:
+ Closing price
+ Low price
+ High price
+ Volume
+ SRSI
+ KAMA
+ TSI
+ UA
+ MACD


**To use the bot yourself create a config.py file and paste:** `token = '<Your tinkoff invest token>'`
