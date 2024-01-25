# Exploring order book predictability in cryptocurrency markets in a deep learning perspective using JAX

[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-View%20Notebook-blue?logo=google-colab)](https://colab.research.google.com/github/toma-x/exploring-order-book-predictability/blob/main/Exploring-book-predictability.ipynb)

## Overview

This project explores predictability in cryptocurrency markets, focusing on Bitcoin, the most widely known and liquid cryptocurrency. The analysis is conducted using deep learning techniques, specifically a Convolutional Neural Network (CNN), implemented with the Flax library and [JAX](https://jax.readthedocs.io/en/latest/).
Our experiment show clear sign of short/mid term predictability in a trading perspective.

## Business understanding

The [order book](https://www.investopedia.com/terms/o/order-book.asp) contains the bid and ask orders placed by all the market participants, this information convolve a lot of information on the current state of a given market. While this is difficult to use the order book and read it with the naked eye, a machine can efficiently read and extract crucial information from the order book. The objective of our analysis is to identify order books leading to movements in the market.

## Data dowload and processing

The work in [5] suggests that the first level is the most informative.
Guided by this, the data we use is the BTCUSDT book ticker (the first level of the order book), publicly available for the first day of each month on Tardis.dev. 
The notebook will automatically fetche historical order book data for BTCUSDT for the past six months, process the data, and prepare it for training.

## Training the Model

Recent developments in [1,2,3] show that a CNN can accurately learn from time series data, our model is also a CNN implemented with Flax and JAX.
Hyperparameters are tuned using the framework [Optuna](https://optuna.readthedocs.io/en/stable/index.html) and the model is trained on 10 epochs.
We log the metrics of the evaluation of the unseen test dataset, achieving 86% accuracy.


## Evaluation and Trading Strategy
The trading strategy is implemented based on the predictions, considering latency, slippage, and fees. The performance of the strategy is discussed and visualized, including wallet value over time and the Sharpe ratio, as in [4].
Results are presented for different confidence levels, and the impact of latency, slippage, and fees on the trading strategy is underlined.

## Conclusion

This project provides good insights into the predictability of order book data in cryptocurrency markets using a deep learning approach. It demonstrates the implementation of a trading strategy based on the model's predictions and evaluates its performance under realistic trading conditions. The use of such a predictive system could easily be wrapped within a trading bot for automated use, if able of consistently producing reliable predictions.

## References

 - [1] [DeepLOB: Deep convolutional neural networks for limit order books](https://arxiv.org/abs/1808.03668) - Zhang Z, Zohren S, Roberts S.
 - [2] [Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141) - Kolm, Petter N. et al.
 - [3] [THE SHORT-TERM PREDICTABILITY OF RETURNS IN ORDER BOOK MARKETS: A DEEP LEARNING PERSPECTIVE](https://arxiv.org/pdf/2211.13777.pdf) - Lucchese L, S.Pankkanen M, E.D.Veraart A.
 - [4] [Order Flow Imbalance - A High-Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html) - Dean Markwick.
 - [5] [How informative is the Order Book Beyond the Best Levels? Machine Learning Perspective](https://arxiv.org/pdf/2203.07922.pdf) - Tran D, Kanniainen J, Iosifidis A.
