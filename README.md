# Exploring order book predictability in cryptocurrency markets in a deep learning perspective using JAX

## Overview

This project explores order book predictability in cryptocurrency markets, focusing on Bitcoin, the most widely known and liquid cryptocurrency. The analysis is conducted using deep learning techniques, specifically a Convolutional Neural Network (CNN), implemented with the Flax library and [JAX](https://jax.readthedocs.io/en/latest/).
Our experiment show clear sign of short/mid term predictability in a trading perspective.

## Data

The data used is the BTCUSDT book ticker, publicly available for the first day of each month on Tardis.dev. The data is downloaded using the tardis-dev Python library. 

## References

- [DeepLOB: Deep convolutional neural networks for limit order books](https://arxiv.org/abs/1808.03668) - Zhang Z, Zohren S, Roberts S.
- [Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141) - Kolm, Petter N. et al.
- [THE SHORT-TERM PREDICTABILITY OF RETURNS IN ORDER BOOK MARKETS: A DEEP LEARNING PERSPECTIVE](https://arxiv.org/pdf/2211.13777.pdf) - Lucchese L, S.Pankkanen M, E.D.Veraart A.
- [Order Flow Imbalance - A High-Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html) - Dean Markwick.

## Data Download and Processing

The notebook fetches historical order book data for BTCUSDT for the past six months, processes the data, and prepares it for training. The model is trained on the first level of the order book, including bid and ask amounts and the spread.

## Training the Model

Recent developments in [1,2,3] show that a CNN can accurately learn from time series, our model is also a CNN implemented with Flax and JAX. Hyperparameters are tuned using Optuna and the model is trained on 10 epochs, then evaluated on a separate test dataset, achieving 86% accuracy.

## Evaluation and Trading Strategy
The trading strategy is implemented based on the predictions, considering latency, slippage, and fees. The performance of the strategy is discussed and visualized, including wallet value over time and the Sharpe ratio, as in [4].

## Results and Analysis

Results are presented for different confidence levels, and the impact of latency, slippage, and fees on the trading strategy is underlined.

## Conclusion

This project provides good insights into the predictability of order book data in cryptocurrency markets using deep learning. It demonstrates the implementation of a trading strategy based on the model's predictions and evaluates its performance under realistic trading conditions.
