# Decision by Supervised Learning with Deep Ensemble

This repository contains the experiments and code for the paper **"Decision by Supervised Learning with Deep Ensemble: A Practical Framework for Robust Portfolio Optimization"**

## Requirements
All dependencies are listed in `requirements.txt`. To install them, simply run:
```bash
pip install -r requirements.txt
```

## Target Portfolio caculation
You can get target portfolio by:

```bash
python portfolio_optimization.py -f <universe file name>
```
The <universe_file_name> should reference an OHLCV dataset with one date per row. For details on the file format, see the preprocess_data function in src/utils.py.


## Model Training
Once you have created the target portfolio, you can train the model using:

```bash
CUDA_VISIBLE_DEVICES=<GPU Number> python experiment.py -u <Universe> -p <Portfolo> -o <Model> -z sophia
```
<Universe> specifies the universe file.
<Portfolio> indicates the target portfolio type (e.g., Max-Sortino or One-Hot).
<Model> can be LSTM, Transformer, or Mamba.
-z sophia sets the optimizer to Sophia.

## Deep Ensemble experiment
To run the Deep Ensemble experiments and analyze performance under varying ensemble sizes, use:

```bash
CUDA_VISIBLE_DEVICES=<GPU_Number> python experiment_ensemble.py -u <Universe> -p <Portfolio> -o <Model> -z sophia -r 1000
```
Here, -r 1000 denotes the number of model samples to generate for the ensemble.

## Analysis
The notebooks portfolio.ipynb, backtest.ipynb, and ensemble.ipynb contain scripts and visualizations for analyzing the experiment results.

## Paper
My paper is at https://arxiv.org/abs/2503.13544 <br>
Citation: <br>
```

```
