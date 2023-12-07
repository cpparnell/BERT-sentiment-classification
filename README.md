# BERT-sentiment-analysis

An exploration into training a [BERT model](https://huggingface.co/bert-base-uncased) for sentiment classification using [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders).

## Setting Up and Running the Code

**You can skip this part if you are not interested in doing this experiment for yourself, but are interested in the results.**

### Setup

Must install miniforge3 so that we can train models using Mac ARM architecture:

[Install the latest for OS X arm64](https://github.com/conda-forge/miniforge#miniforge3)

Next, run:
```
conda create --name myenv python=3.11
conda activate myenv
```
Make sure the conda environment is active before running any code in this project.

Then to install our needed packages from 🤗
```
pip install transformers accelerate datasets
```

Install pytorch nightly to allow Metal Performance Shaders usage:
```
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

### Training the Model

If you would like to run the model for yourself, run the following:
```
python training.py
```
Play around with the training parameters and see what happens.

### Playground

```playground.py``` allows you to mess around with trained models. To play around with your own models, please update ```PATH_TO_MODEL``` with the path to your trained model and ```TEXTS``` with strings to classify.