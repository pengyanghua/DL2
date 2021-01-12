# DL2
DL2 is a deep learning-driven scheduler for elastic training in deep learning clusters.

## Prerequisites
Install prerequisites:

```
pip install tensorflow-gpu==1.13.1
```

## Training
To train model, run the following command. It will start multiple threads to train a centralized model. Check [parameters.py](./parameters.py) if you want to change some hyper-parameters.

```
python train.py
```

## Trace
We put some traces collected from our testbed in [config_speed.txt](./config_speed.txt). You may need to collect your own trace if running on a different setup. For k8s setup, please check [Optimus](https://github.com/pengyanghua/optimus).

