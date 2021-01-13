# DL2
DL2 is a deep learning-driven scheduler for elastic training in deep learning clusters. DL2 advocates a joint supervised learning and reinforcement learning approach: a neural network is warmed up via offline supervised learning based on job traces produced by the existing cluster scheduler; then the neural network is plugged into the live DL cluster, fine-tuned by reinforcement learning carried out throughout the training progress of the DL jobs, and used for deciding job resource allocation in an online fashion.

Check [this figure](./workflow.pdf) for the overall workflow illustration.


## Prerequisites
Install prerequisites:

```
pip install tensorflow-gpu==1.13.1
```

## Training Workflow
To train model, run the following command. It will start multiple threads to train a centralized model. Check [parameters.py](./parameters.py) if you want to change some hyper-parameters.

```
python train.py
```


## Trace
We put some traces collected from our testbed in [config_speed.txt](./config_speed.txt). You may need to collect your own trace if running on a different setup. For k8s setup, please check [Optimus](https://github.com/pengyanghua/optimus).

## Publication
To Add.
