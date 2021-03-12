# Images Fire Detection



## Codes and Notebook
`data`: data folder

`logsCloud`: tensorboard files from hyperparameter optimization. To view, type `tensorboard --logdir logsCloud` in the terminal

`models`: folder to save models when run nootebooks to train models.

`1.first_run.ipynb`: The first train of MLP and CNN. We found that CNN model has far better performance. So, we decided to keep on developing CNN.

`2train_scalable.py`: The script train CNN model with our own generator function.
`3cloud_train.py`: This script is used to train hyperparameter optimization on GCP. It use datagenerator() from keras instrad of our own generator.
`amornsaensuk.py`: script to fro functions
`last_CNN_1.ipynb`: train CNN based on the best parameter from `3cloud_train.py`
`last_CNN_1_batchNorm`: adding batch normalization and adjust drop off layer from `last_CNN_1.ipynb`