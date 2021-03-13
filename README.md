# Images Fire Detection


Fire detection is a problem that people have been trying to solve for a longtime. It is critical to detect the fire at the earliest stage in order to stop the fire and evacuate people in time. For indoor cases, there are various systems and equipment which are used such as CO2 sensors, smoke detectors, temperature sensors (heat detectors), etc. Sometimes the system alone had a failure rate of 0.32% (Carter, 2008). Nowadays, the use of security cameras has dramatically increased, with an estimation of 85 million cameras in the US in 2021 (Lin & Purnell, 2019). Leveraging the use of a security camera for early detection of fire will potentially help improve the rate of detection. The task can be effectively achieved by using deep learning algorithms to detect fire in images. Furthermore, image fire detection is also beneficial for outdoor cases such as wildfire where fire prevention systems are impossible to be installed. This project will present various deep learning models that are used to classify fire and no fire images.


## Codes and Notebook
`data`: data folder

`logsCloud`: tensorboard files from hyperparameter optimization. To view, type `tensorboard --logdir logsCloud` in the terminal

`models`: folder to save models when run nootebooks to train models.

`1.first_run.ipynb`: The first train of MLP and CNN. We found that CNN model has far better performance. So, we decided to keep on developing CNN.

`2train_scalable.py`: The script train CNN model with our own generator function.

`3cloud_train.py`: This script is used to train hyperparameter optimization on GCP. It use datagenerator() from keras instrad of our own generator.

`amornsaensuk.py`: script to for functions to be imported

`last_CNN_1.ipynb`: train CNN based on the best parameter from `3cloud_train.py`

`last_CNN_1_batchNorm`: adding batch normalization and adjust drop off layer from `last_CNN_1.ipynb`


`transfer_learning_model`: Inside this directory we have the entire code that we used train the tranfer learning model

`transfer_learning_model/scripts/`: In this directory all the transfer learning code exists

`transfer_learning_model/rename.py`: Script used to rename the testing images

`transfer_learning_model/results.csv`: Final results reside inside this file after completing all the testing

`transfer_learning_model/results.ipynb`: Pre-processing the testing images dataset to get them ready for finally running into test.py

`transfer_learning_model/test.py`: Script to get the output of model on a different testing directories

## Data Source

For the data source we have decided to use the following [repositrory](https://github.com/cair/Fire-Detection-Image-Dataset).
This image dataset contains images of three different classes:

- Neutral
- Smoke
- Fire

For the purpose of this assignment we only kept two classes: `Fire` and `Neutral`. 
Originally this dataset in unbalanced but we decided to make it balanced. Before the preprocessing step we accumulated 1000 images for each class. 

In order to obtain a larger dataset and better results we have decided to apply some data augmentaion:

- Crop images to 150 by 150px
- Mirror the images
- Rotate images

That allowed us to increase the count of the images in each class to be 8000. 6000 images of each class would be Training and 2000 Testing. 
