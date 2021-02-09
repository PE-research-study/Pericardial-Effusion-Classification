# -*- coding: utf-8 -*-
# @Date               : 2021-01-29 16:56:36
# @Author             : Haoyu Kong, WZMIAOMIAO
# @Python version     : 3.6.12
# @Tensorflow version : 2.0.0
# @cudatoolkit version: 10.0.130
# @cudnn version      : 7.6.5

# This is the script used when training the VGGNet model. When testing the PEAD framework,
# you need to load the database in different steps by changing the database path (data_path in line 42).
# It is also necessary to test different models by changing the model setting in line 124 of this file.

import  os
# Hide tensorflow error message
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Set which GUP to use at runtime
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import numpy as np
import pandas as pd
import pathlib
from VGG_model import vgg
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  tensorflow as tf
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Limit the amount of VRAM to the extent required by the model, since tensorflow will automatically use all VRAM when running the deep learning model.
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)

# Set the path of the dataset and the path where the weight "*.h5" file is saved
data_path = pathlib.Path(r'D:\NMDID\NMDID_PEdata')
weight_path = r'C:\Users\Haoyu Kong\Desktop\DL\VGGNet\weight\PE-VGG'

# The prefix of the tensorboard file name
TOT = '#PE-VGG8#'

# Set the image parameters, class number, batch size learning rate, Cross Validation fold number and the number of training epochs
im_height = 224
im_width = 224
class_num = 2
batch_size = 16
learnr = 1e-5
fold_num = 10
epochs = 30

# Extract all PMCT slices' paths and shuffle the order
all_image_paths = list(data_path.glob('*/*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)

# Find the label names of all categories, and correspond each PMCT slices to the label name according to its path.
label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[(pathlib.Path(path).parent.parent.name)] for path in all_image_paths]

# Set the number of PMCT slices in each Cross Validation fold
fold_size = int(image_count//fold_num)

# Create a reusable index list for Cross Validation in case of insufficient VRAM.
def creat_idx(image_count):
    idx = tf.range(image_count)
    idx = tf.random.shuffle(idx)
    idxlist = list(idx.numpy())
    idxdf = pd.DataFrame({'idx':idxlist})
    idxdf.to_csv('idx.csv', index = False, sep=',')

#creat_idx(image_count)

def read_idx(filename):
    idxdf = pd.read_csv(filename)
    idxlist = idxdf['idx'].values.tolist()
    idx = tf.cast(idxlist, dtype=tf.int32)
    return idx

idx = read_idx('idx.csv')

# Loading and preprocessing of all PMCT slices and corresponding labels
def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)

    # Resize the slice and limit the pixel value to (-1, 1)
    image = tf.image.resize(image, [im_height, im_width])
    image = 2*tf.cast(image, dtype=tf.float32) / 255.-1
    label = tf.cast(label, dtype=tf.int32)

    return image, label

# Draw the loaded slices. Used to test whether the data is loaded correctly
def plotImages(images_path):
    for i in range(len(images_path)):
        img = imread(images_path[i])
        plt.imshow(img)
        plt.show()

# Training and testing
def main():
    # Define the variables that need to be used in the summary
    CV_summary = []
    CV_CM = []
    CV_BestModel = []
    OA_CM = np.array([[0,0],[0,0]])
    t_CV = time.perf_counter()

    # 10 iterations for Cross Validation
    for fold in range(1, fold_num+1):

        # Overall time recorder
        t_fold = time.perf_counter()

        # Initialize the model
        model = vgg("vgg8", im_height, im_width, class_num)
        model.summary()

        # Set optimizer
        optimizer = optimizers.Adam(lr=learnr)

        # Set the measurement values: training loss, training accuracy, test loss and test accuracy
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        # Select the training set and test set based on the index list corresponding to the Cross validation fold
        test_idx = idx[(fold-1)*fold_size:fold*fold_size]
        train_idx = list(set(idx.numpy())-set(test_idx.numpy()))
        test_x, test_y = tf.gather(all_image_paths, test_idx),tf.gather(all_image_labels,test_idx)
        train_x, train_y = tf.gather(all_image_paths,train_idx),tf.gather(all_image_labels,train_idx)

        # Create tensorflow training set and test set
        test_CT = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_db = test_CT.shuffle(10000).map(load_and_preprocess_from_path_label).batch(batch_size)
        train_CT = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_db = train_CT.shuffle(10000).map(load_and_preprocess_from_path_label).batch(batch_size)

        # Establish the file path that can be used in tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        CV_log_dir = 'logs/CV' + TOT + str(fold) + '%%' + current_time
        CV_summary_writer = tf.summary.create_file_writer(CV_log_dir)

        # Initialize summary list for each epoch and the critical value for best model selection
        Epoch_summary = []
        epsilon = 0
        for epoch in range(1,epochs+1):

            # clear history information
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

            # Time recorder for training an epoch
            t1 = time.perf_counter()

            # Train one batch images each time
            for step, (x,y) in enumerate(train_db):

                with tf.GradientTape() as tape:

                    # Get model output
                    logits = model(x, training=True)
                    # Convert labels into one hot encoding
                    y_onehot = tf.one_hot(y, depth=2)
                    # Compute loss
                    loss = tf.losses.binary_crossentropy(y_onehot, logits, from_logits=False)
                    loss = tf.reduce_mean(loss)
                    # Get training loss and training accuracy
                    train_loss(loss)
                    train_accuracy(y_onehot, logits)

                # Compute gradient
                grads = tape.gradient(loss, model.trainable_variables)
                # Update parameters
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Create tensorboard records
            with CV_summary_writer.as_default():
                tf.summary.scalar('train-CrossEntropy', float(train_loss.result()), step=epoch)
                tf.summary.scalar('train-Accuracy', float(train_accuracy.result()), step=epoch)
            print('-----------------------------------------------------------------')
            print('Training time: ',time.perf_counter() - t1)

            # Set test summary measurements
            test_pred = []
            test_GT = []
            t2 = time.perf_counter()

            # Perform a test after completed each epoch training
            for x,y in test_db:
                # Get predict result
                logits = model(x, training=False)
                pred = tf.argmax(logits, axis=1)
                pred = tf.cast(pred, dtype=tf.int32)
                test_pred.extend(pred)
                test_GT.extend(y)
                yt_onehot = tf.one_hot(y, depth=2)
                # Compute loss
                t_loss = tf.losses.binary_crossentropy(yt_onehot, logits, from_logits=False)
                test_loss(t_loss)

            # Compute confusion matrix
            CM = tf.math.confusion_matrix(test_GT,test_pred)
            # Compute test accuracy, precision and re-call
            TP = CM[1,1]
            TN = CM[0,0]
            FP = CM[0,1]
            FN = CM[1,0]

            Acc = ((TP + TN) / (TP + TN + FP + FN))
            Precision = TP / (TP + FP)
            Re_call = TP / (TP + FN)

            # Select best model and save weights in certain path
            if Acc > epsilon:
                file_name = weight_path + r'\\' + TOT + '-Fold-' + str(fold) + '-BestWeight.h5'
                epsilon = Acc
                model.save_weights(file_name)
                # Save best model's statistics and confusion matrix
                BestModel = [train_loss.result().numpy(),
                          train_accuracy.result().numpy(),
                          test_loss.result().numpy(),
                          Acc.numpy(),
                          Precision.numpy(),
                          Re_call.numpy()]
                BestCM = np.array(CM)

            # Compute the average statistics when convergence
            if epoch > epochs/2:
                summary = [train_loss.result().numpy(),
                          train_accuracy.result().numpy(),
                          test_loss.result().numpy(),
                          Acc.numpy(),
                          Precision.numpy(),
                          Re_call.numpy()]
                Epoch_summary.append(summary)

            print('Test time: ', time.perf_counter() - t2)
            template1 = 'Fold {}, Epoch {}'
            template2 = 'Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template1.format(fold, epoch))
            print(np.array(CM))
            print(' Precision:', float(Precision), ' Re_call:', float(Re_call))
            print(template2.format(loss,
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  Acc))
            print('Current Best Model:')
            print(BestCM)

            print(' Precision:', float(BestModel[4]), ' Re_call:', float(BestModel[5]))
            print(template2.format(BestModel[0],
                                   BestModel[1],
                                   BestModel[2],
                                   BestModel[3],))
            print('-----------------------------------------------------------------')

            with CV_summary_writer.as_default():
                tf.summary.scalar('test-loss', float(test_loss.result()), step=epoch)
                tf.summary.scalar('test-Accurency', float(Acc), step=epoch)
                tf.summary.scalar('test-Precision', float(Precision), step=epoch)
                tf.summary.scalar('test-Re_call', float(Re_call), step=epoch)

        print('Fold time: ', time.perf_counter() - t_fold)
        print('Summary for fold: ',fold)
        print(Epoch_summary)
        epoch_mean = np.mean(Epoch_summary,axis=0)
        print('Mean:')
        print(epoch_mean)
        print('Best Model:')
        print(BestModel)
        CV_summary.append(epoch_mean)
        CV_CM.append(BestCM)
        CV_BestModel.append(BestModel)
        # Compute overall confusion matrix
        OA_CM += BestCM
    print('_________________________________________________________________')
    print('Cross validation summary: ')
    print('Total time: ', time.perf_counter() - t_CV)
    print('Fold means: ')
    print(CV_summary)
    print('Overall Mean:')
    print(np.mean(CV_summary,axis=0))
    print('_________________________________________________________________')
    print('Best Models:')
    print(CV_BestModel)
    print('Confusion Matrices:')
    print(CV_CM)
    print('_________________________________________________________________')
    print('Overall Confusion Matrix:')
    print(OA_CM)
    # Compute overall accuracy, precision and re-call
    TP = OA_CM[1, 1]
    TN = OA_CM[0, 0]
    FP = OA_CM[0, 1]
    FN = OA_CM[1, 0]

    OA_Acc = ((TP + TN) / (TP + TN + FP + FN))
    OA_Precision = TP / (TP + FP)
    OA_Re_call = TP / (TP + FN)
    print('Overall Test Accuracy: ',float(OA_Acc))
    print('Overall Precision: ',float(OA_Precision))
    print('Overall Re-call: ',float(OA_Re_call))





if __name__ == '__main__':
    main()