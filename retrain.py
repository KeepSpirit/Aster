import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from mixup_utils import mixup_data, compute_loss, compute_loss_100, compute_acc
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_retrain(epoch):
    lr = 1e-4
    # lr = 1e-3
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "adapt_duration": [5, 5, 10, 10]}

    # parameters
    param_index = 1
    dataset_name = param_dict["dataset_name"][param_index]
    model_architecture = param_dict["model_architecture"][param_index]
    num_classes = param_dict["num_classes"][param_index]
    adapt_duration = param_dict["adapt_duration"][param_index]
    ae_generation_technique = "robot"      # ["dlfuzz", "adapt", "robot"]

    # path
    data_architecture = dataset_name + "_" + model_architecture
    base_path = "./checkpoint/" + data_architecture + "/"
    original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
    original_model_path = base_path + "/mixup_models/" + data_architecture + ".h5"
    ae_data_path = base_path + ae_generation_technique + "/"
    retrain_path = base_path + ae_generation_technique + "/retrain/"
    if not os.path.isdir(retrain_path):
        os.makedirs(retrain_path)

    print(f"Dataset: {dataset_name}, Model architecture: {model_architecture}, num_classes: {num_classes}, ae_generation_technique:{ae_generation_technique}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    with np.load(original_data_path) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=2023)
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if dataset_name == "fashion_mnist":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Load the generated adversarial inputs for testing. FGSM and PGD.
    with np.load(base_path + "fgsm/fgsm_valid_ae.npz") as f:
        fgsm_valid_idx, fgsm_valid, fgsm_valid_label = f['idx'], f['ae'], f['ae_label']

    with np.load(base_path + "pgd/pgd_valid_ae.npz") as f:
        pgd_valid_idxs, pgd_valid, pgd_valid_label = f['idx'], f['ae'], f['ae_label']

    with np.load(base_path + "fgsm/fgsm_test_ae.npz") as f:
        fgsm_test_idx, fgsm_test, fgsm_test_label = f['idx'], f['ae'], f['ae_label']

    with np.load(base_path + "pgd/pgd_test_ae.npz") as f:
        pgd_test_idxs, pgd_test, pgd_test_label = f['idx'], f['ae'], f['ae_label']

    ae_valid_dataset = np.concatenate((fgsm_valid, pgd_valid))
    ae_valid_label = np.concatenate((fgsm_valid_label, pgd_valid_label))
    ae_test_dataset = np.concatenate((fgsm_test, pgd_test))
    ae_test_label = np.concatenate((fgsm_test_label, pgd_test_label))

    durations = [1000, 2000, 3000, 4000]  # the fuzzing durations [1000, 2000, 3000, 4000]
    for duration in durations:
        for ae_type in ['train_ae', 'dlfuzz_seeds_ae', 'adapt_seeds_ae', 'robot_seeds_ae', 'aster_seeds_ae']:
            print("*" * 10 + f"duration: {duration}, ae type: {ae_type}" + "*" * 10)
            retrained_model_path = retrain_path + "/best_retrain_%d_%s.h5" % (duration, ae_type)

            # Load the generated adversarial inputs for training
            if ae_generation_technique == "dlfuzz" or ae_generation_technique == "adapt":
                with np.load(ae_data_path + "/" + ae_type + ".npz") as f:
                    ae_train_idx, ae_train, ae_train_label = f['idx'], f['ae'], f['ae_label']
                cur_idxs = np.where(ae_train_idx < (duration//adapt_duration))[0]
            else:
                with np.load(ae_data_path + "/" + ae_type + ".npz") as f:
                    ae_train_idx, ae_train, ae_train_label, ae_time, ae_fol = f['idx'], f['ae'], f['ae_label'], f['time'], f['fol']
                cur_idxs = np.where(ae_time <= duration)[0]
            print("*" * 10 + f"duration: {duration}, seed num: {max(cur_idxs)+1}, ae num: {len(cur_idxs)}, ae type: {ae_type}" + "*" * 10)

            x_train_mix = np.concatenate((x_train, ae_train[cur_idxs]), axis=0)
            y_train_mix = np.concatenate((y_train, ae_train_label[cur_idxs]), axis=0)
            checkpoint = ModelCheckpoint(filepath=retrained_model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
            lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            callbacks = [checkpoint, lr_reducer, lr_scheduler]

            # load old model
            if param_index == 3:
                model = keras.models.load_model(original_model_path, custom_objects={'compute_loss_100': compute_loss_100, 'compute_acc': compute_acc})
            else:
                model = keras.models.load_model(original_model_path, custom_objects={'compute_loss': compute_loss, 'compute_acc': compute_acc})

            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule_retrain(0)), metrics=['accuracy'])
            datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            datagen.fit(x_train_mix)
            history = model.fit(datagen.flow(x_train_mix, y_train_mix, batch_size=256), validation_data=(ae_valid_dataset, ae_valid_label), epochs=40, verbose=2,
                                callbacks=callbacks, steps_per_epoch=x_train_mix.shape[0] // 256, shuffle=True)

            model = keras.models.load_model(retrained_model_path)
            scores_fgsm_valid = model.evaluate(fgsm_valid, fgsm_valid_label, verbose=2)
            scores_pgd_valid = model.evaluate(pgd_valid, pgd_valid_label, verbose=2)
            scores_ae_valid = model.evaluate(ae_valid_dataset, ae_valid_label, verbose=2)
            print(f'Duration: {duration}, fgsm test: {scores_fgsm_valid[1]}, pgd test: {scores_pgd_valid[1]}, ae test: {scores_ae_valid[1]}')

            scores_fgsm_test = model.evaluate(fgsm_test, fgsm_test_label, verbose=2)
            scores_pgd_test = model.evaluate(pgd_test, pgd_test_label, verbose=2)
            scores_ae_test = model.evaluate(ae_test_dataset, ae_test_label, verbose=2)
            print(f'Duration: {duration}, fgsm test: {scores_fgsm_test[1]}, pgd test: {scores_pgd_test[1]}, ae test: {scores_ae_test[1]}')

            scores_std_train = model.evaluate(x_train, y_train, verbose=2)
            scores_std_valid = model.evaluate(x_valid, y_valid, verbose=2)
            scores_std_test = model.evaluate(x_test, y_test, verbose=2)
            print(f'Duration: {duration}, std train: {scores_std_train[1]}, std valid: {scores_std_valid[1]}, std test: {scores_std_test[1]}')
            print()
