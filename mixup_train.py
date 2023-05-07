import os
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from architectures import mobilenet, lenet, resnet, shufflenet
from mixup_utils import mixup_data, compute_loss, compute_loss_100, compute_acc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_mixup(epoch, lr):
    if epoch == 150:
        lr = lr * 1e-1
    elif epoch == 100:
        lr = lr * 1e-1
    print('Learning rate: ', lr)
    return lr


def lr_schedule_svhn_cifar10(epoch, lr):
    if epoch == 140:
        lr = lr * 1e-1
    elif epoch == 110:
        lr = lr * 1e-1
    elif epoch == 80:
        lr = lr * 1e-1
    elif epoch == 50:
        lr = lr * 1e-1
    print('Learning rate: ', lr)
    return lr


def lr_schedule_cifar100_tiny(epoch, lr):
    if epoch == 250:
        lr = lr * 1e-1
    elif epoch == 200:
        lr = lr * 1e-1
    elif epoch == 150:
        lr = lr * 1e-1
    elif epoch == 100:
        lr = lr * 1e-1
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "batch_size": [256, 256, 256, 256],
                  "epochs": [150, 150, 200, 200],
                  "init_lr": [0.001, 0.001, 0.001, 0.001]}

    for param_index in [1]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        batch_size = param_dict["batch_size"][param_index]
        epochs = param_dict["epochs"][param_index]
        init_lr = param_dict["init_lr"][param_index]

        # path
        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture + "/"
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        save_model_path = base_path + '/mixup_models/'
        if not os.path.isdir(save_model_path):
            os.makedirs(save_model_path)

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

        if dataset_name == "fashion_mnist" and model_architecture == "mobilenetv2":
            model = mobilenet.MobileNetV2(input_shape=x_train.shape[1:], num_classes=num_classes)
            model.compile(loss=compute_loss, optimizer=Adam(lr=lr_schedule_svhn_cifar10(0, init_lr)), metrics=[compute_acc])
            lr_scheduler = LearningRateScheduler(lr_schedule_svhn_cifar10)
        elif dataset_name == "svhn" and model_architecture == "shufflenetv2":
            model = shufflenet.ShuffleNetV2(num_classes=num_classes, input_shape=x_train.shape[1:])
            model.compile(loss=compute_loss, optimizer=Adam(lr=lr_schedule_svhn_cifar10(0, init_lr)), metrics=[compute_acc])
            lr_scheduler = LearningRateScheduler(lr_schedule_svhn_cifar10)
        # elif dataset_name == "svhn" and model_architecture == "lenet5":
        #     model = lenet.Lenet5(input_shape=x_train.shape[1:], num_classes=num_classes)
        #     model.compile(loss=compute_loss, optimizer=Adam(lr=lr_schedule_svhn_cifar10(0, init_lr)), metrics=[compute_acc])
        #     lr_scheduler = LearningRateScheduler(lr_schedule_svhn_cifar10)
        elif dataset_name == "cifar10" and model_architecture == "resnet20":
            model = resnet.resnet_v1(input_shape=x_train.shape[1:], depth=20, num_classes=num_classes)
            model.compile(loss=compute_loss, optimizer=Adam(lr=lr_schedule_mixup(0, init_lr)), metrics=[compute_acc])
            lr_scheduler = LearningRateScheduler(lr_schedule_mixup)
        elif dataset_name == "cifar100" and model_architecture == "resnet56":
            model = resnet.resnet_v1(input_shape=x_train.shape[1:], depth=56, num_classes=num_classes)
            model.compile(loss=compute_loss_100, optimizer=Adam(lr=lr_schedule_cifar100_tiny(0, init_lr)), metrics=[compute_acc])
            lr_scheduler = LearningRateScheduler(lr_schedule_cifar100_tiny)

        model_name = dataset_name + '_' + model_architecture + '_model.{epoch:03d}.h5'
        filepath = os.path.join(save_model_path, model_name)

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_compute_acc', verbose=1, save_best_only=True, mode='auto')
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        datagen.fit(x_train)

        mixed_indexes_df = pd.DataFrame()
        train_history_list = []
        for epoch in range(int(epochs)):
            datagen_result = datagen.flow(x_train, y_train, batch_size=len(x_train), shuffle=False)
            mixed_x_train, y_a, y_b, lam, mixed_indexes = mixup_data(datagen_result[0][0], datagen_result[0][1], 1.0)
            mixed_indexes_df[epoch] = mixed_indexes
            mixed_y_train = np.concatenate([y_a, y_b, lam], axis=-1)
            history = model.fit(mixed_x_train, mixed_y_train, validation_data=(x_valid, y_valid), epochs=epoch+1, verbose=2,
                                callbacks=callbacks, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=True, initial_epoch=epoch)

            test_acc = len(np.where(np.argmax(model(x_test), axis=1).reshape(1, -1) == y_test.reshape(1, -1))[0]) / len(x_test)
            print(f'Test accuracy: {test_acc}')
            train_history_list.append([history.history["compute_acc"][0], history.history["val_compute_acc"][0], test_acc, history.history["loss"][0], history.history["val_loss"][0]])

        mixed_indexes_df.to_csv(save_model_path + "/mixed_indexes.csv", index=False)
        pd.DataFrame(train_history_list, columns=['compute_acc', 'val_compute_acc', 'test_acc', 'train_loss', 'val_loss']).to_csv(save_model_path + "/train_history.csv", index=False)
