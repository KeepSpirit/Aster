import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils.adapt import Network
from utils.adapt.metric import NC, TKNC
from utils.adapt.fuzzer import WhiteBoxFuzzer
from utils.adapt.strategy import AdaptiveParameterizedStrategy, UncoveredRandomStrategy, MostCoveredStrategy
from mixup_utils import compute_loss, compute_loss_100, compute_acc

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.05, 0.05, 0.05, 0.05],
                  "lr": [0.01, 0.01, 0.01, 0.01],
                  "duration": [1, 1, 1, 1]}

    for param_index in [1]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        ep = param_dict["ep"][param_index]
        lr = param_dict["lr"][param_index]
        duration = param_dict["duration"][param_index]
        ae_generation_technique = "dlfuzz"

        # path
        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture + "/"
        original_model_path = base_path + "/mixup_models/" + data_architecture + ".h5"
        generated_seed_path = base_path + "/generated_seed/"

        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        if not os.path.exists(original_data_path):
            os.makedirs(original_data_path)
        print(f"Seed generation by {ae_generation_technique}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}")

        # load original dataset
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

        if param_index == 3:
            network = Network(keras.models.load_model(original_model_path, custom_objects={'compute_loss_100': compute_loss_100, 'compute_acc': compute_acc}))
        else:
            network = Network(keras.models.load_model(original_model_path, custom_objects={'compute_loss': compute_loss, 'compute_acc': compute_acc}))

        # generate adversarial examples at once.
        num = 4000
        random.seed(2023)
        all_indexes = np.array(random.sample(list(range(len(x_train))), num))
        data_for_ae = x_train[all_indexes]
        label_for_ae = y_train[all_indexes]

        generated_seeds, ground_truth, original_pred_label, ae_pred_label_list, errors = [], [], [], [], []

        metric = NC(0.5)
        # metric = TKNC()

        for i in range(len(data_for_ae)):
            if ae_generation_technique == "adapt":
                strategy = AdaptiveParameterizedStrategy(network)
            elif ae_generation_technique == "deepxplore":
                strategy = UncoveredRandomStrategy(network)
            elif ae_generation_technique == "dlfuzz":
                strategy = MostCoveredStrategy(network)
            else:
                strategy = None

            fuzzer = WhiteBoxFuzzer(network=network, input=data_for_ae[i], ground_truth=label_for_ae[i], metric=metric, strategy=strategy, k=10, delta=ep, class_weight=0.5, neuron_weight=0.5, lr=lr, trail=3, decode=None)
            ae, ae_label, original_seed_pred_label, ae_pred_label = fuzzer.start(seconds=duration, append='min_dist', seed_type="train", original_pred_label=None)

            if len(ae) > 0:
                random.seed(2023)
                random_index = random.choice(list(range(len(ae))))

                generated_seeds.append(ae[random_index])
                ground_truth.append(ae_label[random_index])
                original_pred_label.append(original_seed_pred_label)
                ae_pred_label_list.append(ae_pred_label[random_index])
            else:
                generated_seeds.append(data_for_ae[i])
                ground_truth.append(label_for_ae[i])
                original_pred_label.append(original_seed_pred_label)
                ae_pred_label_list.append(original_seed_pred_label)

        np.savez(generated_seed_path + "/dlfuzz_seeds.npz", generated_seeds=np.array(generated_seeds), ground_truth=np.array(ground_truth),
                 original_pred_label=np.array(original_pred_label), pred_label=np.array(ae_pred_label_list), errors=np.array(errors))
