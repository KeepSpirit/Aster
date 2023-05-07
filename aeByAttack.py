import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils.attack import FGSM, PGD
from mixup_utils import compute_loss, compute_loss_100, compute_acc

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.03, 0.03, 0.01, 0.01]}

    for param_index in [1]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        ep = param_dict["ep"][param_index]

        # path
        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture + "/"
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        original_model_path = base_path + "/mixup_models/" + data_architecture + ".h5"
        generated_seed_path = base_path + "/generated_seed/"
        print(f"Generate by PGD and FGSM. Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}")

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
            model = keras.models.load_model(original_model_path, custom_objects={'compute_loss_100': compute_loss_100, 'compute_acc': compute_acc})
        else:
            model = keras.models.load_model(original_model_path, custom_objects={'compute_loss': compute_loss, 'compute_acc': compute_acc})

        # generate adversarial examples at once.
        data_type_list = ["train", "valid"]        # ["train", "valid", "test", "generated_seed"]
        attack_method_list = ["fgsm", "pgd"]                # ["fgsm", "pgd"]
        for data_type in data_type_list:
            for attack_method in attack_method_list:
                ae_data_path = base_path + attack_method + "/"
                if not os.path.exists(ae_data_path):
                    os.makedirs(ae_data_path)

                if data_type == "train":
                    data_for_ae = x_train
                    label_for_ae = y_train
                elif data_type == "valid":
                    data_for_ae = x_valid
                    label_for_ae = y_valid
                elif data_type == "test":
                    data_for_ae = x_test
                    label_for_ae = y_test
                elif data_type == "generated_seed":
                    with np.load(generated_seed_path + "/seeds.npz") as f:
                        generated_seeds, seed_ground_truth, seed_original_pred_label, seed_errors = f['generated_seeds'], f['ground_truth'], f['original_pred_label'], f['errors']

                    ae_data_path = base_path + attack_method + "/generated_seed_ae/"
                    if not os.path.exists(ae_data_path):
                        os.makedirs(ae_data_path)
                    data_for_ae = generated_seeds
                    label_for_ae = seed_ground_truth

                if attack_method == "fgsm":
                    attack_tech = FGSM(model, ep=ep, isRand=True)
                elif attack_method == "pgd":
                    attack_tech = PGD(model, ep=ep, epochs=10, isRand=True)

                print(f"dataset: {dataset_name}, data_type:{data_type}, attack_method:{attack_method}")
                idx_all = []
                ae_all = []
                ae_label_all = []
                time_all = []

                batch_size = 1000
                for i in range(int(np.ceil(len(data_for_ae)/batch_size))):
                    idx, ae, ae_label, gen_time = attack_tech.generate(data_for_ae[i*batch_size: (i+1)*batch_size], label_for_ae[i*batch_size: (i+1)*batch_size])
                    idx_all.extend(idx+i*batch_size)
                    ae_all.extend(ae)
                    ae_label_all.extend(ae_label)
                    time_all.extend(gen_time)

                np.savez(ae_data_path + attack_method + "_" + data_type + "_ae.npz", idx=np.array(idx_all), ae=np.array(ae_all), ae_label=np.array(ae_label_all), time=np.array(time_all))
