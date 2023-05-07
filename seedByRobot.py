import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import copy
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from mixup_utils import compute_loss, compute_loss_100, compute_acc

np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    # all parameters
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.05, 0.05, 0.05, 0.05],
                  "lr": [0.01, 0.01, 0.01, 0.01]}

    for param_index in [1]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        lr = param_dict["lr"][param_index]
        ep = param_dict["ep"][param_index]
        ae_generation_technique = "robot"

        # path
        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture + "/"
        original_model_path = base_path + "/mixup_models/" + data_architecture + ".h5"
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        generated_seed_path = base_path + "/generated_seed/"
        if not os.path.isdir(generated_seed_path):
            os.makedirs(generated_seed_path)

        if not os.path.exists(original_data_path):
            os.makedirs(original_data_path)

        print(f"Seed generation by {ae_generation_technique}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}, lr: {lr}")

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

        num = 4000
        batch_size = 1000
        random.seed(2023)
        all_indexes = np.array(random.sample(list(range(len(x_train))), num))
        data_for_ae = x_train[all_indexes]
        label_for_ae = y_train[all_indexes]
        generated_seeds, ground_truth, original_pred_label, ae_pred_label_list, errors = [], [], [], [], []

        for i in range(int(np.ceil(len(data_for_ae)/batch_size))):
            seeds = np.array(range(data_for_ae.shape[0]))[i*batch_size: (i+1)*batch_size]
            images = data_for_ae[seeds]
            labels = label_for_ae[seeds]

            # some training samples is static, i.e., grad=<0>, hard to generate.
            gen_img = tf.Variable(images)
            original_prediction_labels_one_hot = keras.utils.to_categorical(np.argmax(model(images), axis=1), num_classes)
            with tf.GradientTape() as g:
                loss = keras.losses.categorical_crossentropy(original_prediction_labels_one_hot, model(gen_img))
                grads = g.gradient(loss, gen_img)

            fols = np.linalg.norm((grads.numpy()+1e-20).reshape(images.shape[0], -1), ord=2, axis=1)
            seeds_filter = np.where(fols > 1e-18)[0]

            lam = 1
            top_k = 5
            steps = 3
            for idx in seeds_filter:
                total_sets = []
                img_list = []
                tmp_img = images[[idx]]
                orig_img = copy.deepcopy(tmp_img)
                orig_norm = np.linalg.norm(orig_img)
                img_list.append(tf.identity(tmp_img))
                logits = model(tmp_img)
                orig_index = np.argmax(logits[0])
                target = keras.utils.to_categorical([orig_index], num_classes)
                label_top5 = np.argsort(logits[0])[-top_k:-1][::-1]
                folMAX = 0.0

                while len(img_list) > 0:
                    gen_img = img_list.pop(0)

                    for _ in range(steps):
                        gen_img = tf.Variable(gen_img, dtype=float)
                        with tf.GradientTape(persistent=True) as g:
                            loss = keras.losses.categorical_crossentropy(target, model(gen_img))
                            grads = g.gradient(loss, gen_img)
                            fol = tf.norm(grads+1e-20)
                            g.watch(fol)
                            logits = model(gen_img)
                            # obj = lam*fol - logits[0][orig_index]
                            obj = logits[0][label_top5[0]] + logits[0][label_top5[1]] + logits[0][label_top5[2]] + logits[0][label_top5[3]] - logits[0][orig_index] + lam*fol
                            dl_di = g.gradient(obj, gen_img)
                        del g

                        gen_img = gen_img + dl_di * lr * (random.random() + 0.5)
                        gen_img = tf.clip_by_value(gen_img, clip_value_min=0.0, clip_value_max=1.0)

                        with tf.GradientTape() as t:
                            t.watch(gen_img)
                            loss = keras.losses.categorical_crossentropy(target, model(gen_img))
                            grad = t.gradient(loss, gen_img)
                            fol = np.linalg.norm(grad.numpy())  # L2 adaption

                        distance = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm
                        if fol > folMAX and distance < ep:
                            folMAX = fol
                            img_list.append(tf.identity(gen_img))

                        if distance < ep:
                            preds = model(gen_img).numpy()
                            gen_index = np.argmax(preds[0])
                            if gen_index != orig_index:
                                total_sets.append((gen_img.numpy(), labels[idx], orig_index, gen_index))

                if len(total_sets) > 0:
                    random.seed(2023)
                    random_index = random.choice(list(range(len(total_sets))))

                    generated_seeds.append(total_sets[random_index][0][0])
                    ground_truth.append(total_sets[random_index][1])
                    original_pred_label.append(total_sets[random_index][2])
                    ae_pred_label_list.append(total_sets[random_index][3])
                else:
                    generated_seeds.append(images[idx])
                    ground_truth.append(labels[idx])
                    original_pred_label.append(orig_index)
                    ae_pred_label_list.append(orig_index)
            np.savez(generated_seed_path + "/robot_seeds.npz", generated_seeds=np.array(generated_seeds), ground_truth=np.array(ground_truth),
                     original_pred_label=np.array(original_pred_label), pred_label=np.array(ae_pred_label_list), errors=np.array(errors))
