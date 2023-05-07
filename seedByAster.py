import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from mixup_utils import compute_loss, compute_loss_100, compute_acc

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class GeneticSeedGen:
    def __init__(self, model, ep=0.01, lr=0.01):
        self.model = model
        self.ep = ep
        self.lr = lr
        self.time_start = time.time()

    def crossover(self, p1, p2):
        # crossover two parents to create two children
        pt = np.random.randint(1, max(2, len(p1)-2))   # select crossover point that is not on the end of the string
        c1 = np.concatenate((p1[:pt], p2[pt:]))
        c2 = np.concatenate((p2[:pt], p1[pt:]))
        return [c1, c2]

    def cos_sim(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b.T)/(np.linalg.norm(a)*np.linalg.norm(b))

    def generation(self, x, y, mixed_indexes, save_path=None):
        xi = x.copy()
        # 1. first stage: generate ae to initialize the ae for the seed generation
        seed_gen_pre_path = save_path + "/seed_gen_pre.npz"
        if not os.path.exists(seed_gen_pre_path):
            original_pred_vector = self.model(x)
            original_pred_label = np.argmax(original_pred_vector, axis=1)
            target = tf.constant(keras.utils.to_categorical(original_pred_label, len(y[0])), dtype=float)

            x_ae = tf.Variable(x, dtype=float)
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(target, self.model(x_ae))
                grads = tape.gradient(loss, x_ae)
            gradient_matrix = tf.sign(grads)
            x_ae = x + gradient_matrix * self.ep
            x_ae = tf.clip_by_value(x_ae, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
            x_ae = tf.clip_by_value(x_ae, clip_value_min=0.0, clip_value_max=1.0)

            seed_array = x_ae.numpy()
            seed_pred_vector = self.model(seed_array).numpy()
            seed_pred_label = np.argmax(seed_pred_vector, axis=1)
            direction_array = seed_array - xi
            np.savez(seed_gen_pre_path, original_pred_vector=original_pred_vector, original_pred_label=original_pred_label, seed_array=seed_array, seed_pred_vector=seed_pred_vector, seed_pred_label=seed_pred_label, direction_array=direction_array)
        else:
            with np.load(seed_gen_pre_path) as f1:
                original_pred_vector, original_pred_label, seed_array, seed_pred_vector, seed_pred_label, direction_array = f1['original_pred_vector'], f1['original_pred_label'], f1['seed_array'], f1['seed_pred_vector'], f1['seed_pred_label'], f1['direction_array']
            print(f"finish loading!")

        # 2. chromosome - [population, evaluation, fitness value, reproduction, crossover, mutation]
        # initialize population
        chromosome_init_path = save_path + "chromosome_init.npz"
        if not os.path.exists(chromosome_init_path):
            chromosome_list = []           # the chromosome for each seed
            fitness_list_all = []          # the fitness array (before calculating mean value) for each chromosome
            fitness_list_mean_all = []     # the mean fitness array for each chromosome between a1 and a2
            error_list = []                # the cosine similarity between prediction vector with the new generated seed
            for i in range(len(xi)):
                chromosome = list(mixed_indexes.iloc[i])
                chromosome_list.append(chromosome)
                fitness_one = np.array([self.cos_sim(seed_pred_vector[i], seed_pred_vector[j]) for j in chromosome])
                fitness_list_all.append(fitness_one)
                fitness_list_mean_all.append(np.mean(fitness_one))
                error_list.append(self.cos_sim(original_pred_vector[i], seed_pred_vector[i]))
            np.savez(chromosome_init_path, chromosome_list=chromosome_list, fitness_list_all=fitness_list_all, fitness_list_mean_all=fitness_list_mean_all, error_list=error_list)
        else:
            with np.load(chromosome_init_path) as f2:
                chromosome_list, fitness_list_all, fitness_list_mean_all, error_list = list(f2['chromosome_list']), list(f2['fitness_list_all']), list(f2['fitness_list_mean_all']), list(f2['error_list'])

        best_fitness_mean = 0                                                   # the best mean fitness value for all mean fitness value
        best_epoch = 0
        start_time = time.time()
        for epoch in range(10):
            if np.mean(fitness_list_mean_all) - best_fitness_mean > 1e-5:       # check overall fitness
                best_fitness_mean = np.mean(fitness_list_mean_all)
                best_epoch = epoch
            elif epoch - best_epoch >= 3:                                       # early stopping by patience
                break
            print(f"epoch: {epoch}, best_fitness_mean: {best_fitness_mean}, time: {time.time() - start_time}")

            for i in range(len(xi)):
                best_idx = 0
                useful_chromosome = []
                sorted_chromosome = np.array(chromosome_list[i])[np.argsort(fitness_list_all[i])[::-1]]
                for cur_idx, sorted_chromosome_i in enumerate(sorted_chromosome):
                    new_ae = seed_array[i] + (direction_array[i] + direction_array[sorted_chromosome_i])/2*self.lr
                    new_ae = tf.clip_by_value(new_ae, clip_value_min=xi[i] - self.ep, clip_value_max=xi[i] + self.ep)
                    new_ae = tf.clip_by_value(new_ae, clip_value_min=0.0, clip_value_max=1.0)
                    new_ae_pred_vector = self.model(np.expand_dims(new_ae, axis=0)).numpy()[0]

                    new_similarity = self.cos_sim(original_pred_vector[i], new_ae_pred_vector)
                    if error_list[i] > new_similarity:                            # smaller similarity means larger error
                        best_idx = cur_idx
                        error_list[i] = new_similarity
                        seed_array[i] = new_ae
                        seed_pred_vector[i] = new_ae_pred_vector
                        seed_pred_label[i] = np.argmax(new_ae_pred_vector)
                        direction_array[i] = new_ae - xi[i]
                        useful_chromosome.append(sorted_chromosome_i)
                    elif cur_idx - best_idx >= 3:                                 # early stopping by patience
                        break

                chromosome_list[i] = useful_chromosome
                new_fitness_one = np.array([self.cos_sim(seed_pred_vector[i], seed_pred_vector[j]) for j in chromosome_list[i]])
                fitness_list_all[i] = new_fitness_one.copy()
                fitness_list_mean_all[i] = np.mean(new_fitness_one) if len(new_fitness_one) > 0 else 0
                # print(f"i: {i}, useful_chromosome: {len(useful_chromosome)} time: {time.time() - start_time}")
            print(f"epoch: {epoch}, best_fitness_mean before crossover: {np.mean(fitness_list_mean_all)}, time: {time.time() - start_time}")

            # crossover with the best fitness
            for i in range(len(xi)):
                if len(fitness_list_all[i]) > 0:
                    c1, c2 = np.zeros(len(seed_array)), np.zeros(len(seed_array))
                    c1[chromosome_list[i]] = 1
                    c2[chromosome_list[np.argmax(fitness_list_all[i])]] = 1
                    c1, _ = self.crossover(c1, c2)
                    chromosome = np.where(c1 == 1)[0]
                    new_fitness_one = np.array([self.cos_sim(seed_pred_vector[i], seed_pred_vector[j]) for j in chromosome])
                    new_mean_fitness = np.mean(new_fitness_one) if len(new_fitness_one) > 0 else 0
                    if new_mean_fitness > fitness_list_mean_all[i]:
                        chromosome_list[i] = chromosome
                        fitness_list_all[i] = new_fitness_one.copy()
                        fitness_list_mean_all[i] = new_mean_fitness
        return seed_array, y, original_pred_label, seed_pred_label, error_list


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.05, 0.05, 0.05, 0.05],
                  "lr": [0.01, 0.01, 0.01, 0.01],
                  "num_gene": [65, 89, 175, 115]}

    # parameters
    for param_index in [2]:
        dataset_name = param_dict["dataset_name"][param_index]
        model_architecture = param_dict["model_architecture"][param_index]
        num_classes = param_dict["num_classes"][param_index]
        ep = param_dict["ep"][param_index]
        lr = param_dict["lr"][param_index]
        num_gene = param_dict["num_gene"][param_index]

        # path
        data_architecture = dataset_name + "_" + model_architecture
        base_path = "./checkpoint/" + data_architecture + "/"
        original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
        mixup_model_path = base_path + "/mixup_models/" + data_architecture + ".h5"
        mixup_indexes_path = base_path + "/mixup_models/mixed_indexes.csv"
        original_model_path = base_path + "/mixup_models/" + data_architecture + ".h5"

        seed_path = base_path + "/generated_seed/"
        if not os.path.isdir(seed_path):
            os.makedirs(seed_path)

        print(f"Model architecture: {model_architecture}, dataset: {dataset_name}")
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
        mixup_indexes = pd.read_csv(mixup_indexes_path)
        mixup_indexes = mixup_indexes.iloc[:, 0:num_gene]

        if param_index == 3:
            original_model = keras.models.load_model(original_model_path, custom_objects={'compute_loss_100': compute_loss_100, 'compute_acc': compute_acc})
        else:
            original_model = keras.models.load_model(original_model_path, custom_objects={'compute_loss': compute_loss, 'compute_acc': compute_acc})

        genetic_seed_gen = GeneticSeedGen(model=original_model, ep=ep, lr=lr)
        generated_seeds, ground_truth, pred_label_0, pred_label_1, errors = genetic_seed_gen.generation(x=x_train, y=y_train, mixed_indexes=mixup_indexes, save_path=seed_path)
        np.savez(seed_path + "/aster_seeds_full.npz", generated_seeds=generated_seeds, ground_truth=ground_truth, original_pred_label=pred_label_0, pred_label=pred_label_1, errors=errors)
