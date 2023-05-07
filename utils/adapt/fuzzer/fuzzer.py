import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils.adapt.network import Network
from utils.adapt.fuzzer.archive import Archive
from utils.adapt.metric import NeuronCoverage
from utils.adapt.strategy import RandomStrategy
from utils.adapt.utils.functional import coverage
from utils.adapt.utils.timer import Timeout
from utils.adapt.utils.timer import Timer
from tensorflow.keras.losses import categorical_crossentropy

from utils.context import Context


class WhiteBoxFuzzer:
    '''A white-box fuzzer for deep neural network.

    White-box testing is a technique that utilizes internal values to generate
    inputs. This class will uses the gradients to generate next input for testing.
    This fuzzer will test one input.
    '''

    def __init__(self, network, input, ground_truth, metric=None, strategy=None, k=10, delta=0.05, class_weight=0.5, neuron_weight=0.5, lr=0.01, trail=3, decode=None):
        '''Create a fuzzer.

        Create a white-box fuzzer. All parameters except for the time budget, should
        be set.

        Args:
          network: A wrapped Keras model with `adapt.Network`. Wrap if not wrapped.
          input: An input to test.
          metric: A coverage metric for testing. By default, the fuzzer will use
            a neuron coverage with threshold 0.5.
          strategy: A neuron selection strategy. By default, the fuzzer will use
            the `adapt.strategy.RandomStrategy`.
          k: A positive integer. The number of the neurons to select.
          delta: A positive floating point number. Limits of distance of created
            inputs. By default, use 0.05.
          class_weight: A floating point number. A weight for the class term in
            optimization equation. By default, use 0.5.
          neuron_weight: A floating point number. A weight for the neuron term in
            optimization equation. By default, use 0.5.
          lr: A floating point number. A learning rate to apply when generating
            the next input using gradients. By default, use 0.01.
          trail: A positive integer. Trails to apply one set of selected neurons.
          decode: A function that gets logits and return the label. By default,
            uses `np.argmax`.

        Raises:
          ValueError: When arguments are not in their proper range.
        '''

        # Store variables.
        if not isinstance(network, Network):
            network = Network(network)
        self.network = network

        self.input = np.array(input)
        self.ground_truth = np.array(ground_truth)

        if not metric:
            metric = NeuronCoverage(0.5)
        self.metric = metric

        if not strategy:
            strategy = RandomStrategy(self.network)
        self.strategy = strategy

        if k < 1:
            raise ValueError('The argument k is not positive.')
        self.k = int(k)

        if delta <= 0:
            raise ValueError('The argument delta is not positive.')
        self.delta = float(delta)

        self.class_weight = float(class_weight)
        self.neuron_weight = float(neuron_weight)
        self.lr = float(lr)

        if trail < 1:
            raise ValueError('The argument trails is not positive.')
        self.trail = trail

        if not decode:
            decode = np.argmax
        self.decode = decode

        # Variables that are set during (or after) testing.
        self.archive = None

        self.start_time = None
        self.time_consumed = None

        self.label = None
        self.orig_coverage = None
        self.covered = None
        self.coverage = None

    def start(self, hours=0, minutes=0, seconds=0, append='meta', verbose=0, seed_type="generated_seed", original_pred_label=None):
        '''Start fuzzing for the given time budget.
        Start fuzzing for a time budget.
        :param hours: A non-negative integer which indicates the time budget in hours. 0 for the default value.
        :param minutes: A non-negative integer which indicates the time budget in minutes. 0 for the defalut value.
        :param seconds: A non-negative integer which indicates the time budget in seconds. 0 for the defalut value.
        If hours, minutes, and seconds are set to be 0, the time budget will automatically set to be 10 seconds.
        :param append: An option that specifies the data that archive stores. Should be one of "meta", "min_dist", or "all". By default, "meta" will be used.
        :param verbose: An option that print out logs or not. Pass 1 for printing and 0 for not printing. Be default, set to be 0.
        :return:
        '''

        # Get the original properties.
        internals, logits = self.network.predict(np.array([self.input]))
        orig_index = np.argmax(logits)
        orig_norm = np.linalg.norm(self.input)
        self.label = self.decode(np.array([logits]))
        self.covered = self.metric(internals=internals, logits=logits)
        self.orig_coverage = coverage(self.covered)

        # Initialize variables.
        self.archive = Archive(self.input, self.label, append=append)

        # Initialize the strategy.
        self.strategy = self.strategy.init(covered=self.covered, label=self.label)

        # Set timer.
        timer = Timer(hours, minutes, seconds)
        if verbose > 0:
            print('Fuzzing started. Press ctrl+c to quit.')

        ae_list = []
        ae_target_list = []
        ae_pred_label = []

        # Loop until timeout, or interrupted by user.
        try:
            while True:
                # Create worklist.
                worklist = [tf.identity(np.array([self.input]))]

                # While worklist is not empty:
                while len(worklist) > 0:
                    # Get current input
                    new_input = worklist.pop(0)

                    # Select neurons.
                    neurons = self.strategy(k=self.k)

                    # Try trail times.
                    for _ in range(self.trail):
                        # Get original coverage
                        orig_cov = coverage(self.covered)

                        # Calculate gradients.
                        with tf.GradientTape() as t:
                            t.watch(new_input)
                            internals, logits = self.network.predict(new_input)
                            loss = self.neuron_weight * K.sum([internals[li][ni] for li, ni in neurons]) - self.class_weight * logits[orig_index]

                        dl_di = t.gradient(loss, new_input)

                        # Generate the next input using gradients.
                        new_input += self.lr * dl_di
                        new_input = tf.clip_by_value(new_input, clip_value_min=0.0, clip_value_max=1.0)

                        # Get the properties of the generated input.
                        internals, logits = self.network.predict(new_input)

                        covered = self.metric(internals=internals, logits=logits)
                        new_pred_label = self.decode(np.array([logits]))
                        distance = np.linalg.norm(new_input - self.input) / orig_norm

                        # Update variables in fuzzer
                        self.covered = np.bitwise_or(self.covered, covered)
                        new_cov = coverage(self.covered)

                        # If coverage increased.
                        if new_cov > orig_cov and distance < self.delta:
                            worklist.append(tf.identity(new_input))

                        # Feedback to strategy.
                        self.strategy.update(covered=covered, label=new_pred_label)

                        # Add created input.
                        self.archive.add(new_input, new_pred_label, distance, timer.elapsed.total_seconds(), new_cov)

                        # find an ae
                        if seed_type == "generated_seed":
                            # if (self.label != new_pred_label and new_pred_label != original_pred_label) or (new_pred_label != original_pred_label):
                            if self.label != new_pred_label or new_pred_label != original_pred_label:
                                ae_list.extend(new_input.numpy())
                                ae_target_list.append(self.ground_truth)
                                ae_pred_label.append(new_pred_label)
                        elif self.label != new_pred_label:
                            ae_list.extend(new_input.numpy())
                            ae_target_list.append(self.ground_truth)
                            ae_pred_label.append(new_pred_label)

                        # Check timeout.
                        timer.check_timeout()

                # Update strategy.
                self.strategy.next()

        except Timeout:
            pass
        except KeyboardInterrupt:
            if verbose > 0:
                print('Stopped by the user.')
            sys.exit()

        # Update meta variables.
        self.coverage = coverage(self.covered)
        self.start_time = timer.start_time
        self.time_consumed = timer.elapsed.total_seconds()
        self.archive.timestamp.append((self.time_consumed, self.coverage))

        if verbose > 0:
            print('Done!')

        return np.array(ae_list), np.array(ae_target_list), self.label, np.array(ae_pred_label)
