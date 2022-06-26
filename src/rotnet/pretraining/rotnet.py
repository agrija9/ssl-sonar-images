import tensorflow as tf
import yaml
import os
from collections import deque
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pdb
import sys
sys.path.append('../../../')
sys.path.append('../../')

from architectures.alexnet import AlexNet
from architectures.resnet import ResNet
from architectures.densenet import DenseNet121
from architectures.mobilenet import mobilenet
from architectures.resnet20 import resnet20_model
from architectures.squeezenet import squeezenet
from architectures.minixception import MiniXception
from dataloaders import *
from losses import categorical_crossentropy
from utils_predict import (plot_batch_predicted_rotation,
                           plot_batch_predicted_class,
                           plot_confusion_matrix,
                           plot_confusion_matrix_rotations,
                           compute_precision_recall_fscore)


class RotNet(object):
    def __init__(self, args):
        # Define dataset, baseline classification model, logging directories
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.baseline_model = args.baseline_model
        self.train_mode = args.train_mode
        self.log_dir = "../../results/" + self.train_mode + "/losses/"
        self.checkpoint_dir = "../../results/" + self.train_mode + "/checkpoints/"
        self.images_dir = "../../results/" + self.train_mode + "/images/"

        # Define hyperparameters and dataset dims
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.channels = args.channels
        self.num_classes = args.num_classes

        # Load baseline classifcation model
        self._load_baseline_model()

        # Load Tensorflow dataset
        self._load_tensorflow_dataset()

        # Define optimizer and loss writer
        self.summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, self.dataset, self.baseline_model, "batch_size_" + str(self.batch_size)))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, epsilon=1) # lr 0.1, 1, 0.0001
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        # Define tf checkpoint object and summary writer (losses)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, self.dataset, self.baseline_model, "batch_size_" + str(self.batch_size))
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_prefix, max_to_keep=3)

    def _load_baseline_model(self):
        input_shape = [self.image_height, self.image_width, self.channels]

        baseline_models = {"alexnet": AlexNet, "resnet": ResNet,
                           "densenet": DenseNet121, "mobilenet": mobilenet,
                           "resnet20": resnet20_model, "squeezenet": squeezenet,
                           "minixception": MiniXception}

        if self.baseline_model.lower() in baseline_models:
            self.model = baseline_models[self.baseline_model.lower()](input_shape, self.num_classes)
            print("[INFO] Loading baseline model: {}".format(self.baseline_model.lower()))
            print("[INFO] Model summary:", self.model.summary())

        else:
            print("[ERROR] Incorrect baseline model name")
            exit()

    def _load_tensorflow_dataset(self):
        if self.dataset in ["CIFAR10", "cifar10", "cifar-10", "CIFAR-10"]:
            self.train_ds, self.val_ds, self.test_ds = load_CIFAR10(self.data_dir, self.batch_size)

        elif self.dataset in ["SONAR", "sonar_1", "sonar1"] and self.train_mode == "self_supervised_learning":
            self.train_ds, self.val_ds, self.test_ds = load_sonar_debris_self_supervised(self.data_dir, self.batch_size)

        elif self.dataset in ["SONAR", "sonar_1", "sonar1"] and self.train_mode == "supervised_learning":
            self.train_ds, self.val_ds, self.test_ds = load_sonar_debris_supervised(self.data_dir, self.batch_size)

        elif self.dataset in ["SONAR2", "sonar_2", "sonar2"] and self.train_mode == "supervised_learning":
            self.train_ds, self.val_ds, self.test_ds = load_sonar_turnedtable_supervised(self.data_dir, self.batch_size)

        elif self.dataset in ["SONAR3", "sonar_3", "sonar3"] and self.train_mode == "self_supervised_learning":
            self.train_ds, self.val_ds, self.test_ds = load_wild_sonar_data(self.data_dir, self.batch_size)

        else:
            print("[ERROR] Check dataset name or train mode")
            exit()

    @tf.function
    def _train_on_batch(self, epoch, X, y):
        """
        Mini batch training in tensorflow 2 using gradient tapes.
        """
        with tf.GradientTape() as tape:
            y_hat = self.model([X], training=True)
            train_loss = categorical_crossentropy()(y, y_hat) # Note: shape 128, compute batch mean to store it in writter

        gradients = tape.gradient(train_loss, self.model.trainable_variables) # model.trainable_weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Compute accuracies
        predicted_labels = tf.math.argmax(tf.nn.softmax(y_hat), axis=1)
        true_labels = tf.argmax(y, axis=1)
        equality = tf.math.equal(predicted_labels, true_labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32)) # tf float

        return np.mean(train_loss), accuracy.numpy()

    @tf.function
    def _validate_on_batch(self, X, y):
        """
        Computes validation loss and accuracy.
        Performs a forward pass to compute validation loss and accuracy.
        """
        # Compute logits
        y_hat = self.model([X], training=False) # logits

        # Compute validation loss
        validation_loss = categorical_crossentropy()(y, y_hat) # tf list

        # Compute accuracies
        predicted_labels = tf.math.argmax(tf.nn.softmax(y_hat), axis=1)
        true_labels = tf.argmax(y, axis=1)
        equality = tf.math.equal(predicted_labels, true_labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32)) # tf float

        return np.mean(validation_loss), accuracy.numpy()

    def train(self):
        """
        Training and validation pipeline.
        Store model if loss improves and every n-th iterations.
        """
        # try to restore checkpoint/training
        print("[INFO] Saving checkpoints to:", self.checkpoint_prefix)
        self.checkpoint.restore(self.manager.latest_checkpoint)

        if self.manager.latest_checkpoint:
            print("[INFO] Restored checkpoint from {}".format(self.manager.latest_checkpoint))
        else:
            print("[INFO] Initializing checkpoint from scratch.")
            print()

        # Epoch training
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_acc = []

        # Setup early stopping
        best_val_loss = 99999
        patience = 30 # patience epochs
        min_delta = 1e-4
        loss_history = deque(maxlen=patience + 1)

        # print()
        # print(tf.data.experimental.cardinality(self.val_ds).numpy())
        # pdb.set_trace()

        for epoch in tqdm(range(self.epochs)):
            start = time.time()

            # Train on batch
            batches_train_loss = []
            batches_train_acc = []

            steps_per_epoch = 0

            for batch, (X, y) in enumerate(self.train_ds):
                batch_train_loss, batch_train_accuracy = self._train_on_batch(epoch, X, y)
                batches_train_loss.append(batch_train_loss)
                batches_train_acc.append(batch_train_accuracy)
                self.checkpoint.step.assign_add(1) # step is batch step (1173, not epoch)

                steps_per_epoch += 1
                bound_image_gen = 1838 if self.dataset == "sonar1" else 1505
                if steps_per_epoch >= bound_image_gen / self.batch_size: # size x_train marine debris
                    # break the loop by hand, ImageGenerator loops indefinitely
                    break

            # Validate on batch
            batches_val_loss = []
            batches_val_acc = []
            for (X, y) in self.val_ds:
                batch_val_loss, batch_val_accuracy = self._validate_on_batch(X, y)
                batches_val_loss.append(batch_val_loss)
                batches_val_acc.append(batch_val_accuracy)

            # Store training, validation and accuracy values (txt files and tensorboard)
            epoch_train_loss.append(np.array(batches_train_loss).mean())
            epoch_train_acc.append(np.array(batches_train_acc).mean())
            epoch_val_loss.append(np.array(batches_val_loss).mean())
            epoch_val_acc.append(np.array(batches_val_acc).mean())

            with open(os.path.join(self.log_dir, self.dataset, self.baseline_model, "batch_size_" + str(self.batch_size) + "/train_results_statistics.txt"), "a") as f:
                f.write("Epoch no. {}".format(epoch) + "\n")
                f.write("Train Loss {}".format(epoch_train_loss[epoch]) + "\n")
                f.write("Valid Loss {}".format(epoch_val_loss[epoch]) + "\n")
                f.write("Train Acc {}".format(epoch_train_acc[epoch]) + "\n")
                f.write("Valid Acc {}".format(epoch_val_acc[epoch]) + "\n")
                f.write("-----------------------------------------------"+ "\n")

            with self.summary_writer.as_default():
                tf.summary.scalar("training_loss", epoch_train_loss[epoch], step=epoch) # tf.reduce_mean(loss_value)
                tf.summary.scalar("validation_loss", epoch_val_loss[epoch], step=epoch)
                tf.summary.scalar("training_accuracy", epoch_train_acc[epoch], step=epoch)
                tf.summary.scalar("validation_accuracy", epoch_val_acc[epoch], step=epoch)

            print()
            print("Epoch [%d/%d]:" % (epoch + 1, self.epochs))
            print("Training Loss: {},  Val Loss: {} , Train Acc {}, Val Acc: {}%".format(epoch_train_loss[epoch], epoch_val_loss[epoch], epoch_train_acc[epoch]*100, epoch_val_acc[epoch]*100))
            print("Time taken for epoch {} is {} sec".format(epoch + 1, time.time()-start))

            # Early stopping implementation (validation accuracy is monitored metric)
            loss_history.append(epoch_val_loss[epoch])
            if len(loss_history) > patience:
                if loss_history.popleft() < min(loss_history) and epoch_val_acc[epoch] > 0.98: # or epoch_train_acc[epoch] > 0.9930:
                    print(f'\nEarly stopping. No improvement in '
                          f'validation loss in the last {patience} epochs.')
                    break

            # if np.array(val_loss).mean() < best_loss:
            #    best_loss = np.array(val_loss).mean()

            # Save checkpoint (model) every nth-epochs
            # if int(checkpoint.step) % 10 == 0:
            if (epoch + 1) % 10 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}, epoch {}: {}".format(int(self.checkpoint.step), epoch+1, save_path))
                print()

        self.manager.save()

        # Save plots of losses/accuracies
        fig = plt.figure()
        plt.title(self.baseline_model + " losses", fontsize=16)
        plt.plot(epoch_train_loss, label="train")
        plt.plot(epoch_val_loss, label="validation")
        plt.legend(loc='best')
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        fig.savefig(os.path.join(self.log_dir, self.dataset, self.baseline_model, "batch_size_" + str(self.batch_size) + "/" + self.baseline_model + "_losses.png"))

        fig = plt.figure()
        plt.title(self.baseline_model + " accuracies", fontsize=16)
        plt.plot(epoch_train_acc, label="train")
        plt.plot(epoch_val_acc, label="validation")
        plt.legend(loc='best')
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        fig.savefig(os.path.join(self.log_dir, self.dataset, self.baseline_model, "batch_size_" + str(self.batch_size) + "/" + self.baseline_model + "_accuracies.png"))

    @tf.function
    def _predict_on_batch(self, X, y):
        """
        Computes validation loss and accuracy.
        Performs a forward pass to compute validation loss and accuracy.
        """
        # Compute logits
        y_hat = self.model([X], training=False) # logits

        # Compute predictions and accuracies
        predicted_labels = tf.math.argmax(tf.nn.softmax(y_hat), axis=1)
        true_labels = tf.argmax(y, axis=1)

        equality = tf.math.equal(predicted_labels, true_labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32)) # tf float

        return X, true_labels, predicted_labels, accuracy.numpy()

    def predict(self):
        """
        Restores model variables from checkpoint and runs inference.
        Inference is one forward pass to get rotation predictions.
        """

        # Define how many batch predictions and path to store image predictions
        no_predictions = 10
        images_prefix = os.path.join(self.images_dir, self.dataset, self.baseline_model, "batch_size_" + str(self.batch_size))

        # Restore model checkpoint
        print(self.checkpoint_prefix)
        print(self.manager)
        print(self.manager.latest_checkpoint)
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        status.assert_existing_objects_matched()

        if self.manager.latest_checkpoint:
            print("Checkpoint restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("No checkpoint file found, model has to be trained")
            exit()

        # Create folder (if doesnt exist) to store predicted rotations
        Path(images_prefix).mkdir(parents=True, exist_ok=True)

        # Do one forward pass to run predictions on full test set
        print("[INFO] Prediction mode: performing forward pass")

        # for i in range(no_predictions): only once for now since we load full test set into memory
        X_batch, y_batch = next(iter(self.test_ds))
        # X_batch, y_batch = iter(self.test_ds.take(100))
        # print(len(X_batch))
        # print()

        images, true_labels, predicted_labels, accuracy = self._predict_on_batch(X_batch, y_batch)
        print("Testing Accuracy:", accuracy.mean())

        # Store test acc in simple txt file
        with open(images_prefix + "/test_results_statistics.txt", "a") as f:
            (precision, recall, f1, _) = compute_precision_recall_fscore(true_labels.numpy(), predicted_labels.numpy())
            # f.write("Batch Index {}: Testing accuracy {}".format(i, accuracy.mean()) + "\n")
            f.write("Test Accuracy {}".format(accuracy.mean()) + "\n")
            f.write("Precision {}".format(precision) + "\n")
            f.write("Recall {}".format(recall) + "\n")
            f.write("F1 Score {}".format(f1))

        # Store test acc in tensorboard (skipping it if working w full test set)
        # with self.summary_writer.as_default():
        #     # tf.summary.scalar("testing_accuracy", accuracy.mean(), step=i)
        #     tf.summary.scalar("testing_accuracy", accuracy.mean())

        # Store image predictions
        # rotation classes
        if self.train_mode == "self_supervised_learning":
            fig = plot_batch_predicted_rotation(images.numpy(), true_labels.numpy(), predicted_labels.numpy(), images_prefix)
            fig.savefig(os.path.join(images_prefix, "predicted_angle_test_set.png"), dpi=fig.dpi)

            confusion_matrix = tf.math.confusion_matrix(true_labels.numpy(), predicted_labels.numpy()).numpy()
            print("Unnormalized confusion matrix")
            print(confusion_matrix)
            cm_fig = plot_confusion_matrix_rotations(confusion_matrix)
            cm_fig.savefig(os.path.join(images_prefix, "confusion_matrix_rotations.png"), dpi=fig.dpi)

        # normal classes
        else:
            fig = plot_batch_predicted_class(images.numpy(), true_labels.numpy(), predicted_labels.numpy(), images_prefix, self.dataset)
            fig.savefig(os.path.join(images_prefix, "predicted_class_test_set.png"), dpi=fig.dpi)

            # print(tf.math.confusion_matrix(true_labels.numpy(), predicted_labels.numpy(), num_classes=12).numpy())
            confusion_matrix = tf.math.confusion_matrix(true_labels.numpy(), predicted_labels.numpy()).numpy()
            print("Unnormalized confusion matrix")
            print(confusion_matrix)
            cm_fig = plot_confusion_matrix(confusion_matrix, self.dataset)
            cm_fig.savefig(os.path.join(images_prefix, "confusion_matrix.png"), dpi=fig.dpi)

    # def _update_learning_rate(self, epoch):
    #     #In the paper the learning rate is updated after certain
    #     # epochs to slow down learning.
    #     if epoch == 80 or epoch == 60 or epoch == 30:
    #         self.learning_rate = self.learning_rate * 0.2
