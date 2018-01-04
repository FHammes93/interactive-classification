import os
import cv2
import numpy as np
from scipy import spatial
import tensorflow as tf
import cPickle as pickle
from alexnet import AlexNet


class NearestNeighbor:

    K = 3
    neighbor_dict = {}

    def __init__(self):

        # The number of output classes of Alexnet
        num_classes = 1000

        # No train on the layers
        train_layers = []

        # TF placeholder for graph input and output
        self.x = tf.placeholder(tf.float32, [1, 227, 227, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        # Initialize model
        self.model = AlexNet(self.x, self.keep_prob, num_classes, train_layers)

        # Declare the feature-vector-dictionary
        self.load_neighbor_dict()

    def classify(self, name):

        nearest_neighbours = []
        for i in range(self.K):
            nearest_neighbours.append(("", np.inf))

        # Get the image
        image_path = "classification/pictures/" + name + ".jpeg"

        im1 = cv2.imread(image_path).astype(np.float32)
        im1 = im1 - np.mean(im1)

        # RGB to BGR
        im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

        features = self.get_features(np.array([im1]))

        # Compute all distances to the neighbours and search the nearest
        for key in self.neighbor_dict.keys():
            for neighbor_features in self.neighbor_dict[key]:

                distance = spatial.distance.euclidean(np.array(features[0]), neighbor_features)
                neighbor = (key, distance)

                # Check if the neighbor is nearer then the actual nearest neighbors
                for i in range(self.K):
                    if neighbor[1] < nearest_neighbours[i][1]:
                        tmp_neighbor = nearest_neighbours[i]
                        nearest_neighbours[i] = neighbor
                        neighbor = tmp_neighbor

        # Compute the probabilities of the label
        probabilities = {}
        for i in range(self.K):

            if nearest_neighbours[i][0] in probabilities.keys():
                probabilities[nearest_neighbours[i][0]] += 1.0

            else:
                probabilities[nearest_neighbours[i][0]] = 1.0

        print("\n")
        best_match = ("", 0.0)
        for key in probabilities.keys():

            probabilities[key] = (probabilities[key] / self.K) * 100
            print("The probability that the object is an '" + key + "' is " + str(int(probabilities[key])) + "%")

            # search highest probability
            if probabilities[key] > best_match[1]:
                best_match = (key, probabilities[key])

        # check if the estimation was correct or not
        while True:
            correct_estimation = raw_input("Is the shown object a '" + best_match[0] + "' ? (y/n)")

            # add the features to the neighbor dict if it was the correct estimation
            if correct_estimation == "y" or correct_estimation == "Y":
                self.neighbor_dict[best_match[0]] = np.concatenate((self.neighbor_dict[best_match[0]], features))
                return

            # ask what the correct label is
            elif correct_estimation == "n" or correct_estimation == "N":
                correct_label = raw_input("What is the correct label then? (correct spelling required!)")

                if correct_label in self.neighbor_dict.keys():
                    self.neighbor_dict[correct_label] = np.concatenate((self.neighbor_dict[correct_label], features))
                    return

                # ask if it is a new label
                else:
                    new_label = raw_input("Is this a complete new label? (y/n)")

                    if new_label == "y":
                        self.neighbor_dict[correct_label] = features
                        return

                    else:
                        print("then may check the correct spelling.")

            else:
                print("Wrong input, please check again.")

    def batch_learn(self, label, folder_path="classification/pictures/"):

        folder_path = folder_path + label

        # Creates an array with all images as arrays
        images = []
        for image in os.listdir(folder_path):
            if image.endswith(".jpg") or image.endswith(".jpeg"):

                # Get the image
                image_path = os.path.join(folder_path, image)
                im1 = (cv2.imread(image_path)[:, :, :3]).astype(np.float32)
                im1 = im1 - np.mean(im1)

                # RGB to BGR
                im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
                images.append(im1)

        features = self.get_features(np.array(images))

        if label not in self.neighbor_dict.keys():
            self.neighbor_dict[label] = features
        else:
            self.neighbor_dict[label] = np.concatenate((self.neighbor_dict[label], features))

        print "Finished " + label

    # Get the feature vectors of all the images
    def get_features(self, images):

        features = []

        # Link variable to model output
        score = self.model.fc7

        # Start Tensorflow session
        with tf.Session() as sess:

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load the pre-trained weights into the non-trainable layer
            self.model.load_initial_weights(sess)

            for im in images:

                # Skip image if the shape is not the right one
                if im.shape != (227, 227, 3):
                    print ("Image has wrong resolution!")
                    continue

                # Feed alexnet with the image
                output = sess.run(score, feed_dict={self.x: [im], self.keep_prob: 1.})
                features.append(output[0])

            return features

    def load_neighbor_dict(self):

        # Load the neighbor list out of the file
        try:
            print "open file"
            self.neighbor_dict = dict(pickle.load(open("classification/neighbor_list.p", "rb")))
            print "done"
        except IOError:
            print "no file found. start with empty dictionary"

    def show_labels(self):
        for key in self.neighbor_dict.keys():
            print(key + ": " + str(len(self.neighbor_dict[key])) + " feature vectors")

    def delete_label_neighbor_dict(self, label):
        if label in self.neighbor_dict.keys():
            del self.neighbor_dict[label]
        else:
            print("The label is not included.")

    def clear_neighbor_dict(self):
        self.neighbor_dict.clear()

    def save_neighbor_dict(self):
        pickle.dump(self.neighbor_dict, open("classification/neighbor_list.p", "wb"))
