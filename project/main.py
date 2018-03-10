from api.kinect import Kinect
from classification.my_classifier import KNearestNeighbor
from classification.my_classifier import NearestNeighbor
from api.audio import Audio


def main():

    type_classifier = raw_input("Which classifier you want to use? (1:kNN , 2: NN)")

    # init the named classifier and the kinect
    if type_classifier == "1":
        classifier = KNearestNeighbor()
        print("You are using K-Nearest-Neighbor")
    elif type_classifier == "2":
        classifier = NearestNeighbor()
        print("You are using Nearest-Neighbor")
    else:
        classifier = KNearestNeighbor()
        print("Input was invalid. You are using K-Nearest-Neighbor")

    # init the audio
    audio = Audio()
    speech = raw_input("Do you want to use the speech communication? (yes or no)")
    if speech == "y" or speech == "Y" or speech == "yes":
        speech = True
    else:
        speech = False

    kinect = Kinect()

    while True:

        print("\nC:classify | V: video | P:get picture | L:batch_learn | "
              "S:show_labels | D:delete_label | K:Modify K | Q:Quit and save \n")
        task = audio.input("What do you want to do ?", speech=speech)

        if task == "c" or task == "C" or task == "classify":
            if not kinect.connected:
                kinect.connect()
            kinect.get_picture(name="tmp_picture")
            kinect.disconnect()
            label, new_label = classifier.classify("tmp_picture", speech=speech)

            # If it was a new label, take some more pictures of the object
            if new_label:
                new_features = audio.input("\nTake another picture of this object now? (yes or no)", speech=speech)
                while new_features == "y" or new_features == "Y" or new_features == "yes":
                    if not kinect.connected:
                        kinect.connect()
                    kinect.get_picture(name="tmp_picture")
                    kinect.disconnect()
                    classifier.learn(label, image_name="tmp_picture")
                    new_features = audio.input("One more picture? (yes or no)", speech=speech)

        # starts a video from the kinect with the given scale
        elif task == "v" or task == "V" or task == "video":
            if not kinect.connected:
                kinect.connect()
            kinect.start_video()
            kinect.disconnect()

        # that pictures will be saved in pictures/new/
        elif task == "p" or task == "P" or task == "get picture":
            if not kinect.connected:
                kinect.connect()
            kinect.get_picture()
            kinect.disconnect()
            audio.output("Picture done.")

        # Modify k of the kNN, if kNN is used as classifier
        elif task == "k" or task == "K" or task == "modify k":
            if type_classifier != "1":
                print("You are not using k-NN. Option not available.")
                continue

            print("\nThe actual k is: " + str(classifier.K))
            new_k = int(audio.input("Which k do you want to use?(use keyboard) ", speech=False))

            if type(new_k) is int:
                classifier.K = new_k
            else:
                print("The given input is no integer.")

        # to learn with multiple pictures you have to create a folder in pictures named as the label
        elif task == "l" or task == "L" or task == "batch learn":
            label = audio.input("What label do you want to learn?", speech=speech)
            try:
                classifier.batch_learn(label)
            except OSError:
                print("The folder /pictures/" + label + " does not exist.")

        # shows all labels of neighbor_dict
        elif task == "s" or task == "S" or task == "show labels":
            classifier.show_labels()

        # deletes a label from the neighbor_dict
        elif task == "d" or task == "D" or task == "delete label":
            label = audio.input("What label do you want to delete? ", speech=speech)
            classifier.delete_label(label)

        # saves the new neighbour_list and exits
        elif task == "q" or task == "Q" or task == "quit and save":
            if kinect.connected:
                kinect.disconnect()
            classifier.close()
            break

        else:
            print("unknown input, please try again")

    print("Program closed.\n")


if __name__ == '__main__':
    main()
