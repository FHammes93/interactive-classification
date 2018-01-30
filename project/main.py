from api.kinect import Kinect
from classification.my_classifier import NearestNeighbor


def main():


    # TODO: Wie kann man mit freenect2 auf die andere Grafikkarte zugreifen ?
    # TODO: testen mit aelteren treibern damit tensorflow nicht ueber CPU laeuft
    classifier = NearestNeighbor()
    kinect = Kinect()

    while True:

        task = raw_input("\nC:classify | V: video | P:get_picture | L:batch_learn | "
                         "S:show_labels | D:delete_label | K:Modify K | Q:save & exit \n")

        if task == "c" or task == "C":
            if not kinect.connected:
                kinect.connect()
            kinect.get_picture(name="tmp_picture")
            kinect.disconnect()
            label, new_label = classifier.classify("tmp_picture")

            # If it was a new label, take some more pictures of the object
            if new_label:
                new_features = raw_input("\nTake another picture of this object now? (y/n)")
                while new_features == "y" or new_features == "Y":
                    if not kinect.connected:
                        kinect.connect()
                    kinect.get_picture(name="tmp_picture")
                    kinect.disconnect()
                    classifier.learn(label, image_name="tmp_picture")
                    new_features = raw_input("Again? (y/n)")

        # starts a video from the kinect with the given scale
        elif task == "v" or task == "V":
            if not kinect.connected:
                kinect.connect()
            kinect.start_video()
            kinect.disconnect()

        # that pictures will be saved in pictures/new/
        elif task == "p" or task == "P":
            if not kinect.connected:
                kinect.connect()
            kinect.get_picture()
            kinect.disconnect()

        # Modify k of the kNN
        elif task == "k" or task == "K":
            print("\nThe actual k is: " + str(classifier.K))
            new_k = int(raw_input("Which k do you want to use?(integer) "))
            print(new_k.__class__)
            if type(new_k) is int:
                classifier.K = new_k
            else:
                print("The given input is no integer.")

        # to learn with multiple pictures you have to create a folder in pictures named as the label
        elif task == "l" or task == "L":
            label = raw_input("What label do you want to learn? ")
            try:
                classifier.batch_learn(label)
            except OSError:
                print("The folder /pictures/" + label + " does not exist.")

        # shows all labels of neighbor_dict
        elif task == "s" or task == "S":
            classifier.show_labels()

        # deletes a label from the neighbor_dict
        elif task == "d" or task == "D":
            label = raw_input("What label do you want to delete? ")
            classifier.delete_label_neighbor_dict(label)

        # saves the new neighbour_list and exits
        elif task == "q" or task == "Q":
            if kinect.connected:
                kinect.disconnect()
            classifier.close()
            break

        else:
            print("unknown input, please try again")

    print("Program closed.\n")


if __name__ == '__main__':
    main()
