from api.kinect import Kinect
from classification.my_classifier import NearestNeighbor


def main():


    # TODO: Wie kann man mit freenect2 auf die andere Grafikkarte zugreifen ?
    # TODO: testen mit aelteren treibern damit tensorflow nicht ueber CPU laeuft
    classifier = NearestNeighbor()
    kinect = Kinect()

    while True:

        task = raw_input("\nC:classify | V: video | P:get_picture | L:batch_learn | "
                         "S:show_labels | D:delete_label | Q:save & exit \n")

        if task == "c" or task == "C":
            if not kinect.connected:
                kinect.connect()
            kinect.get_picture(name="classify")
            kinect.disconnect()
            classifier.classify("classify")

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

        # to learn with multiple pictures you have to create a folder in pictures named as the label
        elif task == "l" or task == "L":
            label = raw_input("What label do you want to learn?")
            try:
                classifier.batch_learn(label)
            except OSError:
                print("The folder /pictures/" + label + " does not exist.")

        # shows all labels of neighbor_dict
        elif task == "s" or task == "S":
            classifier.show_labels()

        # deletes a label from the neighbor_dict
        elif task == "d" or task == "D":
            label = raw_input("What label do you want to delete?")
            classifier.delete_label_neighbor_dict(label)

        # saves the new neighbour_list and exits
        elif task == "q" or task == "Q":
            if kinect.connected:
                kinect.disconnect()
            classifier.save_neighbor_dict()
            break

        else:
            print("unknown input, please try again")


if __name__ == '__main__':
    main()
