import cv2
import sys
import numpy as np
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame


class Kinect:

    fn = None
    device = None
    serial = None
    listener = None
    connected = False

    # Frame-Filter Variables
    SCALE = 2
    OUTPUT_RESOLUTION = 227
    BOX_FILTER_SIZE = 3

    def __init__(self):

        # Import the pipeline which will be used with the kinect
        try:
            from pylibfreenect2 import OpenCLPacketPipeline
            self.pipeline = OpenCLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenGLPacketPipeline
                self.pipeline = OpenGLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline
                self.pipeline = CpuPacketPipeline()
        print("Packet pipeline:", type(self.pipeline).__name__)

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        # Get the serial number of the kinect
        self.serial = self.fn.getDeviceSerialNumber(0)

        self.image_counter = 1

    # Connects to the kinect and starts the stream
    def connect(self):

        # Connect to kinect
        self.device = self.fn.openDevice(self.serial, pipeline=self.pipeline)

        # Define RGB listener
        self.listener = SyncMultiFrameListener(FrameType.Color)

        # Register listeners
        self.device.setColorFrameListener(self.listener)

        # Get a frame from the listener
        self.device.startStreams(rgb=True, depth=False)

        self.connected = True

    # Takes an image out of the stream and saves it
    def get_picture(self, name=None):

        # Get a frame from the listener
        frames = self.listener.waitForNewFrame()

        # Extract the RBG frame
        color = frames["color"]
        _, filtered_color_frame = self.__filter_frame(color)

        # Save the frame as picture
        if name is None:
            cv2.imwrite("classification/pictures/new/" + str(self.image_counter) + ".jpeg", filtered_color_frame)
            self.image_counter += 1

        else:
            cv2.imwrite("classification/pictures/" + str(name) + ".jpeg", filtered_color_frame)

        # Release the frame from the listener
        self.listener.release(frames)

    def start_video(self):
        while True:
            # Get a frame from the listener
            frames = self.listener.waitForNewFrame()

            # Extract the RBG frame
            color = frames["color"]
            cut_frame, _ = self.__filter_frame(color, to_stream=True)

            # Show the cut stream of pictures
            cv2.imshow("cut", cut_frame)

            # Release the frame from the listener
            self.listener.release(frames)

            # Press q to exit loop and close all cv2-windows
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    # Filter the frame of the kinect to the desired scale of frame-filter variables
    def __filter_frame(self, frame, to_stream=False):
        # original frame resolution: (1080, 1920, 4)
        assert type(frame) is Frame

        # the desired pixel grid to get the desired resolution in the end
        cut_size = self.OUTPUT_RESOLUTION * self.SCALE + self.BOX_FILTER_SIZE - 1

        # the amount of rows/columns to cut away
        cut_rows = int((1080 - cut_size) / 2)
        cut_columns = int((1920 - cut_size) / 2)

        # the width of the filter from the middle to the edge
        filter_edge = (self.BOX_FILTER_SIZE - 1) / 2

        frame = frame.asarray()
        filtered_frame = np.zeros([self.OUTPUT_RESOLUTION, self.OUTPUT_RESOLUTION, 4], dtype=int)

        cut_frame = frame[cut_rows:cut_rows + cut_size, cut_columns:cut_columns + cut_size]

        if self.BOX_FILTER_SIZE != 1 and not to_stream:

            # go through complete output array
            for i in range(self.OUTPUT_RESOLUTION):
                for j in range(self.OUTPUT_RESOLUTION):

                    # use for every output pixel the whole filter
                    for m in range(-filter_edge, filter_edge + 1):
                        for n in range(-filter_edge, filter_edge + 1):
                            filtered_frame[i, j] += cut_frame[i * self.SCALE + filter_edge + m,
                                                              j * self.SCALE + filter_edge + n]

                    filtered_frame[i, j] = filtered_frame[i, j] / (self.BOX_FILTER_SIZE * self.BOX_FILTER_SIZE)

        return cut_frame, filtered_frame

    # Terminates the connection to the kinect
    def disconnect(self):

        self.device.stop()
        self.device.close()
        self.connected = False
