import sys
import os
import cv2

class ExtractFrames:
    def __init__(self, params={}):
        self.params = params

        self.output_img_dir = params.get('output_img_dir')
        self.video_file     = params.get('video_file')

    def execute(self):
        cap = cv2.VideoCapture(self.video_file)

        if cap.isOpened() == False:
            print("Error in opening video file {}".format(self.video_file))

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()

            filename = "{}/{}.jpg".format(self.output_img_dir, frame_counter)
            
            print("Writing to {}".format(filename)
            cv2.imwrite(filename, frame)

            frame_counter += 1

        cap.release()

        print("Done.")
