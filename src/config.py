import argparse
import os


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("-", "", required= , help = '')
        self.parser.add_argument("-", "", required= , help = '')
        self.parser.add_argument("-", "", required= , help = '')
        self.parser.add_argument("-", "", required= , help = '')
        self.parser.add_argument("--debug", type=int, default=0, help = help='0: show nothing\n'
                                      '1: visualize pre-processed image and boxes\n'
                                      '2: visualize detections.')
        self.parser.add_argument("debug_dir", "", required= , help = '')
        self.parser.add_argument("-", "", required= , help = '')


args = parser.parse_args()
