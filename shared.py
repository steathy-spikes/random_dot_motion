from multiprocessing import Value, sharedctypes, RawArray
import numpy as np
import ctypes
import time
from stimulus_module import StimulusModule
import pickle


class Shared():
    def __init__(self):
        # MAIN PARAMETERS
        self.stimulus_properties_direction_of_dots = Value('d', 45.0)
        self.stimulus_properties_coherence_of_dots = Value('d', 10)
        self.stimulus_properties_lifetime_of_dots = Value('d', 1.2)
        self.stimulus_properties_speed_of_dots = Value('d', 1.0)

        self.window_properties_x = Value('i', 1400)
        self.window_properties_y = Value('i', 600)
        self.window_properties_width = Value('i', 300)
        self.window_properties_height = Value('i', 300)
        self.window_properties_background = Value('d', 0)
        self.window_properties_radius = Value('d', 1)
        self.control_window_position_x = Value('i', 100)
        self.control_window_position_y = Value('i', 100)

        self.window_properties_update_requested = Value('b', 0)

        self.stimulus_properties_number_of_dots = Value('i', 1000)
        self.stimulus_properties_size_of_dots = Value('d', 0.01)

        self.stimulus_properties_brightness_of_dots = Value('d', 1.0)

        self.stimulus_properties_update_requested = Value('b', 0)

        self.running = Value('b', 1)

    def load_values(self):
        try:
            values = pickle.load(open("values.pickle", "rb"))
            self.window_properties_x.value = values["window_properties_x"]
            self.window_properties_y.value = values["window_properties_y"]
            self.window_properties_width.value = values["window_properties_width"]
            self.window_properties_height.value = values["window_properties_height"]
            self.window_properties_radius.value = values["window_properties_radius"]
            self.window_properties_background.value = values["window_properties_background"]
            self.control_window_position_x.value = values["control_window_position_x"]
            self.control_window_position_y.value = values["control_window_position_y"]

            self.stimulus_properties_number_of_dots.value = values["stimulus_properties_number_of_dots"]
            self.stimulus_properties_size_of_dots.value = values["stimulus_properties_size_of_dots"]
            self.stimulus_properties_speed_of_dots.value = values["stimulus_properties_speed_of_dots"]
            self.stimulus_properties_direction_of_dots.value = values["stimulus_properties_direction_of_dots"]
            self.stimulus_properties_coherence_of_dots.value = values["stimulus_properties_coherence_of_dots"]
            self.stimulus_properties_lifetime_of_dots.value = values["stimulus_properties_lifetime_of_dots"]
            self.stimulus_properties_brightness_of_dots.value = values["stimulus_properties_brightness_of_dots"]


        except Exception as e:
            print(e)

    def save_values(self):

        try:
            values = dict({})

            values["window_properties_x"] = self.window_properties_x.value
            values["window_properties_y"] = self.window_properties_y.value
            values["window_properties_width"] = self.window_properties_width.value
            values["window_properties_height"] = self.window_properties_height.value
            values["window_properties_radius"] = self.window_properties_radius.value
            values["window_properties_background"] = self.window_properties_background.value
            values["control_window_position_x"] = self.control_window_position_x.value
            values["control_window_position_y"] = self.control_window_position_y.value

            values["stimulus_properties_number_of_dots"] = self.stimulus_properties_number_of_dots.value
            values["stimulus_properties_size_of_dots"] = self.stimulus_properties_size_of_dots.value
            values["stimulus_properties_speed_of_dots"] = self.stimulus_properties_speed_of_dots.value
            values["stimulus_properties_coherence_of_dots"] = self.stimulus_properties_coherence_of_dots.value
            values["stimulus_properties_direction_of_dots"] = self.stimulus_properties_direction_of_dots.value
            values["stimulus_properties_lifetime_of_dots"] = self.stimulus_properties_lifetime_of_dots.value
            values["stimulus_properties_brightness_of_dots"] = self.stimulus_properties_brightness_of_dots.value

            pickle.dump(values, open("values.pickle", "wb"))
        except Exception as e:
            print(e)

    def start_threads(self, plot):
        StimulusModule(self, plot=plot).start()
