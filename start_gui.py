"""
(C) Armin Bahl, Dec. 2019
arminbahl@fas.harvard.edu
"""

import argparse

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = vars(ap.parse_args())

    plot = args["plot"]
    print(f"Plot: {plot}")


    from shared import Shared

    shared = Shared()
    # shared.load_values()
    shared.start_threads(plot = plot)

    import os
    import sys
    import PyQt5
    from PyQt5 import QtCore, QtGui, uic, QtWidgets

    # Some properties that one might need to set on high-res screens
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    class GUI_Dialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__()

            self.shared = shared
            path = os.path.dirname(__file__)

            uic.loadUi(os.path.join(path, "gui.ui"), self)

            self.spinBox_window_properties_x.setValue(self.shared.window_properties_x.value)
            self.spinBox_window_properties_y.setValue(self.shared.window_properties_y.value)
            self.spinBox_window_properties_width.setValue(self.shared.window_properties_width.value)
            self.spinBox_window_properties_height.setValue(self.shared.window_properties_height.value)
            self.doubleSpinBox_window_properties_radius.setValue(self.shared.window_properties_radius.value)
            self.doubleSpinBox_window_properties_background.setValue(self.shared.window_properties_background.value)

            self.spinBox_stimulus_properties_number_of_dots.setValue(self.shared.stimulus_properties_number_of_dots.value)
            self.doubleSpinBox_stimulus_properties_size_of_dots.setValue(self.shared.stimulus_properties_size_of_dots.value)
            self.doubleSpinBox_stimulus_properties_speed_of_dots.setValue(self.shared.stimulus_properties_speed_of_dots.value)
            self.doubleSpinBox_stimulus_properties_lifetime_of_dots.setValue(self.shared.stimulus_properties_lifetime_of_dots.value)
            self.doubleSpinBox_stimulus_properties_coherence_of_dots.setValue(self.shared.stimulus_properties_coherence_of_dots.value)
            self.doubleSpinBox_stimulus_properties_direction_of_dots.setValue(self.shared.stimulus_properties_direction_of_dots.value)
            self.doubleSpinBox_stimulus_properties_brightness_of_dots.setValue(self.shared.stimulus_properties_brightness_of_dots.value)

            self.spinBox_window_properties_x.valueChanged.connect(self.spinBox_window_properties_x_valueChanged)
            self.spinBox_window_properties_y.valueChanged.connect(self.spinBox_window_properties_y_valueChanged)
            self.spinBox_window_properties_width.valueChanged.connect(self.spinBox_window_properties_width_valueChanged)
            self.spinBox_window_properties_height.valueChanged.connect(self.spinBox_window_properties_height_valueChanged)
            self.doubleSpinBox_window_properties_radius.valueChanged.connect(self.doubleSpinBox_window_properties_radius_valueChanged)
            self.doubleSpinBox_window_properties_background.valueChanged.connect(self.doubleSpinBox_window_properties_background_valueChanged)

            self.spinBox_stimulus_properties_number_of_dots.valueChanged.connect(self.spinBox_stimulus_properties_number_of_dots_valueChanged)
            self.doubleSpinBox_stimulus_properties_size_of_dots.valueChanged.connect(self.doubleSpinBox_stimulus_properties_size_of_dots_valueChanged)
            self.doubleSpinBox_stimulus_properties_speed_of_dots.valueChanged.connect(self.doubleSpinBox_stimulus_properties_speed_of_dots_valueChanged)
            self.doubleSpinBox_stimulus_properties_coherence_of_dots.valueChanged.connect(self.doubleSpinBox_stimulus_properties_coherence_of_dots_valueChanged)
            self.doubleSpinBox_stimulus_properties_lifetime_of_dots.valueChanged.connect(self.doubleSpinBox_stimulus_properties_lifetime_of_dots_valueChanged)
            self.doubleSpinBox_stimulus_properties_direction_of_dots.valueChanged.connect(self.doubleSpinBox_stimulus_properties_direction_of_dots_valueChanged)
            self.doubleSpinBox_stimulus_properties_brightness_of_dots.valueChanged.connect(self.doubleSpinBox_stimulus_properties_brightness_of_dots_valueChanged)

            # move the window to where it has been last time
            self.move(self.shared.control_window_position_x.value,
                      self.shared.control_window_position_y.value)

        def doubleSpinBox_stimulus_properties_size_of_dots_valueChanged(self):
            self.shared.stimulus_properties_size_of_dots.value = self.doubleSpinBox_stimulus_properties_size_of_dots.value()
            self.shared.stimulus_properties_update_requested.value = 1


        def spinBox_stimulus_properties_number_of_dots_valueChanged(self):
            self.shared.stimulus_properties_number_of_dots.value = self.spinBox_stimulus_properties_number_of_dots.value()
            self.shared.stimulus_properties_update_requested.value = 1

        def doubleSpinBox_stimulus_properties_speed_of_dots_valueChanged(self):
            self.shared.stimulus_properties_speed_of_dots.value = self.doubleSpinBox_stimulus_properties_speed_of_dots.value()

        def doubleSpinBox_stimulus_properties_coherence_of_dots_valueChanged(self):
            self.shared.stimulus_properties_coherence_of_dots.value = self.doubleSpinBox_stimulus_properties_coherence_of_dots.value()
            self.shared.stimulus_properties_update_requested.value = 1

        def doubleSpinBox_stimulus_properties_lifetime_of_dots_valueChanged(self):
            self.shared.stimulus_properties_lifetime_of_dots.value = self.doubleSpinBox_stimulus_properties_lifetime_of_dots.value()

        def doubleSpinBox_stimulus_properties_direction_of_dots_valueChanged(self):
            self.shared.stimulus_properties_direction_of_dots.value = self.doubleSpinBox_stimulus_properties_direction_of_dots.value()

        def doubleSpinBox_stimulus_properties_brightness_of_dots_valueChanged(self):
            self.shared.stimulus_properties_brightness_of_dots.value = self.doubleSpinBox_stimulus_properties_brightness_of_dots.value()

        def spinBox_window_properties_x_valueChanged(self):
            self.shared.window_properties_x.value = self.spinBox_window_properties_x.value()
            self.shared.window_properties_update_requested.value = 1

        def spinBox_window_properties_y_valueChanged(self):
            self.shared.window_properties_y.value = self.spinBox_window_properties_y.value()
            self.shared.window_properties_update_requested.value = 1

        def spinBox_window_properties_width_valueChanged(self):
            self.shared.window_properties_width.value = self.spinBox_window_properties_width.value()
            self.shared.window_properties_update_requested.value = 1

        def spinBox_window_properties_height_valueChanged(self):
            self.shared.window_properties_height.value = self.spinBox_window_properties_height.value()
            self.shared.window_properties_update_requested.value = 1

        def doubleSpinBox_window_properties_radius_valueChanged(self):
            self.shared.window_properties_radius.value = self.doubleSpinBox_window_properties_radius.value()
            self.shared.window_properties_update_requested.value = 1

        def doubleSpinBox_window_properties_background_valueChanged(self):
            self.shared.window_properties_background.value = self.doubleSpinBox_window_properties_background.value()
            self.shared.window_properties_update_requested.value = 1

        def closeEvent(self, event):
            self.shared.control_window_position_x.value = self.pos().x()
            self.shared.control_window_position_y.value = self.pos().y()

            self.shared.running.value = 0
            self.close()


    # print("ak,sdjfkladsjf")
    # exit(0)

    app = QtWidgets.QApplication(sys.argv)

    main = GUI_Dialog()



    main.show()
    app.exec_()

    shared.running.value = 0
    shared.save_values()
