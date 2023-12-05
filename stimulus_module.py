from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from multiprocessing import Process
import numpy as np
import os
import cv2
import sys

PREV_FRAME = None
TIMESTEP = -3


def optical_flow_image(prev_frame, curr_frame):
    # code based from https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    mask_x = np.zeros_like(prev_frame)
    mask_y = np.zeros_like(prev_frame)
    try:
        # Sets image saturation to maximum
        mask_x[..., 1] = 255
        mask_y[..., 1] = 255

        flow = cv2.calcOpticalFlowFarneback(prev=prev_gray, next=curr_gray,
                                            flow=None,
                                            pyr_scale=0.5, levels=3, winsize=3, iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # print("angle", angle.shape)
        # hue -> The colour determines the angle
        magnitude_x = magnitude * np.abs(np.cos(angle))
        magnitude_y = magnitude * np.abs(np.sin(angle))
        angle[np.where(np.logical_and(angle > 0, angle < 180))] = 255
        angle[np.where(np.logical_and(angle > 180, angle < 360))] = 1
        # if angle > 0 and angle < 180:
        #     angle = 255
        # if angle > 180 and angle < 360:
        #     angle = 1
        # mask_x[..., 0] = angle * 180 / np.pi / 2
        mask_x[..., 0] = angle
        mask_y[..., 0] = angle
        # value -> The value of hue which is determined by magnitude
        mask_x[..., 2] = cv2.normalize(magnitude_x, None, 0, 255, cv2.NORM_MINMAX)
        mask_y[..., 2] = cv2.normalize(magnitude_y, None, 0, 255, cv2.NORM_MINMAX)

        optical_flow_rgb_x = cv2.cvtColor(mask_x, cv2.COLOR_HSV2BGR)
        optical_flow_rgb_y = cv2.cvtColor(mask_y, cv2.COLOR_HSV2BGR)
    except cv2.error:
        optical_flow_rgb_x = cv2.cvtColor(mask_x, cv2.COLOR_HSV2BGR)
        optical_flow_rgb_y = cv2.cvtColor(mask_y, cv2.COLOR_HSV2BGR)
    return optical_flow_rgb_x, optical_flow_rgb_y


dot_motion_coherence_shader = [
    """ #version 140

        uniform sampler2D p3d_Texture0;
        uniform mat4 p3d_ModelViewProjectionMatrix;

        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;
        uniform int number_of_dots;
        uniform float size_of_dots;
        uniform float radius;

        out float dot_color;

        void main(void) {
            vec4 newvertex;
            float dot_i;
            float dot_x, dot_y;
            float maxi = 10000.0;
            vec4 dot_properties;

            dot_i = float(p3d_Vertex[1]);
            dot_properties = texture(p3d_Texture0, vec2(dot_i/maxi, 0.0));

            dot_x = dot_properties[2];
            dot_y = dot_properties[1];
            dot_color = dot_properties[0];

            newvertex = p3d_Vertex;

            if (dot_x*dot_x + dot_y*dot_y > radius*radius || dot_i > number_of_dots) { // only plot a certain number of dots in a circle
                newvertex[0] = 0.0;
                newvertex[1] = 0.0;
                newvertex[2] = 0.0;
            } else {
                newvertex[0] = p3d_Vertex[0]*size_of_dots+dot_x;
                newvertex[1] = 0.75;
                newvertex[2] = p3d_Vertex[2]*size_of_dots+dot_y;
            }

            gl_Position = p3d_ModelViewProjectionMatrix * newvertex;
        }
    """,

    """ #version 140
        in float dot_color;
        out vec4 frag_color;

        void main() {
            frag_color = vec4(dot_color, dot_color, dot_color, 1);
        }
    """
]


class MyApp(ShowBase):
    def __init__(self, shared, plot):

        self.shared = shared
        self.plot = plot

        loadPrcFileData("",
                        """fullscreen 0
                           gl-version 3 2
                           win-origin 100 100
                           win-size 800 800
                           sync-video 0
                           undecorated 1
                           load-display pandagl""")

        ShowBase.__init__(self)

        ############
        # Update the lense
        self.disableMouse()

        ############
        # Compile the motion shader
        self.compiled_dot_motion_shader = Shader.make(Shader.SLGLSL, dot_motion_coherence_shader[0],
                                                      dot_motion_coherence_shader[1])

        filepath = os.path.join(os.path.split(__file__)[0], "circles.bam")
        self.circles = self.loader.loadModel(Filename.fromOsSpecific(filepath))
        self.circles.reparentTo(self.render)
        self.circles.setShaderInput("number_of_dots", self.shared.stimulus_properties_number_of_dots.value)
        self.circles.setShaderInput("size_of_dots", self.shared.stimulus_properties_size_of_dots.value)
        self.circles.setShaderInput("radius", self.shared.window_properties_radius.value)
        self.setBackgroundColor(self.shared.window_properties_background.value,
                                self.shared.window_properties_background.value,
                                self.shared.window_properties_background.value, 1)

        self.dummytex = Texture("dummy texture")
        self.dummytex.setup2dTexture(10000, 1, Texture.T_float, Texture.FRgb32)
        self.dummytex.setMagfilter(Texture.FTNearest)

        ts1 = TextureStage("part2")
        ts1.setSort(-100)

        self.circles.setTexture(ts1, self.dummytex)
        self.circles.setShader(self.compiled_dot_motion_shader)

        self.dots_position = np.empty((1, 10000, 3)).astype(np.float32)
        self.dots_position[0, :, 0] = 2 * np.random.random(10000).astype(np.float32) - 1  # x
        self.dots_position[0, :, 1] = 2 * np.random.random(10000).astype(np.float32) - 1  # y
        self.dots_position[0, :, 2] = np.ones(10000) * self.shared.stimulus_properties_brightness_of_dots.value

        memoryview(self.dummytex.modify_ram_image())[:] = self.dots_position.tobytes()

        self.last_time = 0
        self.shared.stimulus_properties_update_requested.value = 1
        self.shared.window_properties_update_requested.value = 1

        self.task_mgr.add(self.update_stimulus, "update_stimulus")

    def update_stimulus(self, task):

        #######
        # Listen to the commands coming from the gui
        if self.shared.running.value == 0:
            sys.exit()

        if self.shared.window_properties_update_requested.value == 1:
            self.shared.window_properties_update_requested.value = 0

            props = WindowProperties()
            props.setSize(self.shared.window_properties_height.value, self.shared.window_properties_width.value)
            props.setOrigin(self.shared.window_properties_x.value, self.shared.window_properties_y.value)

            self.win.requestProperties(props)

            self.lens = PerspectiveLens()
            self.lens.setFov(90, 90)
            self.lens.setNearFar(0.001, 1000)
            self.lens.setAspectRatio(self.shared.window_properties_height.value /
                                     self.shared.window_properties_width.value)

            self.cam.node().setLens(self.lens)

            self.setBackgroundColor(self.shared.window_properties_background.value,
                                    self.shared.window_properties_background.value,
                                    self.shared.window_properties_background.value, 1)

            self.circles.setShaderInput("radius", self.shared.window_properties_radius.value)

        if self.shared.stimulus_properties_update_requested.value == 1:
            self.shared.stimulus_properties_update_requested.value = 0

            random_vector = np.random.randint(100, size=10000)
            self.coherent_change_vector_ind = np.where(
                random_vector < self.shared.stimulus_properties_coherence_of_dots.value)

            ######
            # Update the shader variables
            self.circles.setShaderInput("number_of_dots", self.shared.stimulus_properties_number_of_dots.value)
            self.circles.setShaderInput("size_of_dots", self.shared.stimulus_properties_size_of_dots.value)

        #######
        # Continously update the dot stimulus
        dt = task.time - self.last_time
        self.last_time = task.time

        #####
        self.dots_position[0, :, 0][self.coherent_change_vector_ind] += np.cos(
            self.shared.stimulus_properties_direction_of_dots.value * np.pi / 180) * \
                                                                        self.shared.stimulus_properties_speed_of_dots.value * \
                                                                        dt
        self.dots_position[0, :, 1][self.coherent_change_vector_ind] += np.sin(
            self.shared.stimulus_properties_direction_of_dots.value * np.pi / 180) * \
                                                                        self.shared.stimulus_properties_speed_of_dots.value * \
                                                                        dt

        # Randomly redraw dot with a short lifetime
        k = np.random.random(10000)
        if self.shared.stimulus_properties_lifetime_of_dots.value == 0:
            ind = np.where(k >= 0)[0]
        else:
            ind = np.where(k < dt / self.shared.stimulus_properties_lifetime_of_dots.value)[0]

        self.dots_position[0, :, 0][ind] = 2 * np.random.random(len(ind)).astype(np.float32) - 1  # x
        self.dots_position[0, :, 1][ind] = 2 * np.random.random(len(ind)).astype(np.float32) - 1  # y
        self.dots_position[0, :, 2] = np.ones(10000) * self.shared.stimulus_properties_brightness_of_dots.value

        # Wrap them
        self.dots_position[0, :, 0] = (self.dots_position[0, :, 0] + 1) % 2 - 1
        self.dots_position[0, :, 1] = (self.dots_position[0, :, 1] + 1) % 2 - 1

        memoryview(self.dummytex.modify_ram_image())[:] = self.dots_position.tobytes()
        # # memoryview(self.dummytex.get_ram_image())
        # print(self.win.size)
        # print(type(memoryview(self.dummytex.get_ram_image())))
        # img = np.frombuffer(memoryview((self.circles.get_texture().get_ram_image())))
        #
        # screenshot = PNMImage()
        #
        # print(img, img.shape)

        screenshot = self.win.getScreenshot()
        screenshot_byte_array = screenshot.getRamImageAs('RGBA')

        # Convert the screenshot to a NumPy array
        img_array = np.frombuffer(screenshot_byte_array, dtype=np.uint8)
        img_array = img_array.reshape((screenshot.getYSize(), screenshot.getXSize(), 4))

        print("img_array_shape", img_array.shape)

        # Optionally, remove the alpha channel if not needed
        img_array = img_array[:, :, :3]
        # print(img_array.shape)

        # img = img.reshape((400, 400, 1))
        img = img_array[::-1]

        global PREV_FRAME
        global TIMESTEP

        # print(PREV_FRAME)
        # print("img_shape", img.shape)

        if PREV_FRAME is None:
            PREV_FRAME = img
            # print("img_shape", PREV_FRAME.shape)

        optical_flow_img_x, optical_flow_img_y = optical_flow_image(prev_frame=PREV_FRAME, curr_frame=img)

        PREV_FRAME = img

        print("optical_flow_img", optical_flow_img_x.shape)

        if self.plot:
            n_dots = self.shared.stimulus_properties_number_of_dots.value
            coherence = self.shared.stimulus_properties_coherence_of_dots.value
            direction = self.shared.stimulus_properties_direction_of_dots.value
            lifetime = self.shared.stimulus_properties_lifetime_of_dots.value
            size = self.shared.stimulus_properties_size_of_dots.value
            speed = self.shared.stimulus_properties_speed_of_dots.value


            dir_name = f"coh_{coherence}_dir_{direction}_speed_{speed}_ndots_{n_dots}_lifetime_{lifetime}_size_{size}"
            total_dir = os.path.join("saved_runs", dir_name)
            if not os.path.exists(total_dir):
                os.makedirs(total_dir)

            total_img_dir = os.path.join(total_dir, "img")
            if not os.path.exists(total_img_dir):
                os.makedirs(total_img_dir)

            total_npy_dir = os.path.join(total_dir, "npy")
            if not os.path.exists(total_npy_dir):
                os.makedirs(total_npy_dir)

            # the first frame is 800 by 800.
            # The starting TIMESTEP is -3 to not save the first few frames
            if TIMESTEP >= 0:

                np.save(file=os.path.join(total_npy_dir, f"optical_flow_img_x_{TIMESTEP}.npy"), arr=optical_flow_img_x)
                np.save(file=os.path.join(total_npy_dir, f"optical_flow_img_y_{TIMESTEP}.npy"), arr=optical_flow_img_y)

                cv2.imwrite(os.path.join(total_img_dir, f"optical_flow_img_x_{TIMESTEP}.png"), optical_flow_img_x)
                cv2.imwrite(os.path.join(total_img_dir, f"optical_flow_img_y_{TIMESTEP}.png"), optical_flow_img_y)

            # np.save(file=os.path.join(total_img_dir, f"optical_flow_img_x_{TIMESTEP}"), arr=optical_flow_img_x)
            # np.save(file=os.path.join(total_npy_dir, f"optical_flow_img_y_{TIMESTEP}"), arr=optical_flow_img_y)

            TIMESTEP += 1

        cv2.imshow('img_horizontal', optical_flow_img_x)
        cv2.imshow('img_vertical', optical_flow_img_y)
        # cv2.waitKey(0)

        return task.cont


class StimulusModule(Process):
    def __init__(self, shared, plot=False):
        Process.__init__(self)

        self.shared = shared
        self.plot = plot

    def run(self):
        app = MyApp(self.shared, self.plot)
        app.run()
