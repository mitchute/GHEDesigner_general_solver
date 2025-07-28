import time
from pathlib import Path

from District_system_class import GHEHPSystem

# from OpenGL_2D_class_GLFW import gl2D
from ghedesigner.ghe.runner_code import read_data_from_json_file

System = GHEHPSystem()
start_time = time.time()


def AnimationCallback(frame):
    # calculations needed to configure the picture
    # these could be done here or by calling a class method
    System.current_frame = (frame + 1) * 145


f_path = Path(__file__).parent / "3ghe-6hp_layout_input file.txt"


def main():
    with open(f_path) as f1:  # open the file for reading     # "3ghe-6hp_layout_input file.txt"
        data = f1.readlines()  # read the entire file as a list of strings
    # f1.close()  # close the file  ... very important

    System.read_GHEHPSystem_data(data)

    fluid, pipe, grout, soil, borehole, sim_params = read_data_from_json_file()
    System.solveSystem(fluid, pipe, grout, soil, borehole, sim_params)
    System.createOutput()
    System.current_frame = 1

    # # Draw the house, set the window width and height
    # gl2d = gl2D(None, System.drawnetwork, width=2000, height=1500)
    # gl2d.setViewSize(-10, 50, -10, 80, False)
    # gl2d.glWait()  # wait for the user to close the window

    # # Draw the house, set the window width and height
    # gl2d = gl2D(None, System.drawnetwork, width=2000, height=1500)
    # gl2d.setViewSize(-10, 50, -10, 80, False)
    # nframes = 59
    # gl2d.glStartAnimation(AnimationCallback, nframes, delaytime=0.1, reverse=False, repeat=False, reset=False)

    # gl2d.glWait()  # wait for the user to close the window

    print("Finished drawing 1")


main()

end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
