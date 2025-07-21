from District_system_class import GHEHPSystem
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle, gl2DText,gl2DArrow,gl2DArc
from ghedesigner.ghe.runner_code import read_data_from_json_file
import time

start_time = time.time()

def main():
    f1 = open("3ghe-6hp_layout_input file.txt", 'r')  # open the file for reading
    data = f1.readlines()  # read the entire file as a list of strings
    f1.close()  # close the file  ... very important

    System = GHEHPSystem()
    System.read_GHEHPSystem_data(data)

    fluid, pipe, grout, soil, borehole, sim_params = read_data_from_json_file()
    System.solveSystem(fluid, pipe, grout, soil, borehole, sim_params)
    System.createOutput()

    # Draw the house, set the window width and height
    gl2d = gl2D(None, System.drawnetwork, width=2000, height=1500)
    gl2d.setViewSize(-10, 50, -10, 80, False)
    gl2d.glWait()  # wait for the user to close the window

    print("Finished drawing 1")


    #System.drawnetwork()
    #System.precalculate()
    # loop over all devices and tell all devices to precalculate their important numbers
    #System.solve()

    # System.solve will loop over all times
        # loop over all devices and tell all devices to calculate their important numbers
        # loop over all devices, grab the numbers, and put them in the matrix
        # solve the matrix
        # loop over all devices to post process



    # #✅ Check how Building instances are stored
    # for a in stored_data.nodes:
    #     print(a.x, a.y)

main()

end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")





