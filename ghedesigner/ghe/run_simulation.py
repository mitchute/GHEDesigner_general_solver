import time

from District_system_class import GHEHPSystem

from ghedesigner.ghe.runner_code import read_data_from_json_file

System = GHEHPSystem()
start_time = time.time()


def main():
    with open("3ghe-6hp_layout_input file.txt") as f1:
        data = f1.readlines()

    System.read_ghe_hp_system_data(data)

    fluid, pipe, grout, soil, borehole, sim_params = read_data_from_json_file()
    System.solve_system(fluid, pipe, grout, soil, borehole, sim_params)
    System.create_output()
    System.current_frame = 1


if __name__ == "__main__":
    main()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
