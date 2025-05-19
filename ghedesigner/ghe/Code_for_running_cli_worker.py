from pathlib import Path
import json
from ghedesigner.ghe.manager import GroundHeatExchanger
import time
import traceback


def run_simulation_from_json(input_file: Path, output_dir: Path):
    with open(input_file, "r") as f:
        data = json.load(f)

    ghe = GroundHeatExchanger()
    ghe.set_fluid(**data["fluid"])

    ghe_data = data["ground-heat-exchanger"]["ghe1"]
    ghe.set_soil(**ghe_data["soil"])
    ghe.set_grout(**ghe_data["grout"])

    pipe = ghe_data["pipe"]
    ghe.set_single_u_tube_pipe(
        pipe["inner_diameter"], pipe["outer_diameter"],
        pipe["shank_spacing"], pipe["roughness"],
        pipe["conductivity"], pipe["rho_cp"]
    )

    borehole = ghe_data["borehole"]
    ghe.set_borehole(borehole["buried_depth"], borehole["diameter"])

    ghe.set_simulation_parameters(num_months=12)
    geo = ghe_data["geometric_constraints"]
    ghe.set_geometry_constraints_bi_rectangle(
        geo["max_height"], geo["min_height"],
        geo["length"], geo["width"],
        geo["b_min"], geo["b_max_x"], geo["b_max_y"]
    )

    # Dummy loads for testing
    ghe.set_ground_loads_from_hourly_list([100.0] * 8760)

    design = ghe_data["design"]
    ghe.set_design(
        flow_rate=design["flow_rate"],
        flow_type_str=design["flow_type"],
        max_eft=design["max_eft"],
        min_eft=design["min_eft"]
    )

    ghe.find_design()

    ghe.prepare_results("MyProject", "Manual runner", "Niranjan", "v1")
    ghe.write_output_files(output_dir)


# Run it manually
if __name__ == "__main__":
    input_path = Path("C:\\Users\\nbast\\GHEDesigner_fork\\demos\\find_design_bi_rectangle_single_u_tube.json")
    output_path = Path("C:\\Users\\nbast\\Desktop")

    start_time = time.time()

    try:
        run_simulation_from_json(input_path, output_path)
    except Exception as e:
        print("\n❌ Error occurred:", str(e))
        traceback.print_exc()
    finally:
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total simulation time: {elapsed_time:.2f} seconds")