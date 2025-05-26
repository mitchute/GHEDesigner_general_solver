from pathlib import Path
import json
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.media import Soil, Grout, Pipe, GHEFluid
from pygfunction.boreholes import Borehole
from ghedesigner.ghe.simulation import SimulationParameters ## NOT REQUIRED???check
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
import time

# ✅ Start timing before simulation setup
start_time = time.time()

# Read JSON file
json_path = Path("C:/Users/nbast/GHEDesigner_fork/demos/find_design_bi_rectangle_single_u_tube.json")
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract input values
fluid_data = data["fluid"]
soil_data = data["ground-heat-exchanger"]["ghe1"]["soil"]
grout_data = data["ground-heat-exchanger"]["ghe1"]["grout"]
pipe_data = data["ground-heat-exchanger"]["ghe1"]["pipe"]
borehole_data = data["ground-heat-exchanger"]["ghe1"]["borehole"]
geometric_data = data["ground-heat-exchanger"]["ghe1"]["geometric_constraints"]

# Construct objects
fluid = GHEFluid(
    fluid_data["fluid_name"],
    fluid_data["concentration_percent"],
    fluid_data["temperature"]
)
# Pipe object (Single U-tube)
r_in = pipe_data["inner_diameter"] / 2.0
r_out = pipe_data["outer_diameter"] / 2.0
s = pipe_data["shank_spacing"]

pipe_positions = Pipe.place_pipes(s, r_out, 1)

pipe = Pipe(
    pipe_positions,
    r_in,
    r_out,
    s,
    pipe_data["roughness"],
    pipe_data["conductivity"],
    pipe_data["rho_cp"]
)

soil = Soil(soil_data["conductivity"], soil_data["rho_cp"], soil_data["undisturbed_temp"])
grout = Grout(grout_data["conductivity"], grout_data["rho_cp"])
borehole = Borehole(100.0, borehole_data["buried_depth"], borehole_data["diameter"] / 2.0, 0.0, 0.0)

# Simulation parameters
sim_params = SimulationParameters(num_months=12)
sim_params.set_design_heights(geometric_data["max_height"], geometric_data["min_height"])

# Dummy g-function calculation (log_time and coordinates are provided in ground_heat_exchangers.py)
h_values = [100.0]
log_time = [-10 + i*(14/24) for i in range(25)]
r_b = borehole.r_b
depth = borehole.D

# Create GHE object
hourly_ground_loads = [0.0] * 8760  # Not used in _simulate_detailed()
ghe_obj = GHE(
    v_flow_system=0.5,  # Dummy value
    b_spacing=5.0,
    bhe_type=BHPipeType.SINGLEUTUBE,
    fluid=fluid,
    borehole=borehole,
    pipe=pipe,
    grout=grout,
    soil=soil,
    sim_params=sim_params,
    hourly_extraction_ground_loads=hourly_ground_loads
)

# Run simulation

# 👇 Get b/H for each GHE and generate the g-function
b_over_h1 = ghe_obj.B_spacing / ghe_obj.bhe1.b.H
g1, _ = ghe_obj.grab_g_function(ghe_obj.gFunction1, b_over_h1, ghe_obj.bhe1_eq)

# 👇 Run the simulation with one of the g-functions (used as reference)
ghe_obj._simulate_detailed(g1)

# ✅ End timing after simulation
end_time = time.time()
elapsed_time = end_time - start_time
print(f"✅ Simulation completed in {elapsed_time:.2f} seconds.")