from math import ceil

import numpy as np
from pygfunction.boreholes import Borehole
from scipy.interpolate import interp1d

from ghedesigner.constants import SEC_IN_HR, TWO_PI, VERSION
from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.ghe.coaxial_borehole import get_bhe_object
from ghedesigner.ghe.gfunction import GFunction, calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_loads import HybridLoad
from ghedesigner.ghe.simulation import SimulationParameters
from ghedesigner.media import Grout, Pipe, Soil
from ghedesigner.utilities import solve_root
from pathlib import Path

# My addition for multiple GHE-HP begins here
import pandas as pd

# Reading the Heat Pump Loads file and case file
repo_root = Path(__file__).resolve().parents[2]
file_path_hp_loads = repo_root / "buildingloads" / "MultipleGHE-HP_heatpumploads_DULUTH_3ghe-6hp.csv"
file_path_cases = repo_root / "cases" / "3ghe-6hp_case.csv"

selected_configuration = "3ghe-6hp"
selected_case = "Case 1"


def read_hp_loads(hp_loads_path):
    """Reads HP loads from CSVs with MultiIndex columns."""
    df1 = pd.read_csv(hp_loads_path, header=[0, 1, 2, 3])
    df1.columns = pd.MultiIndex.from_tuples([(int(l1), int(l2), int(l3),  l4) for l1, l2, l3, l4 in df1.columns])

    return df1


df1 = read_hp_loads(file_path_hp_loads)


def extract_time_array():
    """
    Extracts the time values from the specified location in the dataframe
    and converts them into a NumPy array.

    Returns:
    A NumPy array containing the extracted time values.
    """
    array = df1.iloc[:, 0].to_numpy()
    return array


def time_array_size():
    size = len(extract_time_array())
    return size


time_array = extract_time_array()
n = time_array_size()


def read_case_data(cases_path, selected_case: str):
    """Reads selected case data and extracts parameters."""
    df_cases = pd.read_csv(cases_path)
    case_data = df_cases[df_cases["Case"] == selected_case].iloc[0]

    # Borehole layout
    n_rows1, n_cols1 = int(case_data["n_rows1"]), int(case_data["n_columns1"])
    n_rows2, n_cols2 = int(case_data["n_rows2"]), int(case_data["n_columns2"])
    n_rows3, n_cols3 = int(case_data["n_rows3"]), int(case_data["n_columns3"])
    b_spacing = case_data["b_spacing"]

    # Heat pump design flowrates
    (m_hp1_design, m_hp2_design, m_hp3_design,
     m_hp4_design, m_hp5_design, m_hp6_design) = (
        case_data["m_hp1"], case_data["m_hp2"], case_data["m_hp3"],
        case_data["m_hp4"], case_data["m_hp5"], case_data["m_hp6"]
    )

    # Boreholes per GHE and GHE flowrates
    nbh1, nbh2, nbh3 = case_data["nbh_1GHE"], case_data["nbh_2GHE"], case_data["nbh_3GHE"]
    m_ghe1, m_ghe2, m_ghe3 = case_data["m_ghe1"], case_data["m_ghe2"], case_data["m_ghe3"]

    # Single HP values
    (m_hp1_single_as, m_hp2_single_as, m_hp3_single_as, m_hp_single_wtw) = (
        case_data["m_hp1_single_WTA"], case_data["m_hp2_single_WTA"],
        case_data["m_hp3_single_WTA"], case_data["m_hp_single_WTW"]
    )

    return (n_rows1, n_cols1, n_rows2, n_cols2, n_rows3, n_cols3, b_spacing,
            m_hp1_design, m_hp2_design, m_hp3_design, m_hp4_design, m_hp5_design, m_hp6_design,
            nbh1, nbh2, nbh3, m_ghe1, m_ghe2, m_ghe3,
            m_hp1_single_as, m_hp2_single_as, m_hp3_single_as, m_hp_single_wtw)


(n_rows1, n_cols1, n_rows2, n_cols2, n_rows3, n_cols3, b_spacing, m_hp1_design, m_hp2_design, m_hp3_design,
 m_hp4_design, m_hp5_design, m_hp6_design, nbh1, nbh2, nbh3, m_ghe1, m_ghe2, m_ghe3, m_hp1_single_as, m_hp2_single_as,
 m_hp3_single_as, m_hp_single_wtw) = (read_case_data(file_path_cases, selected_case))

# My addition for multiple GHE ends here for this part. next changes are made on clss BaseGHE


class BaseGHE:
    def __init__(
        self,
        v_flow_system: float,
        b_spacing: float,
        bhe_type: BHPipeType,
        fluid,
        borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        sim_params: SimulationParameters,
        hourly_extraction_ground_loads: list,
        field_type="N/A",
        field_specifier="N/A",
    ) -> None:
        self.fieldType = field_type
        self.fieldSpecifier = field_specifier
        self.V_flow_system = v_flow_system
        self.B_spacing = b_spacing
        self.nbh1 = nbh1
        self.nbh2 = nbh2
        self.nbh3 = nbh3
        self.m_ghe1 = m_ghe1
        self.m_ghe2 = m_ghe2
        self.m_ghe3 = m_ghe3
        self.m_ghe1_borehole = float(self.m_ghe1)/float(self.nbh1)
        self.m_ghe2_borehole = float(self.m_ghe2) / float(self.nbh2)
        self.m_ghe3_borehole = float(self.m_ghe3) / float(self.nbh3)

        self.gFunction1 = None
        self.gFunction2 = None
        self.gFunction3 = None
        self.log_time = np.linspace(-10, 4, 25).tolist()

        # Borehole Heat Exchanger
        self.bhe_type = bhe_type
        self.bhe1 = get_bhe_object(bhe_type, self.m_ghe1_borehole, fluid, borehole, pipe, grout, soil)
        self.bhe2 = get_bhe_object(bhe_type, self.m_ghe2_borehole, fluid, borehole, pipe, grout, soil)
        self.bhe3 = get_bhe_object(bhe_type, self.m_ghe3_borehole, fluid, borehole, pipe, grout, soil)


        # Equivalent borehole Heat Exchanger

        self.bhe1_eq = self.bhe1.to_single()
        self.bhe2_eq = self.bhe2.to_single()
        self.bhe3_eq = self.bhe3.to_single()

        # Radial numerical short time step
        self.bhe1_eq.calc_sts_g_functions()
        self.bhe2_eq.calc_sts_g_functions()
        self.bhe3_eq.calc_sts_g_functions()

        # Additional simulation parameters
        self.sim_params = sim_params
        # Hourly ground extraction loads
        # Building cooling is negative, building heating is positive
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.times = np.empty((0,), dtype=np.float64)
        self.loading = None
        self.generate_g_functions_for_all_ghe()

    def as_dict(self) -> dict:
        output = {
            "title": f"GHEDesigner GHE Output - Version {VERSION}",
            "number_of_boreholes": len(self.gFunction.bore_locations),
            "borehole_depth": {"value": self.bhe.b.H, "units": "m"},
            "borehole_spacing": {"value": self.B_spacing, "units": "m"},
            "borehole_heat_exchanger": self.bhe.as_dict(),
            "equivalent_borehole_heat_exchanger": self.bhe_eq.as_dict(),
            "simulation_parameters": self.sim_params.as_dict(),
        }
        return output

    @staticmethod
    def combine_sts_lts(log_time_lts: list, g_lts: list, log_time_sts: list, g_sts: list) -> interp1d:
        # make sure the short time step doesn't overlap with the long time step
        max_log_time_sts = max(log_time_sts)
        min_log_time_lts = min(log_time_lts)

        if max_log_time_sts < min_log_time_lts:
            log_time = log_time_sts + log_time_lts
            g = g_sts + g_lts
        else:
            # find where to stop in sts
            i = 0
            value = log_time_sts[i]
            while value <= min_log_time_lts:
                i += 1
                value = log_time_sts[i]
            log_time = log_time_sts[0:i] + log_time_lts
            g = g_sts[0:i] + g_lts
        g = interp1d(log_time, g)

        return g

    """
    def grab_g_function(self, b_over_h):
        # interpolate for the Long time step g-function
        g_function, rb_value, _, _ = self.gFunction.g_function_interpolation(b_over_h)
        # correct the long time step for borehole radius
        g_function_corrected = self.gFunction.borehole_radius_correction(g_function, rb_value, fse.b.r_b)
        # Don't Update the HybridLoad (its dependent on the STS) because
        # it doesn't change the results much, and it slows things down a lot
        # combine the short and long time step g-function
        g = self.combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.bhe_eq.lntts.tolist(),
            self.bhe_eq.g.tolist(),
        )

        g_bhw = self.combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.bhe_eq.lntts.tolist(),
            self.bhe_eq.g_bhw.tolist(),
        )

        return g, g_bhw
    """
    # I have made changes in grab_g_function so that I chan generate 3 different gfunctions for my 3 different GHEs
    def grab_g_function(self, g_function_obj, b_over_h, bhe_eq):
        # Interpolate for the Long Time Step (LTS) g-function
        g_function, rb_value, _, _ = g_function_obj.g_function_interpolation(b_over_h)

        # Correct the g-function for borehole radius
        g_function_corrected = g_function_obj.borehole_radius_correction(
            g_function, rb_value, bhe_eq.b.r_b
        )

        # Combine short and long time step g-functions
        g = self.combine_sts_lts(
            self.log_time,
            g_function_corrected,
            bhe_eq.lntts.tolist(),
            bhe_eq.g.tolist(),
        )

        g_bhw = self.combine_sts_lts(
            self.log_time,
            g_function_corrected,
            bhe_eq.lntts.tolist(),
            bhe_eq.g_bhw.tolist(),
        )

        return g, g_bhw

    def cost(self, max_eft, min_eft):
        delta_t_max = max_eft - self.sim_params.max_EFT_allowable
        delta_t_min = self.sim_params.min_EFT_allowable - min_eft
        t_excess = max(delta_t_max, delta_t_min)
        return t_excess

# My part of code for multiple GHE systems begins here

    def generate_g_functions_for_all_ghe(self):
        h_values = [100.0]  # fixed h value
        r_b1 = self.bhe1.calc_effective_borehole_resistance()
        r_b2 = self.bhe2.calc_effective_borehole_resistance()
        r_b3 = self.bhe3.calc_effective_borehole_resistance()
        depth = self.bhe1.b.D
        depth = self.bhe1.b.D
        b = self.B_spacing
        log_time = self.log_time

        # Generate coordinates based on n_rows and n_cols for each GHE
        coordinates_ghe1 = [(i * b, j * b) for i in range(n_rows1) for j in range(n_cols1)]
        coordinates_ghe2 = [(i * b, j * b) for i in range(n_rows2) for j in range(n_cols2)]
        coordinates_ghe3 = [(i * b, j * b) for i in range(n_rows3) for j in range(n_cols3)]

        # Generate separate g-functions
        self.gFunction1 = calc_g_func_for_multiple_lengths(
            b, h_values, r_b1, depth,
            self.m_ghe1_borehole, self.bhe_type, log_time, coordinates_ghe1, self.bhe1.fluid, self.bhe1.pipe,
            self.bhe1.grout, self.bhe1.soil
        )
        self.gFunction2 = calc_g_func_for_multiple_lengths(
            b, h_values, r_b2, depth,
            self.m_ghe2_borehole, self.bhe_type, log_time, coordinates_ghe2, self.bhe2.fluid, self.bhe2.pipe,
            self.bhe2.grout, self.bhe2.soil
        )
        self.gFunction3 = calc_g_func_for_multiple_lengths(
            b, h_values, r_b3, depth,
            self.m_ghe3_borehole, self.bhe_type, log_time, coordinates_ghe3, self.bhe3.fluid, self.bhe3.pipe,
            self.bhe3.grout, self.bhe3.soil
        )
        """
        # checking to see whether g-functions are calculated correctly
        for h_val, g_values in self.gFunction3.g_lts.items():
            print("📈 G-function values for GHE3:")
            for ln_tts, g in zip(self.gFunction3.log_time, g_values):
                print(f"ln(t/ts) = {ln_tts:.4f}, g = {g:.4f}")
        """
    def calculation_of_ghe_constant_c_n(self, g1: interp1d, g2: interp1d, g3: interp1d):
        """Calculate constant C_n for two Ground Heat Exchangers (GHE1 and GHE2).

        Formula: Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        Each GHE has its unique K_s and R_b.
        """
        ts = self.bhe1_eq.t_s  # (-)
        two_pi_k = TWO_PI * self.bhe1.soil.k

        # **GHE1 Properties**
        rb1 = self.bhe1.calc_effective_borehole_resistance()  # (m.K/W)

        # **GHE2 Properties**
        rb2 = self.bhe2.calc_effective_borehole_resistance()  # (m.K/W)

        # **GHE3 Properties**
        rb3 = self.bhe3.calc_effective_borehole_resistance()  # (m.K/W)
        # Initialize arrays for C_n values of GHE1 and GHE2
        c_n1 = np.zeros(n, dtype=float)
        c_n2 = np.zeros(n, dtype=float)
        c_n3 = np.zeros(n, dtype=float)

        for i in range(1, n):
            d_less_time = np.log((time_array[i] - time_array[i - 1]) / (ts / 3600))
            g_values1 = g1(d_less_time)
            g_values2 = g2(d_less_time)
            g_values3 = g3(d_less_time)

            # Compute C_n for each GHE
            c_n1[i] = (1 / two_pi_k * g_values1) + rb1
            c_n2[i] = (1 / two_pi_k * g_values2) + rb2
            c_n3[i] = (1 / two_pi_k * g_values3) + rb3

        return c_n1, c_n2, c_n3  # Return C_n values for GHEs

    # I am commenting def_simulation_detailed function and adding my own for multiple GHE-HP systems
    """
    def _simulate_detailed(self, q_dot: np.ndarray, time_values: np.ndarray, g: interp1d):
        # Perform a detailed simulation based on a numpy array of heat rejection
        # rates, Q_dot (Watts) where each load is applied at the time_value
        # (seconds). The g-function can interpolate.
        # Source: Chapter 2 of Advances in Ground Source Heat Pumps

        n = q_dot.size

        # Convert the total load applied to the field to the average over
        # borehole wall rejection rate
        # At time t=0, make the heat rejection rate 0.
        q_dot_b = np.hstack((0.0, q_dot / float(self.nbh)))
        time_values = np.hstack((0.0, time_values))

        q_dot_b_dt = np.hstack(q_dot_b[1:] - q_dot_b[:-1])

        ts = self.bhe_eq.t_s  # (-)
        two_pi_k = TWO_PI * self.bhe.soil.k  # (W/m.K)
        h = self.bhe.b.H  # (meters)
        tg = self.bhe.soil.ugt  # (Celsius)
        rb = self.bhe.calc_effective_borehole_resistance()  # (m.K/W)
        m_dot = self.bhe.m_flow_borehole  # (kg/s)
        cp = self.bhe.fluid.cp  # (J/kg.s)

        hp_eft: list[float] = []
        delta_tb: list[float] = []
        for i in range(1, n + 1):
            # Take the last i elements of the reversed time array
            _time = time_values[i] - time_values[0:i]
            # _time = time_values_reversed[n - i:n]
            g_values = g(np.log((_time * SEC_IN_HR) / ts))
            # Tb = Tg + (q_dt * g)  (Equation 2.12)
            delta_tb_i = (q_dot_b_dt[0:i] / h / two_pi_k).dot(g_values)
            # Tf = Tb + q_i * R_b^* (Equation 2.13)
            tb = tg + delta_tb_i
            # Bulk fluid temperature
            tf_bulk = tb + q_dot_b[i] / h * rb
            # T_out = T_f - Q / (2 * m_dot cp)  (Equation 2.14)
            tf_out = tf_bulk - q_dot_b[i] / (2 * m_dot * cp)
            hp_eft.append(tf_out)
            delta_tb.append(delta_tb_i)

        return hp_eft, delta_tb
    """

    def _simulate_detailed(self, g: interp1d):

        # My part of code for multiple GHE systems begins here
        if selected_configuration == "3ghe-6hp":

            ts = self.bhe1_eq.t_s  # (-)
            tg = self.bhe1.soil.ugt  # (Celsius)
            two_pi_k = TWO_PI * self.bhe1.soil.k  # (W/m.K)

            # These are constant values for capacity which has equation: c1*EFT^2 + c2*EFT + c3

            # Read the file for Heat Pump Coefficients
            file_path_hp_coefficients = repo_root / "HeatPumpData" / "HeatPumpCoefficients.csv"
            df = pd.read_csv(file_path_hp_coefficients)
            hp_coefficients = df.set_index("parameter")

            def get_coeffs(param, df):
                """Returns a tuple of values for hp1 through hp6 for the given parameter."""
                return tuple(df.loc[param, f"hp{i}"] for i in range(1, 7))

            c1_hp1_htg, c1_hp2_htg, c1_hp3_htg, c1_hp4_htg, c1_hp5_htg, c1_hp6_htg = get_coeffs("c1_htg",
                                                                                                hp_coefficients)
            c2_hp1_htg, c2_hp2_htg, c2_hp3_htg, c2_hp4_htg, c2_hp5_htg, c2_hp6_htg = get_coeffs("c2_htg",
                                                                                                hp_coefficients)
            c3_hp1_htg, c3_hp2_htg, c3_hp3_htg, c3_hp4_htg, c3_hp5_htg, c3_hp6_htg = get_coeffs("c3_htg",
                                                                                                hp_coefficients)

            c1_hp1_clg, c1_hp2_clg, c1_hp3_clg, c1_hp4_clg, c1_hp5_clg, c1_hp6_clg = get_coeffs("c1_clg",
                                                                                                hp_coefficients)
            c2_hp1_clg, c2_hp2_clg, c2_hp3_clg, c2_hp4_clg, c2_hp5_clg, c2_hp6_clg = get_coeffs("c2_clg",
                                                                                                hp_coefficients)
            c3_hp1_clg, c3_hp2_clg, c3_hp3_clg, c3_hp4_clg, c3_hp5_clg, c3_hp6_clg = get_coeffs("c3_clg",
                                                                                                hp_coefficients)

            a_htg_hp1, a_htg_hp2, a_htg_hp3, a_htg_hp4, a_htg_hp5, a_htg_hp6 = get_coeffs("a_htg", hp_coefficients)
            b_htg_hp1, b_htg_hp2, b_htg_hp3, b_htg_hp4, b_htg_hp5, b_htg_hp6 = get_coeffs("b_htg", hp_coefficients)
            c_htg_hp1, c_htg_hp2, c_htg_hp3, c_htg_hp4, c_htg_hp5, c_htg_hp6 = get_coeffs("c_htg", hp_coefficients)

            a_clg_hp1, a_clg_hp2, a_clg_hp3, a_clg_hp4, a_clg_hp5, a_clg_hp6 = get_coeffs("a_clg", hp_coefficients)
            b_clg_hp1, b_clg_hp2, b_clg_hp3, b_clg_hp4, b_clg_hp5, b_clg_hp6 = get_coeffs("b_clg", hp_coefficients)
            c_clg_hp1, c_clg_hp2, c_clg_hp3, c_clg_hp4, c_clg_hp5, c_clg_hp6 = get_coeffs("c_clg", hp_coefficients)

            nbh_ghe1, nbh_ghe2, nbh_ghe3 = self.nbh1, self.nbh2, self.nbh3
            nbh_total = nbh_ghe1 + nbh_ghe2 + nbh_ghe3
            cp = self.bhe1.fluid.cp  # (J/kg.s)
            rho_fluid = self.bhe1.fluid.rho

            b_over_h1 = self.B_spacing / self.bhe1.b.H
            b_over_h2 = self.B_spacing / self.bhe2.b.H
            b_over_h3 = self.B_spacing / self.bhe3.b.H

            g1, _ = self.grab_g_function(self.gFunction1, b_over_h1, self.bhe1_eq)
            g2, _ = self.grab_g_function(self.gFunction2, b_over_h2, self.bhe2_eq)
            g3, _ = self.grab_g_function(self.gFunction3, b_over_h3, self.bhe3_eq)

            c_n_ghe1, c_n_ghe2, c_n_ghe3 = self.calculation_of_ghe_constant_c_n(g1, g2, g3)

            # Initializing the values
            q_ghe1, q_ghe2, q_ghe3 = np.zeros(n), np.zeros(n), np.zeros(n)
            H_n_ghe1, H_n_ghe2, H_n_ghe3 = np.zeros(n), np.zeros(n), np.zeros(n)
            total_values_ghe1, total_values_ghe2, total_values_ghe3 = np.zeros(n), np.zeros(n), np.zeros(n)

            # Initializing mass flow rates
            (m_hp1_array, m_hp2_array, m_hp3_array, m_hp4_array, m_hp5_array, m_hp6_array, m_loop_array, m_ghe1_array,
             m_ghe2_array, m_ghe3_array) = (np.zeros(n) for _ in range(10))

            # Initializing temperatures
            t = np.full((n, 15), tg)
            t[0, :] = tg  # T1 through T_ghe2_ext

            # Calculating run_time_fraction

            q_net_htg_hp1 = df1.loc[:, (1, 1, 1, "HPHtgLd_W")] - df1.loc[:, (1, 1, 1, "HPClgLd_W")].to_numpy()
            q_net_htg_hp2 = df1.loc[:, (1, 2, 2, "HPHtgLd_W")].to_numpy()
            q_net_htg_hp3 = df1.loc[:, (2, 1, 3, "HPHtgLd_W")] - df1.loc[:, (2, 1, 3, "HPClgLd_W")].to_numpy()
            q_net_htg_hp4 = df1.loc[:, (2, 2, 4, "HPHtgLd_W")].to_numpy()
            q_net_htg_hp5 = df1.loc[:, (3, 1, 5, "HPHtgLd_W")] - df1.loc[:, (3, 1, 5, "HPClgLd_W")].to_numpy()
            q_net_htg_hp6 = df1.loc[:, (3, 2, 6, "HPHtgLd_W")].to_numpy()

            results = []

            # Special case for the first iteration
            total_values_ghe1[0], total_values_ghe2[0], total_values_ghe3[0] = (0, 0, 0)
            H_n_ghe1[0], H_n_ghe2[0], H_n_ghe3[0] = (tg, tg, tg)
            q_ghe2[0], q_ghe2[0], q_ghe3[0] = (0, 0, 0)

            labels = [
                "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "Tf_ghe1", "Tf_ghe2", "Tf_ghe3",
                "T_ghe1_ext", "T_ghe2_ext", "T_ghe3_ext", "q_ghe1", "q_ghe2", "q_ghe3"
            ]
            units = ["°C"] * 16 + ["W/m", "W/m", "W/m"]

            for i in range(1, n):
                if (np.isclose(q_net_htg_hp1[i], 0) and np.isclose(q_net_htg_hp2[i], 0)
                        and np.isclose(q_net_htg_hp3[i], 0) and np.isclose(q_net_htg_hp4[i], 0)
                        and np.isclose(q_net_htg_hp5[i], 0) and np.isclose(q_net_htg_hp6[i], 0)):
                    t[i, :] = t[i - 1, :]
                    q_ghe1[i], q_ghe2[i], q_ghe3[i] = (0, 0, 0)

                h1 = df1.loc[i, (1, 1, 1, "HPHtgLd_W")]
                c1 = df1.loc[i, (1, 1, 1, "HPClgLd_W")]
                h2 = df1.loc[i, (1, 2, 2, "HPHtgLd_W")]
                c2 = 0
                h3 = df1.loc[i, (2, 1, 3, "HPHtgLd_W")]
                c3 = df1.loc[i, (2, 1, 3, "HPClgLd_W")]
                h4 = df1.loc[i, (2, 2, 4, "HPHtgLd_W")]
                c4 = 0
                h5 = df1.loc[i, (3, 1, 5, "HPHtgLd_W")]
                c5 = df1.loc[i, (3, 1, 5, "HPClgLd_W")]
                h6 = df1.loc[i, (3, 2, 6, "HPHtgLd_W")]
                c6 = 0

                t_eft_hp1, t_eft_hp2, t_eft_hp3, t_eft_hp4, t_eft_hp5, t_eft_hp6 = t[i - 1, 1:7]

                # Calculating values for heat pump constants r1 and r2 for all heat pumps

                t_eft = np.array([t_eft_hp1, t_eft_hp2, t_eft_hp3, t_eft_hp4, t_eft_hp5, t_eft_hp6])
                h = np.array([h1, h2, h3, h4, h5, h6])
                c = np.array([c1, c2, c3, c4, c5, c6])

                a_htg = np.array([a_htg_hp1, a_htg_hp2, a_htg_hp3, a_htg_hp4, a_htg_hp5, a_htg_hp6])
                b_htg = np.array([b_htg_hp1, b_htg_hp2, b_htg_hp3, b_htg_hp4, b_htg_hp5, b_htg_hp6])
                c_htg = np.array([c_htg_hp1, c_htg_hp2, c_htg_hp3, c_htg_hp4, c_htg_hp5, c_htg_hp6])

                a_clg = np.array([a_clg_hp1, a_clg_hp2, a_clg_hp3, a_clg_hp4, a_clg_hp5, a_clg_hp6])
                b_clg = np.array([b_clg_hp1, b_clg_hp2, b_clg_hp3, b_clg_hp4, b_clg_hp5, b_clg_hp6])
                c_clg = np.array([c_clg_hp1, c_clg_hp2, c_clg_hp3, c_clg_hp4, c_clg_hp5, c_clg_hp6])

                # Heating calculations
                slope_htg = 2 * a_htg * t_eft + b_htg
                ratio_htg = a_htg * t_eft ** 2 + b_htg * t_eft + c_htg
                v = slope_htg
                u = ratio_htg - slope_htg * t_eft

                # Cooling calculations
                slope_clg = 2 * a_clg * t_eft + b_clg
                ratio_clg = a_clg * t_eft ** 2 + b_clg * t_eft + c_clg
                b = slope_clg
                a = ratio_clg - slope_clg * t_eft

                # Final results
                r1 = v * h - b * c
                r2 = u * h - a * c

                # Unpacking the values for r1 and r2 for use in the matrix solver
                r1_hp1, r1_hp2, r1_hp3, r1_hp4, r1_hp5, r1_hp6 = r1
                r2_hp1, r2_hp2, r2_hp3, r2_hp4, r2_hp5, r2_hp6 = r2

                # Calculating heat pump capacity

                # EFTs
                t_eft = np.array([t_eft_hp1, t_eft_hp2, t_eft_hp3, t_eft_hp4, t_eft_hp5, t_eft_hp6])

                # Heating and cooling coefficients
                c1_htg = np.array([c1_hp1_htg, c1_hp2_htg, c1_hp3_htg, c1_hp4_htg, c1_hp5_htg, c1_hp6_htg])
                c2_htg = np.array([c2_hp1_htg, c2_hp2_htg, c2_hp3_htg, c2_hp4_htg, c2_hp5_htg, c2_hp6_htg])
                c3_htg = np.array([c3_hp1_htg, c3_hp2_htg, c3_hp3_htg, c3_hp4_htg, c3_hp5_htg, c3_hp6_htg])

                c1_clg = np.array([c1_hp1_clg, c1_hp2_clg, c1_hp3_clg, c1_hp4_clg, c1_hp5_clg, c1_hp6_clg])
                c2_clg = np.array([c2_hp1_clg, c2_hp2_clg, c2_hp3_clg, c2_hp4_clg, c2_hp5_clg, c2_hp6_clg])
                c3_clg = np.array([c3_hp1_clg, c3_hp2_clg, c3_hp3_clg, c3_hp4_clg, c3_hp5_clg, c3_hp6_clg])

                # Flow rate scaling factors

                m_design = np.array([m_hp1_design, m_hp2_design, m_hp3_design, m_hp4_design, m_hp5_design,
                                     m_hp6_design])
                m_single = np.array([m_hp1_single_as, m_hp_single_wtw, m_hp2_single_as, m_hp_single_wtw,
                                     m_hp3_single_as, m_hp_single_wtw])

                # Heating load conditions at time i
                q_net_htg = np.array([q_net_htg_hp1[i], q_net_htg_hp2[i], q_net_htg_hp3[i], q_net_htg_hp4[i],
                                      q_net_htg_hp5[i], q_net_htg_hp6[i]])

                capacity_htg = c1_htg * t_eft ** 2 + c2_htg * t_eft + c3_htg
                capacity_clg = c1_clg * t_eft ** 2 + c2_clg * t_eft + c3_clg
                hp_capacity = np.where(q_net_htg > 0, capacity_htg, capacity_clg) * (m_design / m_single)

                # Unpacking the hp capacity for use later
                hp1_capacity, hp2_capacity, hp3_capacity, hp4_capacity, hp5_capacity, hp6_capacity = hp_capacity

                # I am introducing the scaling factor here (m_hp3_design/m_hp_single) because loads taken are for
                # whole building but heat pump used is only one. Obviously the heat pump is not able to handle all
                # the building loads. To tackle this I scaled up mass flow rate by ratio of htg/clg_loads to hp
                # capacity. To be consistent in this assumption, I need to scale up hp capacity also by the same factor
                # or else  I will always get rtf greater than 1.

                rtf_hp1 = min(abs(q_net_htg_hp1[i]) / abs(hp1_capacity), 1)
                rtf_hp2 = min(abs(q_net_htg_hp2[i]) / abs(hp2_capacity), 1)
                rtf_hp3 = min(abs(q_net_htg_hp3[i]) / abs(hp3_capacity), 1)
                rtf_hp4 = min(abs(q_net_htg_hp3[i]) / abs(hp4_capacity), 1)
                rtf_hp5 = min(abs(q_net_htg_hp3[i]) / abs(hp5_capacity), 1)
                rtf_hp6 = min(abs(q_net_htg_hp3[i]) / abs(hp6_capacity), 1)

                # Calculation of mass flow rates

                m_hp1 = m_hp1_design * rtf_hp1
                m_hp2 = m_hp2_design * rtf_hp2
                m_hp3 = m_hp3_design * rtf_hp3
                m_hp4 = m_hp4_design * rtf_hp4
                m_hp5 = m_hp5_design * rtf_hp5
                m_hp6 = m_hp6_design * rtf_hp6
                beta = 1.5
                m_loop = beta * (m_hp1 + m_hp2 + m_hp3 + m_hp4 + m_hp5 + m_hp6)
                m_ghe1 = m_loop * nbh_ghe1 / nbh_total
                m_ghe2 = m_loop * nbh_ghe2 / nbh_total
                m_ghe3 = m_loop * nbh_ghe3 / nbh_total

                # Storing these values
                values = [m_hp1, m_hp2, m_hp3, m_hp1, m_hp2, m_hp3, m_loop, m_ghe1, m_ghe2, m_ghe3]
                arrays = [m_hp1_array, m_hp2_array, m_hp3_array, m_hp4_array, m_hp5_array, m_hp6_array,
                          m_loop_array, m_ghe1_array, m_ghe2_array, m_ghe3_array]

                for arr, val in zip(arrays, values):
                    arr[i] = val
                time_n = time_array[i]

                # Compute dimensionless time for all indices from 1 to i-1
                indices = np.arange(1, i)
                dim_less_time = np.log((time_n - time_array[indices - 1]) / (ts / 3600))

                # Compute contributions from all previous steps in one go
                delta_q_ghe1 = (q_ghe1[indices] - q_ghe1[indices - 1]) / two_pi_k
                delta_q_ghe2 = (q_ghe2[indices] - q_ghe2[indices - 1]) / two_pi_k
                delta_q_ghe3 = (q_ghe2[indices] - q_ghe2[indices - 1]) / two_pi_k

                values_ghe1 = np.sum(delta_q_ghe1 * g(dim_less_time))
                values_ghe2 = np.sum(delta_q_ghe2 * g(dim_less_time))
                values_ghe3 = np.sum(delta_q_ghe3 * g(dim_less_time))

                total_values_ghe1[i] = values_ghe1
                total_values_ghe2[i] = values_ghe2
                total_values_ghe3[i] = values_ghe3

                # Dimensionless time for the immediate previous step
                dim1_less_time = np.log((time_n - time_array[i - 1]) / (ts / 3600))
                H_n_ghe1[i] = tg - total_values_ghe1[i] + (q_ghe1[i - 1] / two_pi_k * g(dim1_less_time))
                H_n_ghe2[i] = tg - total_values_ghe2[i] + (q_ghe2[i - 1] / two_pi_k * g(dim1_less_time))
                H_n_ghe3[i] = tg - total_values_ghe3[i] + (q_ghe3[i - 1] / two_pi_k * g(dim1_less_time))

                A = np.array([
                    [1 - (r1_hp1 / (m_loop * cp)), -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 - (r1_hp2 / (m_loop * cp)), -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1 - (r1_hp3 / (m_loop * cp)), -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1 - (r1_hp4 / (m_loop * cp)), -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1 - (r1_hp5 / (m_loop * cp)), -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1 - (r1_hp6 / (m_loop * cp)), -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                    [0, 0, 0, 0, 0, 0, (m_loop - m_ghe1) * cp, -m_loop * cp, 0, 0, 0, 0, m_ghe1 * cp, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, c_n_ghe1[i], 0, 0],
                    [0, 0, 0, 0, 0, 0, -1, 0, 0, 2, 0, 0, -1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, m_ghe1 * cp, 0, 0, 0, 0, 0, -m_ghe1 * cp, 0, 0, self.bhe1.b.H * nbh_ghe1, 0, 0],

                    [0, 0, 0, 0, 0, 0, 0, (m_loop - m_ghe2) * cp, -m_loop * cp, 0, 0, 0, 0, m_ghe2 * cp, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, c_n_ghe2[i], 0],
                    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, m_ghe2 * cp, 0, 0, 0, 0, 0, -m_ghe2 * cp, 0, 0, self.bhe2.b.H * nbh_ghe2, 0],

                    [-m_loop * cp, 0, 0, 0, 0, 0, 0, 0, (m_loop - m_ghe3) * cp, 0, 0, 0, 0, 0, m_ghe3 * cp, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, c_n_ghe3[i]],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, m_ghe3 * cp, 0, 0, 0, 0, 0, -m_ghe3 * cp, 0, 0, self.bhe3.b.H * nbh_ghe3]
                ])

                B = np.array([r2_hp1 / (m_loop * cp), r2_hp2 / (m_loop * cp), r2_hp3 / (m_loop * cp),
                              r2_hp4 / (m_loop * cp), r2_hp5 / (m_loop * cp), r2_hp6 / (m_loop * cp),
                              0, H_n_ghe1[i], 0, 0, 0, H_n_ghe2[i], 0, 0, 0, H_n_ghe3[i], 0, 0])

                X = np.linalg.solve(A, B)

                t[i, :] = X[:15]  # Store 15 temperature values for the current time step
                q_ghe1[i] = X[15]  # Store heat flux for GHE1
                q_ghe2[i] = X[16]  # Store heat flux for GHE2
                q_ghe2[i] = X[17]  # Store heat flux for GHE2

                # Save X values along with the current time step
                row = {"Time Step": time_array[i]}
                row.update({label: value for label, value in zip(labels, X)})
                results.append(row)

            # Create DataFrame and insert units as second row
            df = pd.DataFrame(results)
            unit_row = {"Time Step": "Units"}
            unit_row.update(dict(zip(labels, units)))

            # Insert units row at the top
            df_with_units = pd.concat([pd.DataFrame([unit_row]), df], ignore_index=True)

            # Save to CSV
            df_with_units.to_csv("detailed_simulation_results.csv", index=False)

        else:
            print("Configuration not recognized")

        # My part of code for multiple GHE systems ends here

    # this code now becomes redundant as three gfunctions are calculated separately, so I am commenting it out.
    """
    def compute_g_functions(self):
        # Compute g-functions for a bracketed solution, based on min and max
        # height
        min_height = self.sim_params.min_height
        max_height = self.sim_params.max_height
        avg_height = (min_height + max_height) / 2.0
        h_values = [min_height, avg_height, max_height]

        coordinates = self.gFunction.bore_locations
        log_time = self.gFunction.log_time

        g_function = calc_g_func_for_multiple_lengths(
            self.B_spacing,
            h_values,
            self.bhe.b.r_b,
            self.bhe.b.D,
            self.bhe.m_flow_borehole,
            self.bhe_type,
            log_time,
            coordinates,
            self.bhe.fluid,
            self.bhe.pipe,
            self.bhe.grout,
            self.bhe.soil,
        )

        self.gFunction = g_function
    """

class GHE(BaseGHE):
    def __init__(
        self,
        v_flow_system: float,
        b_spacing: float,
        bhe_type: BHPipeType,
        fluid,
        borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        #g_function: GFunction,
        sim_params: SimulationParameters,
        hourly_extraction_ground_loads: list,
        field_type="N/A",
        field_specifier="N/A",
        load_years=None,
    ) -> None:
        BaseGHE.__init__(
            self,
            v_flow_system,
            b_spacing,
            bhe_type,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            #g_function,
            sim_params,
            hourly_extraction_ground_loads,
            field_type=field_type,
            field_specifier=field_specifier,
        )

        # Split the extraction loads into heating and cooling for input to
        # the HybridLoad object
        if load_years is None:
            load_years = [2019]

        #hybrid_load = HybridLoad(
            #self.hourly_extraction_ground_loads, self.bhe_eq, self.bhe_eq, sim_params, years=load_years
        #)

        # hybrid load object
        #self.hybrid_load = hybrid_load

        # List of heat pump exiting fluid temperatures
        self.hp_eft: list[float] = []
        # list of change in borehole wall temperatures
        self.dTb: list[float] = []

    # def as_dict(self) -> dict:
    #     output = {
    #         "base": super().as_dict(),
    #     }
    #
    #     results = {}
    #     if len(self.hp_eft) > 0:
    #         max_hp_eft = max(self.hp_eft)
    #         min_hp_eft = min(self.hp_eft)
    #         results["max_hp_entering_temp"] = {"value": max_hp_eft, "units": "C"}
    #         results["min_hp_entering_temp"] = {"value": min_hp_eft, "units": "C"}
    #         t_excess = self.cost(max_hp_eft, min_hp_eft)
    #         results["excess_fluid_temperature"] = {"value": t_excess, "units": "C"}
    #     results["peak_load_analysis"] = self.hybrid_load.as_dict()
    #
    #     g_function = {
    #         "coordinates (x[m], y[m])": list(self.gFunction.bore_locations),
    #     }
    #     b_over_h = self.B_spacing / self.bhe.b.H
    #     g, _ = self.grab_g_function(b_over_h)
    #     total_g_values = g.x.size
    #     number_lts_g_values = 27
    #     number_sts_g_values = 50
    #     sts_step_size = floor((total_g_values - number_lts_g_values) / number_sts_g_values)
    #     lntts = []
    #     g_values = []
    #     for idx in range(0, (total_g_values - number_lts_g_values), sts_step_size):
    #         lntts.append(g.x[idx].tolist())
    #         g_values.append(g.y[idx].tolist())
    #     lntts += g.x[total_g_values - number_lts_g_values : total_g_values].tolist()
    #     g_values += g.y[total_g_values - number_lts_g_values : total_g_values].tolist()
    #     pairs = zip(lntts, g_values)
    #     for lntts_val, g_val in pairs:
    #         # TODO why is this attempting to append a string to a dictionary??
    #         output += f"{lntts_val:0.4f}\t{g_val:0.4f}"
    #     g_function["lntts, g"] = [*pairs]
    #
    #     results["g_function_information"] = g_function
    #     output["simulation_results"] = results
    #
    #     return output

    def simulate(self, method: TimestepType):
        b = self.B_spacing
        b_over_h = b / self.bhe.b.H

        # Solve for equivalent single U-tube
        self.bhe_eq = self.bhe.to_single()
        # Update short time step object with equivalent single u-tube
        self.bhe_eq.calc_sts_g_functions()
        # Combine the short and long-term g-functions. The long term g-function
        # is interpolated for specific B/H and rb/H values.
        g, _ = self.grab_g_function(b_over_h)

        if method == TimestepType.HYBRID:
            q_dot = self.hybrid_load.load[2:] * 1000.0  # convert to Watts
            time_values = self.hybrid_load.hour[2:]  # convert to seconds
            self.times = time_values
            self.loading = q_dot

            hp_eft, d_tb = self._simulate_detailed(q_dot, time_values, g)
        elif method == TimestepType.HOURLY:
            n_months = self.sim_params.end_month - self.sim_params.start_month + 1
            n_hours = int(n_months / 12.0 * 8760.0)
            q_dot = self.hourly_extraction_ground_loads
            # How many times does q need to be repeated?
            n_years = ceil(n_hours / 8760)
            if len(q_dot) // 8760 < n_years:
                q_dot = q_dot * n_years
            else:
                n_hours = len(q_dot)
            q_dot = -1.0 * np.array(q_dot)  # Convert loads to rejection
            # print("Times:",self.times)
            if len(self.times) == 0:
                self.times = np.arange(1, n_hours + 1, 1)
            t = self.times
            self.loading = q_dot

            hp_eft, d_tb = self._simulate_detailed(q_dot, t, g)
        else:
            raise ValueError("Only hybrid or hourly methods available.")

        self.hp_eft = hp_eft
        self.dTb = d_tb

        return max(hp_eft), min(hp_eft)

    def size(self, method: TimestepType) -> None:
        # Size the ground heat exchanger
        def local_objective(h):
            self.bhe.b.H = h
            max_hp_eft, min_hp_eft = self.simulate(method=method)
            t_excess = self.cost(max_hp_eft, min_hp_eft)
            return t_excess

        # Make the initial guess variable the average of the heights given
        self.bhe.b.H = (self.sim_params.max_height + self.sim_params.min_height) / 2.0
        # bhe.b.H is updated during sizing
        returned_height = solve_root(
            self.bhe.b.H,
            local_objective,
            lower=self.sim_params.min_height,
            upper=self.sim_params.max_height,
            abs_tol=1.0e-6,
            rel_tol=1.0e-6,
            max_iter=50,
        )

        self.bhe.b.H = returned_height

    def calculate(self, _hour_index: int, inlet_temp: float, _flow_rate: float) -> float:
        effectiveness = 0.5
        soil_temp = 20

        return effectiveness * (soil_temp - inlet_temp) + inlet_temp
