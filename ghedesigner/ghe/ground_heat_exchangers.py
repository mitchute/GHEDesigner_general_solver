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

# My addition for multiple GHE-HP begins here

import pandas as pd

# Reading the Heat Pump Loads file
file_path_hp_loads = "C:\\Users\\nbast\\Desktop\\HeatPumpLoads\\MultipleGHE-HP_heatpumploads_DULUTH.csv"

df1 = pd.read_csv(file_path_hp_loads, header=[0, 1, 2, 3])
df1.columns = pd.MultiIndex.from_tuples([(int(l1), int(l2), int(l3), l4) for l1, l2, l3, l4 in df1.columns])

# Reading the case file
file_path_cases = "C:\\Users\\nbast\\Desktop\\HeatPumpLoads\\cases.csv"

selected_configuration = "3ghe-6hp"
selected_case = "Case 6"


def read_hp_loads(hp_loads_path: str):
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

print(time_array)
print(n)


def read_case_data(cases_path: str, selected_case: str):
    """Reads selected case data and extracts parameters."""
    df_cases = pd.read_csv(cases_path)
    case_data = df_cases[df_cases["Case"] == selected_case].iloc[0]

    n_rows = int(case_data["n_rows"])
    n_cols = int(case_data["n_columns"])
    b_spacing = case_data["b_spacing"]
    m_hp1_design = case_data["m_hp1"]
    m_hp2_design = case_data["m_hp2"]
    m_hp3_design = case_data["m_hp3"]
    m_hp4_design = case_data["m_hp4"]
    m_hp5_design = case_data["m_hp5"]
    m_hp6_design = case_data["m_hp6"]
    nbh1 = case_data["nbh_1GHE"]
    nbh2 = case_data["nbh_2GHE"]
    nbh3 = case_data["nbh_3GHE"]
    m_hp1_single_wta = case_data["m_hp1_single_WTA"]
    m_hp2_singLe_wta = case_data["m_hp2_single_WTA"]
    m_hp3_single_wta = case_data["m_hp3_single_WTA"]
    m_hp_single_wtw = case_data["m_hp_single_WTW"]

    return (n_rows, n_cols, b_spacing, m_hp1_design, m_hp2_design, m_hp3_design, m_hp4_design, m_hp5_design,
            m_hp6_design, nbh1, nbh2, nbh3, m_hp1_single_wta, m_hp2_singLe_wta, m_hp3_single_wta, m_hp_single_wtw)


(n_rows, n_cols, b_spacing, m_hp1_design, m_hp2_design, m_hp3_design, m_hp4_design, m_hp5_design, m_hp6_design, nbh1,
 nbh2, nbh3, m_hp1_single_as, m_hp2_single_as, m_hp3_single_as, m_hp_single_wtw) = read_case_data(file_path_cases,
                                                                                                  selected_case)

# My addition for multiple GHE ends here. Rest of the changes are made in simulate detailed function.

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
        g_function: GFunction,
        sim_params: SimulationParameters,
        hourly_extraction_ground_loads: list,
        field_type="N/A",
        field_specifier="N/A",
    ) -> None:
        self.fieldType = field_type
        self.fieldSpecifier = field_specifier
        self.V_flow_system = v_flow_system
        self.B_spacing = b_spacing
        #self.nbh = len(g_function.bore_locations)
        self.nbh = nbh1
        self.V_flow_borehole = self.V_flow_system / self.nbh
        m_flow_borehole = self.V_flow_borehole / 1000.0 * fluid.rho
        self.m_flow_borehole = m_flow_borehole

        # Borehole Heat Exchanger
        self.bhe_type = bhe_type
        self.bhe = get_bhe_object(bhe_type, m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # Equivalent borehole Heat Exchanger
        self.bhe_eq = self.bhe.to_single()

        # Radial numerical short time step
        self.bhe_eq.calc_sts_g_functions()

        # gFunction object
        self.gFunction = g_function
        # Additional simulation parameters
        self.sim_params = sim_params
        # Hourly ground extraction loads
        # Building cooling is negative, building heating is positive
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.times = np.empty((0,), dtype=np.float64)
        self.loading = None

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

    def grab_g_function(self, b_over_h):
        # interpolate for the Long time step g-function
        g_function, rb_value, _, _ = self.gFunction.g_function_interpolation(b_over_h)
        # correct the long time step for borehole radius
        g_function_corrected = self.gFunction.borehole_radius_correction(g_function, rb_value, self.bhe.b.r_b)
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

    def cost(self, max_eft, min_eft):
        delta_t_max = max_eft - self.sim_params.max_EFT_allowable
        delta_t_min = self.sim_params.min_EFT_allowable - min_eft
        t_excess = max(delta_t_max, delta_t_min)
        return t_excess

# My part of code for multiple GHE systems begins here

    def calculation_of_ghe_constant_c_n(self, g: interp1d):
        """Calculate constant C_n for two Ground Heat Exchangers (GHE1 and GHE2).

        Formula: Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        Each GHE has its unique K_s and R_b.
        """
        ts = self.bhe_eq.t_s  # (-)
        two_pi_k = TWO_PI * self.bhe.soil.k

        # **GHE1 Properties**
        rb1 = self.bhe.calc_effective_borehole_resistance()  # (m.K/W)

        # **GHE2 Properties**
        rb2 = self.bhe.calc_effective_borehole_resistance()  # (m.K/W)

        # **GHE3 Properties**
        rb3 = self.bhe.calc_effective_borehole_resistance()  # (m.K/W)

        # Initialize arrays for C_n values of GHE1 and GHE2
        c_n1 = np.zeros(n, dtype=float)
        c_n2 = np.zeros(n, dtype=float)
        c_n3 = np.zeros(n, dtype=float)

        for i in range(1, n):
            d_less_time = np.log((time_array[i] - time_array[i - 1]) / (ts / 3600))
            g_values = g(d_less_time)

            # Compute C_n for each GHE
            c_n1[i] = (1 / two_pi_k * g_values) + rb1
            c_n2[i] = (1 / two_pi_k * g_values) + rb2
            c_n3[i] = (1 / two_pi_k * g_values) + rb3

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

    def _simulate_detailed(self, q_dot, time_values, g: interp1d):

        # My part of code for multiple GHE systems begins here
        if selected_configuration == "3ghe-6hp":

            ts = self.bhe_eq.t_s  # (-)
            tg = self.bhe.soil.ugt  # (Celsius)
            two_pi_k = TWO_PI * self.bhe.soil.k  # (W/m.K)

            # These are constant values for capacity which has equation: c1*EFT^2 + c2*EFT + c3

            c1_hp1_htg, c1_hp2_htg, c1_hp3_htg, c1_hp4_htg, c1_hp5_htg, c1_hp6_htg = (-0.124, -1.19, 0.178, -1.19,
                                                                                      0.0912, -1.19)
            c2_hp1_htg, c2_hp2_htg, c2_hp3_htg, c2_hp4_htg, c2_hp5_htg, c2_hp6_htg = (112, 212, 158, 212, 224, 212)
            c3_hp1_htg, c3_hp2_htg, c3_hp3_htg, c3_hp4_htg, c3_hp5_htg, c3_hp6_htg = (4114, 8932, 6712, 8932, 9422,
                                                                                      8932)

            c1_hp1_clg, c1_hp2_clg, c1_hp3_clg, c1_hp4_clg, c1_hp5_clg, c1_hp6_clg = (-0.393, 0, -1.2, 0, -2.16, 0)
            c2_hp1_clg, c2_hp2_clg, c2_hp3_clg, c2_hp4_clg, c2_hp5_clg, c2_hp6_clg = (-30.3, 0, -4.72, 0, 46.8, 0)
            c3_hp1_clg, c3_hp2_clg, c3_hp3_clg, c3_hp4_clg, c3_hp5_clg, c3_hp6_clg = (6777, 0, 9629, 0, 12397, 0)

            # These are constant values for ratio (q_ext/q_htg or q_rej/q_clg) a*EFT^2+b*EFT+c
            a_htg_hp1, a_htg_hp2, a_htg_hp3, a_htg_hp4, a_htg_hp5, a_htg_hp6 = (-0.0000713, -0.0000606, -0.0000786,
                                                                                -0.0000606, -0.000068, -0.0000606)
            b_htg_hp1, b_htg_hp2, b_htg_hp3, b_htg_hp4, b_htg_hp5, b_htg_hp6 = (0.00534, 0.0051, 0.00539, 0.0051,
                                                                                0.00497, 0.0051)
            c_htg_hp1, c_htg_hp2, c_htg_hp3, c_htg_hp4, c_htg_hp5, c_htg_hp6 = (0.734, 0.692, 0.734, 0.692, 0.738,
                                                                                0.692)

            a_clg_hp1, a_clg_hp2, a_clg_hp3, a_clg_hp4, a_clg_hp5, a_clg_hp6 = (0.000108, 0, 0.000145, 0, 0.000143, 0)

            b_clg_hp1, b_clg_hp2, b_clg_hp3, b_clg_hp4, b_clg_hp5, b_clg_hp6 = (0.00112, 0, -0.00041, 0, -0.000844, 0)

            c_clg_hp1, c_clg_hp2, c_clg_hp3, c_clg_hp4, c_clg_hp5, c_clg_hp6 = (1.13, 0, 1.15, 0, 1.16, 0)

            nbh_ghe1 = self.nbh
            nbh_ghe2 = self.nbh
            nbh_ghe3 = self.nbh
            nbh_total = nbh_ghe1 + nbh_ghe2 + nbh_ghe3
            cp = self.bhe.fluid.cp  # (J/kg.s)
            rho_fluid = self.bhe.fluid.rho

            h = self.bhe.b.H
            h_ghe1 = h_ghe2 = h_ghe3 = h  # (meters)  # (meters)

            c_n_ghe1, c_n_ghe2, c_n_ghe3 = self.calculation_of_ghe_constant_c_n(g)

            # Initializing the values
            q_ghe1 = np.zeros(n)
            q_ghe2 = np.zeros(n)
            q_ghe3 = np.zeros(n)
            H_n_ghe1 = np.zeros(n)
            H_n_ghe2 = np.zeros(n)
            H_n_ghe3 = np.zeros(n)
            total_values_ghe1 = np.zeros(n)
            total_values_ghe2 = np.zeros(n)
            total_values_ghe3 = np.zeros(n)

            # Initializing mass flow rates
            m_hp1_array = np.zeros(n)
            m_hp2_array = np.zeros(n)
            m_hp3_array = np.zeros(n)
            m_hp4_array = np.zeros(n)
            m_hp5_array = np.zeros(n)
            m_hp6_array = np.zeros(n)
            m_loop_array = np.zeros(n)
            m_ghe1_array = np.zeros(n)
            m_ghe2_array = np.zeros(n)
            m_ghe3_array = np.zeros(n)

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
            total_values_ghe1[0] = 0
            total_values_ghe2[0] = 0
            total_values_ghe3[0] = 0

            H_n_ghe1[0] = tg
            H_n_ghe2[0] = tg
            H_n_ghe3[0] = tg

            q_ghe1[0] = 0
            q_ghe2[0] = 0
            q_ghe3[0] = 0

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
                    q_ghe1[i] = 0
                    q_ghe2[i] = 0
                    q_ghe3[i] = 0

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

                t_eft_hp1 = t[i - 1, 1]
                t_eft_hp2 = t[i - 1, 2]
                t_eft_hp3 = t[i - 1, 3]
                t_eft_hp4 = t[i - 1, 4]
                t_eft_hp5 = t[i - 1, 5]
                t_eft_hp6 = t[i - 1, 6]

                # Calculating values for heat pump constants

                # for heat pump 1
                slope_htg_hp1 = 2 * a_htg_hp1 * t_eft_hp1 + b_htg_hp1
                ratio_htg_hp1 = a_htg_hp1 * t_eft_hp1 ** 2 + b_htg_hp1 * t_eft_hp1 + c_htg_hp1

                v_hp1 = slope_htg_hp1
                u_hp1 = ratio_htg_hp1 - slope_htg_hp1 * t_eft_hp1

                slope_clg_hp1 = 2 * a_clg_hp1 * t_eft_hp1 + b_clg_hp1
                ratio_clg_hp1 = a_clg_hp1 * t_eft_hp1 ** 2 + b_clg_hp1 * t_eft_hp1 + c_clg_hp1

                b_hp1 = slope_clg_hp1
                a_hp1 = ratio_clg_hp1 - slope_clg_hp1 * t_eft_hp1

                r1_hp1 = v_hp1 * h1 - b_hp1 * c1
                r2_hp1 = u_hp1 * h1 - a_hp1 * c1

                # Repeat for HP2 through HP6
                # HP2
                slope_htg_hp2 = 2 * a_htg_hp2 * t_eft_hp2 + b_htg_hp2
                ratio_htg_hp2 = a_htg_hp2 * t_eft_hp2 ** 2 + b_htg_hp2 * t_eft_hp2 + c_htg_hp2
                v_hp2 = slope_htg_hp2
                u_hp2 = ratio_htg_hp2 - slope_htg_hp2 * t_eft_hp2

                slope_clg_hp2 = 2 * a_clg_hp2 * t_eft_hp2 + b_clg_hp2
                ratio_clg_hp2 = a_clg_hp2 * t_eft_hp2 ** 2 + b_clg_hp2 * t_eft_hp2 + c_clg_hp2
                b_hp2 = slope_clg_hp2
                a_hp2 = ratio_clg_hp2 - slope_clg_hp2 * t_eft_hp2

                r1_hp2 = v_hp2 * h2 - b_hp2 * c2
                r2_hp2 = u_hp2 * h2 - a_hp2 * c2

                # HP3
                slope_htg_hp3 = 2 * a_htg_hp3 * t_eft_hp3 + b_htg_hp3
                ratio_htg_hp3 = a_htg_hp3 * t_eft_hp3 ** 2 + b_htg_hp3 * t_eft_hp3 + c_htg_hp3
                v_hp3 = slope_htg_hp3
                u_hp3 = ratio_htg_hp3 - slope_htg_hp3 * t_eft_hp3

                slope_clg_hp3 = 2 * a_clg_hp3 * t_eft_hp3 + b_clg_hp3
                ratio_clg_hp3 = a_clg_hp3 * t_eft_hp3 ** 2 + b_clg_hp3 * t_eft_hp3 + c_clg_hp3
                b_hp3 = slope_clg_hp3
                a_hp3 = ratio_clg_hp3 - slope_clg_hp3 * t_eft_hp3

                r1_hp3 = v_hp3 * h3 - b_hp3 * c3
                r2_hp3 = u_hp3 * h3 - a_hp3 * c3

                # HP4
                slope_htg_hp4 = 2 * a_htg_hp4 * t_eft_hp4 + b_htg_hp4
                ratio_htg_hp4 = a_htg_hp4 * t_eft_hp4 ** 2 + b_htg_hp4 * t_eft_hp4 + c_htg_hp4
                v_hp4 = slope_htg_hp4
                u_hp4 = ratio_htg_hp4 - slope_htg_hp4 * t_eft_hp4

                slope_clg_hp4 = 2 * a_clg_hp4 * t_eft_hp4 + b_clg_hp4
                ratio_clg_hp4 = a_clg_hp4 * t_eft_hp4 ** 2 + b_clg_hp4 * t_eft_hp4 + c_clg_hp4
                b_hp4 = slope_clg_hp4
                a_hp4 = ratio_clg_hp4 - slope_clg_hp4 * t_eft_hp4

                r1_hp4 = v_hp4 * h4 - b_hp4 * c4
                r2_hp4 = u_hp2 * h4 - a_hp2 * c4

                # HP5
                slope_htg_hp5 = 2 * a_htg_hp5 * t_eft_hp5 + b_htg_hp5
                ratio_htg_hp5 = a_htg_hp5 * t_eft_hp5 ** 2 + b_htg_hp5 * t_eft_hp5 + c_htg_hp5
                v_hp5 = slope_htg_hp5
                u_hp5 = ratio_htg_hp5 - slope_htg_hp5 * t_eft_hp5

                slope_clg_hp5 = 2 * a_clg_hp5 * t_eft_hp5 + b_clg_hp5
                ratio_clg_hp5 = a_clg_hp5 * t_eft_hp5 ** 2 + b_clg_hp5 * t_eft_hp5 + c_clg_hp5
                b_hp5 = slope_clg_hp5
                a_hp5 = ratio_clg_hp5 - slope_clg_hp5 * t_eft_hp5

                r1_hp5 = v_hp5 * h5 - b_hp5 * c5
                r2_hp5 = u_hp5 * h5 - a_hp5 * c5

                # HP6
                slope_htg_hp6 = 2 * a_htg_hp6 * t_eft_hp6 + b_htg_hp6
                ratio_htg_hp6 = a_htg_hp6 * t_eft_hp6 ** 2 + b_htg_hp6 * t_eft_hp6 + c_htg_hp6
                v_hp6 = slope_htg_hp6
                u_hp6 = ratio_htg_hp6 - slope_htg_hp6 * t_eft_hp6

                slope_clg_hp6 = 2 * a_clg_hp6 * t_eft_hp6 + b_clg_hp6
                ratio_clg_hp6 = a_clg_hp6 * t_eft_hp6 ** 2 + b_clg_hp6 * t_eft_hp6 + c_clg_hp6
                b_hp6 = slope_clg_hp6
                a_hp6 = ratio_clg_hp6 - slope_clg_hp6 * t_eft_hp6

                r1_hp6 = v_hp6 * h6 - b_hp6 * c6
                r2_hp6 = u_hp6 * h6 - a_hp6 * c6

                # Calculating heat pump capacity

                hp1_capacity = ((c1_hp1_htg * t_eft_hp1 ** 2 + c2_hp1_htg * t_eft_hp1 + c3_hp1_htg) if q_net_htg_hp1[
                                                                                                       i] > 0 else (
                    c1_hp1_clg * t_eft_hp1 ** 2 + c2_hp1_clg * t_eft_hp1 + c3_hp1_clg)) * (
                                       m_hp1_design / m_hp1_single_as)

                hp2_capacity = ((c1_hp2_htg * t_eft_hp2 ** 2 + c2_hp2_htg * t_eft_hp2 + c3_hp2_htg) if q_net_htg_hp2[
                                                                                                       i] > 0 else (
                    c1_hp2_clg * t_eft_hp2 ** 2 + c2_hp2_clg * t_eft_hp2 + c3_hp2_clg)) * (
                                       m_hp2_design / m_hp_single_wtw)

                hp3_capacity = ((c1_hp3_htg * t_eft_hp3 ** 2 + c2_hp3_htg * t_eft_hp3 + c3_hp3_htg) if q_net_htg_hp3[
                                                                                                       i] > 0 else (
                    c1_hp3_clg * t_eft_hp3 ** 2 + c2_hp3_clg * t_eft_hp3 + c3_hp3_clg)) * (
                                       m_hp3_design / m_hp2_single_as)

                hp4_capacity = ((c1_hp4_htg * t_eft_hp4 ** 2 + c2_hp4_htg * t_eft_hp4 + c3_hp4_htg) if q_net_htg_hp4[
                                                                                                       i] > 0 else (
                    c1_hp4_clg * t_eft_hp4 ** 2 + c2_hp4_clg * t_eft_hp4 + c3_hp4_clg)) * (
                                       m_hp4_design / m_hp_single_wtw)

                hp5_capacity = ((c1_hp5_htg * t_eft_hp5 ** 2 + c2_hp5_htg * t_eft_hp5 + c3_hp5_htg) if q_net_htg_hp5[
                                                                                                       i] > 0 else (
                    c1_hp5_clg * t_eft_hp5 ** 2 + c2_hp5_clg * t_eft_hp5 + c3_hp5_clg)) * (
                                       m_hp5_design / m_hp3_single_as)

                hp6_capacity = ((c1_hp6_htg * t_eft_hp6 ** 2 + c2_hp6_htg * t_eft_hp6 + c3_hp6_htg) if q_net_htg_hp6[
                                                                                                       i] > 0 else (
                    c1_hp6_clg * t_eft_hp6 ** 2 + c2_hp6_clg * t_eft_hp6 + c3_hp6_clg)) * (
                                       m_hp6_design / m_hp_single_wtw)

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
                m_hp1_array[i] = m_hp1
                m_hp2_array[i] = m_hp2
                m_hp3_array[i] = m_hp3
                m_hp4_array[i] = m_hp1
                m_hp5_array[i] = m_hp2
                m_hp6_array[i] = m_hp3
                m_loop_array[i] = m_loop
                m_ghe1_array[i] = m_ghe1
                m_ghe2_array[i] = m_ghe2
                m_ghe3_array[i] = m_ghe3
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
                    [0, 0, 0, 0, 0, 0, m_ghe1 * cp, 0, 0, 0, 0, 0, -m_ghe1 * cp, 0, 0, h_ghe1 * nbh_ghe1, 0, 0],

                    [0, 0, 0, 0, 0, 0, 0, (m_loop - m_ghe2) * cp, -m_loop * cp, 0, 0, 0, 0, m_ghe2 * cp, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, c_n_ghe2[i], 0],
                    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, m_ghe2 * cp, 0, 0, 0, 0, 0, -m_ghe2 * cp, 0, 0, h_ghe2 * nbh_ghe2, 0],

                    [-m_loop * cp, 0, 0, 0, 0, 0, 0, 0, (m_loop - m_ghe3) * cp, 0, 0, 0, 0, 0, m_ghe3 * cp, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, c_n_ghe3[i]],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, m_ghe3 * cp, 0, 0, 0, 0, 0, -m_ghe3 * cp, 0, 0, h_ghe3 * nbh_ghe3]
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
        g_function: GFunction,
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
            g_function,
            sim_params,
            hourly_extraction_ground_loads,
            field_type=field_type,
            field_specifier=field_specifier,
        )

        # Split the extraction loads into heating and cooling for input to
        # the HybridLoad object
        if load_years is None:
            load_years = [2019]

        hybrid_load = HybridLoad(
            self.hourly_extraction_ground_loads, self.bhe_eq, self.bhe_eq, sim_params, years=load_years
        )

        # hybrid load object
        self.hybrid_load = hybrid_load

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
