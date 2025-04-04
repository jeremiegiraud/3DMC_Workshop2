import configparser
from dataclasses import dataclass


@dataclass
class InputParameters:
    """
    A class to contains all input parameters used in the Parfile.
    """
    # -------------------------------
    # Section 'FilePaths'.
    # -------------------------------
    model_filename: str = ""
    perturbation_filename: str = ""
    # Geophysical data, e.g., Bouguer anomaly.
    data_vals_filename = 'data/gravity_data/data_vals.txt'
    # Value of the background model used, e.g., in the calculation of the Bouguer anomaly.
    data_background_filename = 'data/gravity_data/data_background.txt'
    path_output: str = "output/"
    sensit_path: str = "input/SENSIT/"
    rotation_mat_filename: str = ""
    geol_model_path: str = ""
    data_outline_filename: str = ""

    # -------------------------------
    # Section 'SolverParameters'.
    # -------------------------------
    # Flag defining if we import sensitivity kernel from Tomofast-x.
    use_tomofast_sensit: bool = True
    # String of characters determining the type of inversion / sensitivity matrix ('grav' or 'magn').
    sensit_type: str = ""
    # Sensitivity matrix. 
    sensit = None  # TODO: fix, it was used as a trick for 3DMC in Freo. 
    # Number of procs used to calculate the sensitivity kernel with Tomofast.
    tomofast_sensit_nbproc: int = 1
    # Flag defining whether we rotate the data for plotting.
    use_rotation_matrix: bool = False
    # Flag on unit conversion
    unit_conv: bool = True
    # Flag on whether we use a mask to reduce the domain where modifications of the model are allowed.
    use_mask_domain: bool = True
    # Weight of prior model term. (<0: larger variations, >0: smaller variations. Depth weight can go here)
    weight_prior_model: float = -5.e-12
    # - 2.5e-12 for plunging crust
    # 1.e-11 for mantle chunk in axial zone.
    # Weight of the positivity constraints (for magnetic data only) [Optional]
    weight_pos_const: float = 0.

    # Tolerance on misfit variations during null space navigation.
    eps: float = 0.25
    # Maximum difference between the first model of navigation and the current model.
    max_change: float = 440.
    # Number of time steps.
    num_epochs: int = 350
    # Length of a time step (scales the perturbation at each iteration).
    time_step: float = 100

    # -------------------------------
    # Section 'GridParameters'.
    # -------------------------------
    # Dimensions of the mesh
    nx: int = 54
    ny: int = 68
    nz: int = 31

    # ------------------------------------
    # Section 'PreProcessingParameters'.
    # ------------------------------------
    # Index of rock unit (by increasing density value) to define the mask on perturbations. 9 = Mantle.
    ind_unit_mask: int = 9
    # Distance max in number of cells away from the outline of rock unit considered.
    distance_max: int = 4  # 4, 8 in tests shown in Pyrenees paper.

    # ------------------------------------
    # Section 'SaveOutput'.
    # ------------------------------------
    save_plots: bool = False
    save_intermediate_plots: bool = False
    interval_intermediate_plots: int = 20

# =============================================================================
def read_input_parameters(parfile_path, print_values=True):
    """
    Read input parameters from Parfile.
    """

    def print_par(str, print_vals=print_values):
        if print_vals:
            print(str)
        return None


    print('Provided Parfile path:', parfile_path)

    config = configparser.ConfigParser()
    if len(config.read(parfile_path)) == 0:
        raise ValueError("Failed to open/find a parameters file!")

    par = InputParameters()

    print_par('----------------- READING THE PARFILE ----------------- \n')

    section = 'FilePaths'
    print_par('\n ------ ' + section + ' ------')
    print_par("\n".join(f"{key} = {value}" for key, value in config.items(section)))

    par.model_filename = config.get(section, 'model_filename', fallback=par.model_filename)
    par.perturbation_filename = config.get(section, 'perturbation_filename', fallback=par.perturbation_filename)
    par.data_vals_filename = config.get(section, 'data_vals_filename', fallback=par.data_vals_filename)
    par.data_background_filename = config.get(section, 'data_background_filename',
                                              fallback=par.data_background_filename)
    par.path_output = config.get(section, 'path_output', fallback=par.path_output)
    par.sensit_path = config.get(section, 'sensit_path', fallback=par.sensit_path)
    par.rotation_mat_filename = config.get(section, 'rotation_mat_filename', fallback=par.rotation_mat_filename)
    par.geol_model_path = config.get(section, 'geol_model_path', fallback=par.geol_model_path)

    section = 'SolverParameters'
    print_par('\n ------ ' + section + ' ------')
    print_par("\n".join(f"{key} = {value}" for key, value in config.items(section)))

    par.sensit_type = config.get(section, 'sensit_type', fallback=par.sensit_type)
    par.tomofast_sensit_nbproc = config.getint(section, 'tomofast_sensit_nbproc', fallback=par.tomofast_sensit_nbproc)
    par.use_rotation_matrix = config.getboolean(section, 'use_rotation_matrix', fallback=par.use_rotation_matrix)
    par.unit_conv = config.getboolean(section, 'unit_conv', fallback=par.unit_conv)
    par.use_mask_domain = config.getboolean(section, 'use_mask_domain', fallback=par.use_mask_domain)
    par.weight_prior_model = config.getfloat(section, 'weight_prior_model', fallback=par.weight_prior_model)
    par.weight_pos_const = config.getfloat(section, 'weight_pos_const', fallback=par.weight_pos_const)
    par.eps = config.getfloat(section, 'eps', fallback=par.eps)  # TODO: reorder order of variables here so it matches the order in the parfile.
    par.max_change = config.getfloat(section, 'max_change', fallback=par.max_change)
    par.num_epochs = config.getint(section, 'num_epochs', fallback=par.num_epochs)
    par.time_step = config.getfloat(section, 'time_step', fallback=par.time_step)

    section = 'GridParameters'
    print_par('\n ------ ' + section + ' ------')
    print_par("\n".join(f"{key} = {value}" for key, value in config.items(section)))

    par.nx = config.getint(section, 'nx', fallback=par.nx)
    par.ny = config.getint(section, 'ny', fallback=par.ny)
    par.nz = config.getint(section, 'nz', fallback=par.nz)

    section = 'PreProcessingParameters'
    print_par('\n ------ ' + section + ' ------')
    print_par("\n".join(f"{key} = {value}" for key, value in config.items(section)))

    par.ind_unit_mask = config.getint(section, 'ind_unit_mask', fallback=par.ind_unit_mask)
    par.distance_max = config.getint(section, 'distance_max', fallback=par.distance_max)

    section = 'SaveOutput'
    print_par('\n ------ ' + section + ' ------')
    print_par("\n".join(f"{key} = {value}" for key, value in config.items(section)))
    par.save_plots = config.getboolean(section, 'save_plots', fallback=par.save_plots)
    par.save_intermediate_plots = config.getboolean(section, 'save_intermediate_plots', fallback=par.save_intermediate_plots)
    par.interval_intermediate_plots = config.getint(section, 'interval_intermediate_plots', fallback=par.interval_intermediate_plots)

    return par
