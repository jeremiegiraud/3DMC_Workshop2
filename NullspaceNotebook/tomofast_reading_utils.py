"""
Functions to read data & model grids from Tomofast-x format.
For more information: https://github.com/TOMOFAST/Tomofast-x
"""

import numpy as np
import os


def read_tomofast_data(grav_data, filename, data_type):
    """
    Read data and grid stored in Tomofast-x format.
    """

    data = np.loadtxt(filename, skiprows=1)

    if data_type == 'field':
        grav_data.data_field = data[:, 3]

    elif data_type == 'background':
        grav_data.background = data[:, 3]

    # Reading the data grid.
    grav_data.x_data = data[:, 0]
    grav_data.y_data = data[:, 1]
    grav_data.z_data = data[:, 2]


def read_tomofast_model(filename, mpars):
    """
    Read model values and model grid stored in Tomofast-x format.
    """

    # Check if the file exists.
    if not os.path.exists(filename):
        print(f"File '{filename}': not found. A dummy value of -9999 will be used instead.")
        dummy_value = -9999
        dummy_model = np.ones(mpars.dim) * dummy_value
        return dummy_model, mpars

    else:
        with open(filename, "r") as f:
            lines = f.readlines()
            n_elements = int(lines[0].split()[0])

        # Sanity check.
        assert n_elements == np.prod(mpars.dim), "Wrong model dimensions in read_tomofast_model!"

        model = np.loadtxt(filename, skiprows=1)

        # Get model values.
        m_inv = model[:, 6]
        m_inv = m_inv.reshape(mpars.dim)

        # Define model grid (cell centers).
        mpars.x = 0.5 * (model[:, 0] + model[:, 1])
        mpars.y = 0.5 * (model[:, 2] + model[:, 3])
        mpars.z = 0.5 * (model[:, 4] + model[:, 5])

        # Convert to km.
        mpars.x = mpars.x / 1000.
        mpars.y = mpars.y / 1000.
        mpars.z = mpars.z / 1000.

        return m_inv, mpars
