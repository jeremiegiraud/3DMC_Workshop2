from dataclasses import dataclass
import matplotlib.pylab as plt
import numpy as np
import colorcet as cc  # Used only for colormaps.
import random as rd
from typing import Optional
from forward_calculation_utils import rotate_mesh
import nullspace_utils as nu
import os
import vtk
import pandas as pd


@dataclass
class PlotParameters:
    """
    Parameters for plots of null space navigation outputs.
    at the moment: plots only in the x direction.
    """

    # Threshold for identification of density differences between models before/after navigation.
    plot_thresh: float = 30
    # Value of density contrast defining range of colors min and max.
    colm: float = 250
    # Slices for plots.
    slice_x: int = -1
    slice_y: int = -1
    slice_z: int = -1
    plot_models: Optional[tuple] = None
    # Titles for plots.
    plot_titles: Optional[tuple] = None
    # Titles for color bars.
    cbar_titles: Optional[tuple] = None
    # Limits for colours on plot.
    clims: Optional[tuple] = None
    # Colour schemes for plots.
    colorschemes: Optional[tuple] = None
    # Ticks for colorbar.
    cbar_ticks: Optional[tuple] = None
    # limits in x and y directions for plots
    xlims: Optional[np.array] = None
    ylims: Optional[np.array] = None
    # Save and plot intermediate results
    save_intermediate_plots: Optional[bool] = False
    # Interval at which intermediate results will the saved during navigation. 
    interval_intermediate_plots: Optional[int] = 1  # save every 10 iterations during navigation.


def save_metrics_csv(misfit_data, hamiltonian_quantities, filename='metrics_file', save=True): 
    """
    A function to save important metrics in a CSV file. 
    misfit_data: np.array of data misfit along the series of perturbations. 
    HamiltonQuantities class containing the quantities to calculate the Hamiltonian.
    filename: string containing the filename without extension. 
    save: Boolean determining whether the data will be saved. 
    """

    if save: 

        # Define column names and values.
        column_names = ['Iteration number', 'Data misfit', 'Artificial Hamiltonian', 'Kinetic Energy', 'Potential Energy']
        it_nums = np.linspace(1, len(misfit_data), len(misfit_data), dtype=int)

        values = [it_nums,misfit_data, hamiltonian_quantities.total_energy, hamiltonian_quantities.kinetic_energy, hamiltonian_quantities.potential_energy]

        # Convert list of lists into a DataFrame.
        df = pd.DataFrame(dict(zip(column_names, values)))

        # Save to CSV.
        df.to_csv(filename + ".csv", index=False)

        print("Data misfit evolution and Hamiltonian metrics saved in: " + filename + ".csv")

    else: 
        print("Data misfit evolution and Hamiltonian metrics NOT saved")

    return None


def save_data_to_vtk(geophy_dataclass, datatype_to_save='data_field', filename='data_file', save=True):
    """
    geophy_dataclass: GeophyData class.
    datatype_to_save can be 'data_calc', 'data_field',  or 'background' (see GeophyData class).
    filename: file name without extension.
    """

    if save:
        x = geophy_dataclass.x_data
        y = geophy_dataclass.y_data
        z = geophy_dataclass.z_data

        if datatype_to_save == 'data_field':
            values = geophy_dataclass.data_field
        elif datatype_to_save == 'data_calc':
            values = geophy_dataclass.data_calc
        elif datatype_to_save == 'background':
            values = geophy_dataclass.background
        elif datatype_to_save == 'difference':
            values = geophy_dataclass.data_field - geophy_dataclass.data_calc
        else: 
            raise Exception("datatype_to_save can only be data_field, data_calc, or background")

        num_points = len(values)

        # Create a VTK points object.
        points = vtk.vtkPoints()
        for i in range(num_points):
            points.InsertNextPoint(x[i], y[i], z[i])

        # Create a PolyData object and set the points.
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)    

        # Add scalar data (optional)
        scalars_array = vtk.vtkFloatArray()
        scalars_array.SetName("GeophyData")  # Name appears in ParaView
        for s in values:
            scalars_array.InsertNextValue(s)

        poly_data.GetPointData().SetScalars(scalars_array)

        # Write to VTK file (.vtp format)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + ".vtp")
        writer.SetInputData(poly_data)
        writer.Write()

        print("VTK data file saved as: " + filename + ".vtp")

    else:
        print("VTK file for data " + datatype_to_save + " NOT saved")

    return None


def save_model_to_vtk(voxel_data, grid_par_class, filename='voxet', save=True):
    """
    Create VTK structured grid from voxel data using the format used in the Nullspace script. 

    voxel_data: Voxel model to save. 
    grid_par_class: GridParameters dataclass. 
    Needs changing dimension order if input follows Python: VTK expects Fortran-style (column-major) ordering for structured grids.
    """

    if save:

        # Change dimension order from row-major to column-major. 
        mesh_dims = np.zeros_like(grid_par_class.dim)
        mesh_dims[0] = grid_par_class.dim[2]
        mesh_dims[1] = grid_par_class.dim[1]
        mesh_dims[2] = grid_par_class.dim[0]

        # Create the structured grid object.
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(mesh_dims[0], mesh_dims[1], mesh_dims[2])

        x_grid = grid_par_class.x.reshape(mesh_dims)
        y_grid = grid_par_class.y.reshape(mesh_dims)
        z_grid = grid_par_class.z.reshape(mesh_dims)

        # Create points for the grid.
        points = vtk.vtkPoints()
        for i in range(mesh_dims[0]):
            for j in range(mesh_dims[1]):
                for k in range(mesh_dims[2]):
                    points.InsertNextPoint(x_grid[i,j,k], y_grid[i,j,k], z_grid[i,j,k])

        structured_grid.SetPoints(points)

        # Create voxel data (example: random values).
        voxel_data = voxel_data.astype(np.float32)

        # Convert NumPy array to VTK array.
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetName("PhysPropertyValue")
        vtk_array.SetArray(voxel_data, voxel_data.size, 1)

        # Assign data to structured grid.
        structured_grid.GetPointData().SetScalars(vtk_array)

        # Write to .vts file.
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename + ".vts")
        writer.SetInputData(structured_grid)
        writer.Write()

        # Print message for the user. 
        # if verbose:
        #     print("Voxet saved as " + filename + ".vts for visualization in ParaView.")
    
    else: 
        print("VTK file for model " + filename + " not saved")

    return None
    

def add_grid(ax):
    """
    Add grid to existing plot axes.
    """

    # Add grid.
    ax.grid()
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='0.2', linestyle='--', alpha=0.1)
    ax.grid(visible=True, which='major', color='0.6', linestyle='--', alpha=1)


def set_plotprops():
    """
    Set default plot properties to use for all plots in the script.
    """

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})


def plot_addticks_cbar(cbar_title, cbar_ticks=None):
    """
    Add colorbar with specified title and ticks.

    :param cbar_title: title for the colorbar.
    :param cbar_ticks: location of the ticks on the colorbar.
    :return: colobar handle.
    """

    if cbar_title is None:
        cbar_title = 'SI'

    if cbar_ticks is None:

        cbar = plt.colorbar(shrink=0.75, orientation='vertical')
        # cbar.set_label(cbar_title, labelpad=-20, y=-0.015, rotation=0, fontfamily='serif')
        # cbar.set_label(cbar_title, labelpad=-20, x=1.15, y=-0.02, rotation=0)
        # cbar.set_label(cbar_title, labelpad=-20, x=1.10, y=1.125, rotation=0)
        cbar.set_label(cbar_title, labelpad=-20, x=1.10, y=1.125, rotation=0)

    else: 

        cbar = plt.colorbar(shrink=0.75, ticks=cbar_ticks, orientation='vertical')
        # cbar.set_label(cbar_title, labelpad=-20, y=-0.015, rotation=0, fontfamily='serif')
        # cbar.set_label(cbar_title, labelpad=-20, x=1.15, y=-0.02, rotation=0)
        cbar.set_label(cbar_title, labelpad=-20, x=1.10, y=1.125, rotation=0)

        ## Changing the font of ticks.
        # for i in cbar.ax.yaxis.get_title():
        #     i.set_family("Comic Sans MS")

    return cbar


def calc_plot_coordinates(mpars, ppars):
    """
    Get the distance along a profile (oblique or not) crossing the modelling mesh.

    :param ppars: PlotParameters object.
    :param mpars: ModelParameters object.
    :return: dist_profile: 2D ndarray containing, z_plot: 2D ndarray of the corresponding depth.
    """

    # For Pyrenees case.
    # x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, 9:-10]
    # y_plot = mpars.y.reshape(mpars.dim)[:, ppars.slice_x, 9:-10]
    # z_plot = -mpars.z.reshape(mpars.dim)[:, ppars.slice_x, 9:-10]

    # For homogenous example.
    x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, :]
    y_plot = mpars.y.reshape(mpars.dim)[:, ppars.slice_x, :]
    z_plot = mpars.z.reshape(mpars.dim)[:, ppars.slice_x, :]

    x_min = np.min(x_plot)
    y_min = np.min(y_plot)

    dist_profile = np.round(np.sqrt((x_plot - x_min) ** 2 + (y_plot - y_min) ** 2))

    return dist_profile, z_plot


def plot_metrics_perturbation(misfit_evolution, hamiltonian, show=True):
    """
    Plot the evolution of geophy data misfit and the misfit due to the anomaly assessed using nullspace shuttle

    :param misfit_evolution: 1D array,  The evolution of geophy data misfit.
    :param geophy_data: GeophyData object, The geophy data class contains the geophy data + coordinates.
    :param hamiltonian: HamiltonQuantities class containing the quantities to calculate the Hamiltonian.
    :return: None, only plots the data.
    """

    # Rounding the total energy so that matplotlib is not affect by numerical inaccuracy to plot nearly constant values.
    # hamiltonian.total_energy = np.around(hamiltonian.total_energy * 10000) / 10000  # Careful with this!
    # hamiltonian.total_energy = np.round(hamiltonian.total_energy, 3)  # Rounding to 4th digit, matplolib problems.

    # Number of iterations that will be plotted. 
    n_it_plot = len(misfit_evolution[misfit_evolution>0] ) - 1

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(10, 6.75), constrained_layout=True)
    # fig.tight_layout()

    ax = fig.add_subplot(4, 1, 1)
    plt.plot(misfit_evolution[:n_it_plot])
    plt.title('(a) Data misfit during null space navigation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Data misfit (mGal)')
    add_grid(ax)

    # fig = plt.figure(rd.randint(0, int(1e6)), figsize=(8, 8))
    ax = fig.add_subplot(4, 1, 2)
    ax.plot(hamiltonian.total_energy[:n_it_plot])
    plt.title('(b) Artificial Hamiltonian (total energy)')
    ax.set_ylabel('Total Energy')
    if n_it_plot>0:
        ax.set_ylim([0, np.max(hamiltonian.total_energy[:n_it_plot])*1.05])
    ax.set_xlabel('Epochs')
    add_grid(ax)

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(hamiltonian.kinetic_energy[:n_it_plot])
    plt.title('(c) Kinetic energy')
    ax.set_ylabel('Kinetic Energy')
    ax.set_xlabel('Epochs')
    add_grid(ax)

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(hamiltonian.potential_energy[:n_it_plot])
    plt.title('(d) Potential energy')
    ax.set_ylabel('Potential Energy')
    ax.set_xlabel('Epochs')
    add_grid(ax)

    if show:
        plt.show()

    return fig


def plot_data(geophy_data, geophy_data_diff, rotation_matrix):
    """   
    :param geophy_data_diff: 1D array, The forward geophy data of the due to the anomaly assessed using nullspace shuttle
    :param rotation_matrix: 2D array, The rotation matrix used to rotate the data.
    """ 

    geophy_data.x_data, geophy_data.y_data = rotate_data(geophy_data, rotation_matrix)

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(10, 10), constrained_layout=True)

    ax = fig.add_subplot(4, 1, 1)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data_diff)  # edgecolors='black')
    plt.scatter(geophy_data.x_data[np.abs(geophy_data_diff) > 1.5] / 1e3,
                geophy_data.y_data[np.abs(geophy_data_diff) > 1.5] / 1e3, 10,
                marker='.',
                color='k', linewidth=1)
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(a) Forward data of the perturbation')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    ax = fig.add_subplot(4, 1, 2)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data.data_field)  # edgecolors='black')
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(b) Field data')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    ax = fig.add_subplot(4, 1, 3)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data.data_calc)  # edgecolors='black')

    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(c) Calculated data')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    ax = fig.add_subplot(4, 1, 4)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data.data_field - geophy_data.data_calc)  # edgecolors='black')
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(d) Residuals')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    plt.show()

    return fig


def plot_model(ax, mesh_dim1, mesh_dim2, mod, slice_plot, title_string, cmap, clim):
    """
    Plots a 2D section of gridded model along a specific slice.

    :param ax: The axes object to plot onto.
    :param mesh_dim1: numpy.ndarray, 2D, mesh along the 1st dimension to plot.
    :param mesh_dim2: numpy.ndarray, 2D, mesh along the 2nd dimension to plot.
    :param mod: numpy.ndarray, 2D, The slice to plot.
    :param slice_plot: int, indices of the slice in the 3D model.
    :param title_string: str, The title of the plot.
    :param cmap: matplotlib.colors.LinearSegmentedColormap, The color map to use for the plot.
    :param clim: tuple, The color limits to use for the plot.
    :return: matplotlib.collections.QuadMesh
    """

    if clim is not None:
        color_min = clim[0]
        color_max = clim[1]
    else: 
        color_min = None
        color_max = None

    # TODO: make use of the PlotParameters class here

    # For Pyrenees case.
    # handle = plt.pcolormesh(mesh_dim1, mesh_dim2, mod[:, slice_plot, 9:-10], cmap=cmap, vmin=color_min, vmax=color_max,
    #                         label='test1')

    # For homogenous model example.
    # matrix_to_plot = np.asfortranarray(mod[:, slice_plot, :])
    matrix_to_plot = mod[:, slice_plot, :]
    handle = plt.pcolormesh(mesh_dim1, mesh_dim2, matrix_to_plot, cmap=cmap, vmin=color_min, vmax=color_max,
                            label='test1')

    # plt.text(np.min(mesh_dim1) - 7, np.max(mesh_dim2) + 3, 'A', fontsize=13)
    # plt.text(np.max(mesh_dim1) + 1, np.max(mesh_dim2) + 3, 'B', fontsize=13)

    add_grid(ax)

    ax.set_aspect('equal', 'box'),
    ax.set_xlabel('Distance along profile (km)')
    ax.set_ylabel('Depth (km)')

    plt.box(on=bool(1))
    plt.title(title_string)

    return handle


def get_large_diff(mpars, ppars, model1, model2):
    """
    Identifies cells from 3D volume along slice with indices ppars.slice_x where the value of teh differences between
     model1 and model 2 is superior to ppars.plot_thresh, using the mask mask_location.

    :param ppars: PlotParameters object.
    :param mpars: ModelParameters object.
    :param model1: numpy.ndarray, 1D array containing the model to calculate the difference / update to plot.
    :param model2: numpy.ndarray, 1D array containing the model to calculate the difference / update to plot.
    :return: A tuple containing two arrays. The first array contains the indices of all cells with values greater than
             ppars.plot_thresh, and the second array contains the indices of cells with values greater than
             ppars.plot_thresh along the slice with indices ppars.slice_x.
    """

    # Find where mask was not applied.
    # mask_model_nodiff = 1 - mask_location.reshape(mpars.dim)
    # Get part of the model where mask was not applied, ie where it could evolve during null space navigation.
    # model_diff_masked = nu.apply_mask_diff(mask_model_nodiff, model1, model2).reshape(mpars.dim)

    # Calculate the differences between models.
    model_diff = nu.calc_model_diff(model1, model2).reshape(mpars.dim)
    # Restrict analysis to subdomain.
    model_diff_masked_slice = model_diff[:, ppars.slice_x, :]
    # Identify indices of cells with values superior to a predefined threshold on the slice.
    ind_bigdiff_all = np.where(np.abs(model_diff) > ppars.plot_thresh)
    # Identify indices of cells with values superior to a predefined threshold on the slice.
    ind_bigdiff_slice = np.where(np.abs(model_diff_masked_slice) > ppars.plot_thresh)

    return ind_bigdiff_all, ind_bigdiff_slice


def plot_navigation_xsection(mpars, ppars, ind_scatter):
    """
    Plots four 2D sections of the gridded model, along a specific slice, with a scatter plot on top for the last one.

    :param ind_scatter: tuple of length 2 with numpy.ndarray of indices along selected profile to use for scatter plot.
    :param mpars: ModelParameters object containing parameters for the model.
    :param ppars: PlotParameters object containing parameters for the plot.
    :return: None.
    """

    plot_models = ppars.plot_models
    colorschemes = ppars.colorschemes
    clims = ppars.clims
    cbar_ticks = ppars.cbar_ticks
    cbar_titles = ppars.cbar_titles
    plot_titles = ppars.plot_titles
    slice_x = ppars.slice_x

    # Get the coordinates for plotting.
    # Used for the Pyrenees. 
    # dist_profile, z_plot = calc_plot_coordinates(mpars, ppars)

    z_plot = mpars.z.reshape(mpars.dim)[:, ppars.slice_x, :]
    x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, :]

    n_subplots = 3
    n_row_subplots = 3
    n_columns_subplot = 1

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(13, 7))

    for i in range(0, n_subplots):

        ax = fig.add_subplot(n_row_subplots, n_columns_subplot, int(i + 1))
        # Used for the Pyrenees.
        # plot_model(ax, mesh_dim1=dist_profile, mesh_dim2=z_plot, mod=plot_models[i],
        #            slice_plot=slice_x, title_string=plot_titles[i], cmap=colorschemes[i], clim=clims[i])
        # plot_addticks_cbar(cbar_titles[i], cbar_ticks[i])

        if clims[i] is None: 
            # Use the End of null space navigation model for clims of plots. 
            clim_tmp = np.array([-np.max(np.abs((plot_models[1]))), +np.max(np.abs((plot_models[1])))])
            plot_model(ax, mesh_dim1=x_plot, mesh_dim2=z_plot, mod=plot_models[i],
                    slice_plot=slice_x, title_string=plot_titles[i], cmap=colorschemes[i], clim=clim_tmp)
        else: 
            plot_model(ax, mesh_dim1=x_plot, mesh_dim2=z_plot, mod=plot_models[i],
                    slice_plot=slice_x, title_string=plot_titles[i], cmap=colorschemes[i], clim=clims[i])
        plot_addticks_cbar(cbar_titles[i])
        
        plt.xlim((ppars.xlims[0], ppars.xlims[1]))

        ax.invert_yaxis()

        # Adding the plot of black dots showing differences above a threshold specified in the parameter file.
        # 2nd panel.
        # if i == 1:
        #     # For Pyrenees.
        #     # plt.scatter(dist_profile[ind_scatter[0], ind_scatter[1]-9], z_plot[ind_scatter[0], ind_scatter[1]-9],
        #     #             alpha=0.5, s=1, c='black', label='Values superior to threshold')
        #     plt.scatter(x_plot[ind_scatter[0], ind_scatter[1]], z_plot[ind_scatter[0], ind_scatter[1]],
        #                 alpha=0.5, s=1, c='black', label='Values superior to threshold')
        # # 4th panel.
        # if i == 3:
        #     # For Pyrenees.
        #     # plt.scatter(dist_profile[ind_scatter[0], ind_scatter[1]-9], z_plot[ind_scatter[0], ind_scatter[1]-9],
        #     #             alpha=0.5, s=1, c='black', label='Values superior to threshold')
        #     plt.scatter(x_plot[ind_scatter[0], ind_scatter[1]], z_plot[ind_scatter[0], ind_scatter[1]-9],
        #                 alpha=0.5, s=1, c='black', label='Values superior to threshold')

    # For Pyrenees. 
    # plt.text(np.min(dist_profile) - 7, np.max(z_plot) + 3, 'A', fontsize=13)
    # plt.text(np.max(dist_profile) + 1, np.max(z_plot) + 3, 'B', fontsize=13)
    # ax.annotate("Original perturbation", xy=(135, -24), xycoords='data', xytext=(15, -25), textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.legend()
    fig.tight_layout()
    plt.show()

    return fig


def rotate_data(geophy_data, rotation_matrix):
    # TODO: move this function somewhere? 
    """
    Rotates the location of geophy data in geophy_data using rotation_matrix

    :param geophy_data: GeophyData object, The geophy data class contains the geophy data + coordinates.
    :param rotation_matrix: numpy.ndarray, a 2x2 rotation matrix
    :return: rotated x_data and y_data, tuple of numpy.ndarray
    """

    x_data = geophy_data.x_data
    y_data = geophy_data.y_data

    coord = np.matmul(rotation_matrix[0:2, 0:2], np.array([x_data, y_data]))
    x_data = coord[0, :]
    y_data = coord[1, :]

    return x_data, y_data


def plot_navigation_depthslice(mpars, ppars, rotation_matrix, indice_scatter):
    """
    Plots four depth slices of geophysical models using rotation_matrix to rotate locations in x, y, z coordinates,
    with a scatter plot on top for the last one.

    :param mpars: ModelParameters object containing parameters for the model.
    :param ppars: PlotParameters object containing parameters for the plot.
    :param rotation_matrix: numpy.ndarray, a 3x3 rotation matrix.
    :param indice_scatter: An array containing the indices of scatter points to be plotted.
    :return: None
    """

    n_subplots = 3
    n_row_subplots = 2
    n_columns_subplot = 2

    plot_models = ppars.plot_models

    coord_x, coord_y, coord_z = rotate_mesh(mpars, rotation_matrix)

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(8, 8), constrained_layout=True)

    for i in range(0, n_subplots):
        ax = fig.add_subplot(n_row_subplots, n_columns_subplot, i + 1)
        plt.pcolormesh(coord_x[ppars.slice_z, :, :], coord_y[ppars.slice_z, :, :],
                       plot_models[i][ppars.slice_z, :, :],
                       cmap=ppars.colorschemes[i], clim=ppars.clims[i])
        add_grid(ax)
        # ax.set_aspect('equal'),
        ax.set_aspect('equal', 'box')
        plt.title(ppars.plot_titles[i])
        # plot_addticks_cbar(ppars.cbar_titles[i], ppars.cbar_ticks[i])
        plot_addticks_cbar(ppars.cbar_titles[i])
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        plt.xlim(ppars.xlims)
        plt.ylim(ppars.ylims)

    plt.scatter(coord_x[indice_scatter[0], indice_scatter[1], indice_scatter[2]],
                coord_y[indice_scatter[0], indice_scatter[1], indice_scatter[2]],
                alpha=0.15, s=5, c='black', marker='o', label='Values superior to threshold')
    # For Pyrenees field application.
    # ax.annotate("Original perturbation", xy=(633, 4792), xycoords='data', xytext=(600, 4820), textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.legend()
    # fig.tight_layout()
    plt.show()

    return fig


def prepare_plots(dim, mvars, m_diff, ppars, xlims, ylims):
    """
    Define plot parameters: models to plot, titles, limits etc.
    """

    print('\nPlot parameters are hardcoded in function', prepare_plots.__name__, "in file",  os.path.basename(__file__))

    # Models to plot in the 2x2 subplot.
    # First subplot.
    m1 = mvars.delta_m_orig.reshape(dim)
    # Second subplot.
    m2 = mvars.m_nullspace_orig.reshape(dim)
    # Third subplot.
    m3 = m_diff.reshape(dim)

    ppars.plot_models = (m1, m2, m3)

    # Color limits for the subplots.
    # For Pyrenees field case.
    # ppars.clims = (np.array([m1.min(), m1.max()]),  # In example shown in paper: m3 is the starting model.
    #                np.array([m1.min(), m1.max()]),
    #                np.array([m1.min(), m1.max()]),
    #                np.array([-200, 200]))
    # For homogenous model example.
    # ppars.clims = (np.array([-300, 300]),
    #                np.array([-300, 300]),
    #                np.array([-300, 300]),
    #                np.array([-300, 300]))
    ppars.clims = (None,
                   None,
                   None)

    # # Colormaps for each subplot.
    ppars.colorschemes = (cc.cm.CET_R4,
                          cc.cm.CET_R4,
                        #   cc.cm.CET_R4,
                          'seismic')

    # Titles for each subplot.
    ppars.plot_titles = ('(a) Start of nullspace navigation:',
                         '(b) End of null space navigation',
                         '(c) Difference: End - Start')

    # Titles for each colorbar attached to the subplots.
    ppars.cbar_titles = ('SI',
                         'SI',
                         'SI')

    # Ticks for each colorbar.
    # For Pyrenees field case.
    # ppars.cbar_ticks = ([2400, 2600, 2800, 3000, 3200],
    #                     [2400, 2600, 2800, 3000, 3200],
    #                     [2400, 2600, 2800, 3000, 3200],
    #                     [-200, -100, 0, 100, 200])
    # For homogenous model example.
    # ppars.cbar_ticks = ([-200, -100, 0, 100, 200],
    #                     [-200, -100, 0, 100, 200],
    #                     [-200, -100, 0, 100, 200],
    #                     [-200, -100, 0, 100, 200])


    ppars.xlims = xlims
    ppars.ylims = ylims

    return ppars


def save_plot(fig=None, filename='myplot', ext='.png', dpi=300, save=False):
    """
    Save the current figure to file or the figure provided in argument.

    :param: filename (str): The name of the output file.
    :param: dpi (int): Dots per inch (resolution) of the saved image (default: 300).
    :param: format (str): The format of the output file (default: 'png').
    :param: save (bool): Flag to indicate whether to save the plot (default: True).

    Returns: None
    """

    filename = filename + ext

    if save:

        # Check that the extension provided is OK.
        _, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
        if ext_lower not in valid_extensions:
            raise ValueError("Invalid file extension. Supported extensions are: " + ", ".join(valid_extensions))

        if fig is None:
            # Get the current figure.
            fig = plt.gcf()

        # Do the saving;
        fig.savefig(filename, dpi=dpi, format=ext_lower[1:])
