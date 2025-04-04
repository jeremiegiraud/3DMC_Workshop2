import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import path
import re
import math

import alphashape
from shapely.geometry import Polygon, MultiPolygon

"""
Functions for plots, loading data and adjacency matrix calc. 
This should be split into several specialised modules if things become more serious.

:Note: None of these functions are yet public so pls do refrain from sharing for now.

"""


def calc_models_misfit(model_ref, models_eval, n_models=0):
    """ Calculate the rms misfit between a ref model and a array containing many models. """

    def calc_rms_misfit(model1, model2):
        """ Calculates the rms difference between model 1 and model 2 """
        model_misfit = np.sqrt(np.mean(np.square(model1.flatten() - model2.flatten())))
        return model_misfit
    # Could do this instead.
    # from scipy.spatial.distance import rmsd

    if n_models > 0:
        misfit_model = np.zeros(n_models)
        for i in range(n_models):
            misfit_model[i] = calc_rms_misfit(model_ref, models_eval[i])

    else:
        misfit_model = calc_rms_misfit(model_ref, models_eval)

    return misfit_model


def get_prefix_filename(rt_file_name_):

    if len(rt_file_name_) == 1:
        file_prefix = rt_file_name_[0].split("\\")[-1]
        print('\n Doing for run: ', file_prefix)
    else:
        extracted_digits = [extract_digits(s) for s in rt_file_name_]
        filen_name_id = 'files_id_'
        for digits in extracted_digits:
            filen_name_id += '_' + digits
        file_prefix = filen_name_id

    return file_prefix


# Function to check if two plots overlap
def check_overlap_1(plot1, plot2):
    return not (plot1["x"] + plot1["width"] <= plot2["x"] or
                plot1["x"] >= plot2["x"] + plot2["width"] or
                plot1["y"] + plot1["height"] <= plot2["y"] or
                plot1["y"] >= plot2["y"] + plot2["height"])


# Function to adjust positions to remove overlap
def resolve_overlaps_1(plots_list, max_iterations=100, epsilon=0.01, canvas_width=1.):

    canvas_height = canvas_width  # Then it works only for square canvas.

    for _ in range(max_iterations):
        moved = False
        for i, plot1 in enumerate(plots_list):
            for j, plot2 in enumerate(plots_list):
                if i != j and check_overlap_1(plot1, plot2):
                    # Calculate displacement vector
                    dx = (plot1["x"] + plot1["width"] / 2) - (plot2["x"] + plot2["width"] / 2)
                    dy = (plot1["y"] + plot1["height"] / 2) - (plot2["y"] + plot2["height"] / 2)
                    distance = np.hypot(dx, dy)

                    # Apply repulsion if they overlap
                    if distance < epsilon:
                        distance = epsilon  # Prevent division by zero

                    shift_x = dx / distance * epsilon  # Small step size for shifting
                    shift_y = dy / distance * epsilon

                    # Move plots to reduce overlap
                    plot1["x"] += shift_x
                    plot1["y"] += shift_y

                    # Enforce boundaries
                    # plot1["x"] = max(0, min(canvas_width - plot1["width"], plot1["x"]))
                    # plot1["y"] = max(0, min(canvas_height - plot1["height"], plot1["y"]))

                    moved = True

        if not moved:
            break  # Stop if no plots were moved in the last iteration


def update_placed_plots(placed_plots, i, x, y, width, height,
                        x_orig=None, y_orig=None, width_orig=None, height_orig=None):
    if x_orig is None and y_orig is None and width_orig is None and height_orig is None:
        placed_plots.append({
            "id": i + 1,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "x_orig": np.nan,
            "y_orig": np.nan,
            "width_orig": np.nan,
            "height_orig": np.nan,
        })
    else:
        placed_plots.append({
            "id": i + 1,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "x_orig": x_orig,
            "y_orig": y_orig,
            "width_orig": width_orig,
            "height_orig": height_orig,
        })

    placed = True

    return placed, placed_plots


def check_overlap(x, y, width, height, placed_plots, margin):
    """ Function to check for overlap between plots on the canvas. """

    for plot in placed_plots:
        if not (x + width + margin <= plot["x"] or x >= plot["x"] + plot["width"] + margin or
                y + height + margin <= plot["y"] or y >= plot["y"] + plot["height"] + margin):
            return True
        
    return False


def create_inset(ax, x, y, width, height):
    """ Create an inset to place a plot on an existing canvas. """

    inset_axes = []
    # inset_axes.append([])

    inset_ax = ax.inset_axes([x, y, width, height], transform=ax.transData)
    inset_axes.append(inset_ax)
    inset_axes[-1].invert_xaxis()
    inset_axes[-1].invert_yaxis()
    # inset_ax[-1].axis('off')
    inset_axes[-1].set_yticklabels([])
    inset_axes[-1].set_xticklabels([])
    inset_axes[-1].set_xticks([])
    inset_axes[-1].set_yticks([])
    # inset_axes[-1].text(-5, 5, i)
    # inset_axes[-1].set_title(i)
    for axis in ['top', 'bottom', 'left', 'right']:
        inset_axes[-1].spines[axis].set_linewidth(0.25)

    return inset_axes


def calculate_subplots(num_subplots):
    """
    Function to calculate the optimum number of rows and columns for the provided total number of subplots.
    Calculate the closest square number that is greater than or equal to the given number.
    """

    num_rows = math.ceil(math.sqrt(num_subplots))
    num_cols = math.ceil(num_subplots / num_rows)

    return num_rows, num_cols


def get_core_indices(X, labels):
    core_indices = np.empty(0).astype(int)
    labels = labels if labels is not None else np.ones(X.shape[0])
    unique_labels = set(labels)
    for k in unique_labels:
        class_index = np.where(labels == k)[0]
        core_indices = np.append(core_indices, class_index)

    return core_indices


def extract_digits(s):
    """ Extract the digits at the end of a string. """
    match = re.search(r'\d{1,4}$', s)
    return match.group() if match else None


def save_plot(fig=None, filename='myplot', extension='.png', dpi=300, save=False):
    """
    Save the current figure to file or the figure provided in argument.

    :param: filename (str): The name of the output file.
    :param: dpi (int): Dots per inch (resolution) of the saved image (default: 300).
    :param: format (str): The format of the output file (default: 'png').
    :param: save (bool): Flag to indicate whether to save the plot (default: True).

    Returns: None
    """

    filename = filename + extension

    if save:

        # Check that the extension provided is OK.
        _, ext = path.splitext(filename)
        ext_lower = ext.lower()

        valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
        if ext_lower not in valid_extensions:
            raise ValueError("Invalid file extension. Supported extensions are: " + ", ".join(valid_extensions))

        if fig is None:
            # Get the current figure.
            fig = plt.gcf()

        # Do the saving;
        fig.savefig(filename, dpi=dpi, format=ext_lower[1:])


def print_info_clusters(labels, show=True):
    if show:
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print(f'Estimated number of clusters: {n_clusters_}')
        print(f'Estimated number of noise points: {n_noise_} on {labels.shape[0]} models)')

    return


def plot_tsne_clusters(clustered_dataset, labels, core_sample_indices, title="Taratata", show=True,
                       title1=None, title2=None, title3=None, title4=None,
                       plot_alpha_shape=False, plot_hull=False, alpha_hull=40., plot_all=True, size_dots=2, add_cluster_index=False):
    """
    Plot using output from DBSCAN.
    """

    def add_hull(ax_, points_, color_, alpha_hull_):

        # Restrict to 3 points or more.
        if np.max(np.shape(points_)) > 2:

            # Compute the alpha shape
            alpha = alpha_hull_  # Adjust this value for your data
            try:
                alpha_shape = alphashape.alphashape(points_, alpha)

                # Plot the points
                # plt.scatter(points_[:, 0], points_[:, 1])

                # Check if the alpha shape is valid and handle different geometries
                if alpha_shape.is_empty:
                    print("The alpha shape is empty. Try increasing the alpha value.")
                else:
                    if isinstance(alpha_shape, Polygon):
                        # Extract and plot the exterior of the polygon
                        x, y = alpha_shape.exterior.xy
                        ax_.plot(x, y, '--', color=color_)
                    elif isinstance(alpha_shape, MultiPolygon):
                        # Loop through and plot each polygon
                        for polygon in alpha_shape:
                            x, y = polygon.exterior.xy
                            ax_.plot(x, y, color=color_)
            except:
                print('The hull cannot be plotted because the points for the cluster are nearly colocated')

    def add_alphashape(ax_, points_, color_):
        """
        Plots the 2D alphashape.
        Some parameters are hardcoded.
        """

        # Restrict to 3 points or more.
        if np.max(np.shape(points_)) > 2:

            # Compute the alpha shape.
            alpha = 25.  # Adjust this value for the data.
            if np.max(np.shape(points_)) < 30:
                alpha /= 10
            else:
                alpha *= 1.5
            try:
                alpha_shape = alphashape.alphashape(points_, alpha)

                # Plot the points
                # plt.scatter(points_[:, 0], points_[:, 1])

                # Check if the alpha shape is valid and handle different geometries
                if alpha_shape.is_empty:
                    print("The alpha shape is empty. Try increasing the alpha value.")
                else:
                    if isinstance(alpha_shape, Polygon):
                        # Extract the exterior coordinates and fill the polygon
                        x_, y_ = alpha_shape.exterior.xy
                        ax_.fill(x_, y_, alpha=0.5, fc=color_, ec='black')
                    elif isinstance(alpha_shape, MultiPolygon):
                        # Loop through each polygon in the multipolygon
                        for polygon in alpha_shape:
                            x_, y_ = polygon.exterior.xy
                            ax_.fill(x_, y_, alpha=0.5, fc=color_, ec='black')
            except:
                print('The AlphaShape cannot be plotted because the points for the cluster are nearly colocated')

    def add_grid(ax_):
        """
        Add grid to existing plot axes as provided in input.
        """

        # Add the grid.
        ax_.grid()
        ax_.minorticks_on()

        # Set the style.
        ax_.grid(visible=True, which='minor', color='0.2', linestyle='--', alpha=0.1)
        ax_.grid(visible=True, which='major', color='0.6', linestyle='--', alpha=1)

    unique_labels = set(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    n_models = labels.shape[0]

    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]
    # colors = [plt.cm.prism(each) for each in np.linspace(0, 1, len(unique_labels))]
    # colors = [plt.cm.gist_rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]

    if plot_all:
        n_plots = 4
    else:
        n_plots = 1
    fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(10, 6), dpi=600)

    fig.suptitle(title, fontsize=10)

    for i in range(n_plots):
        if n_plots > 1:
            ax[i].set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('equal', adjustable='box')

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        # Extract the core samples
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[core_sample_indices] = True

        xy = clustered_dataset[class_member_mask & core_samples_mask]
        for i in range(n_plots):
            if i == 0 and not plot_all:
                if add_cluster_index and col != [0, 0, 0, 1]:
                    # Add text with index of cluster.
                    ax.text(np.mean(xy[:, 0]), np.mean(xy[:, 1]), str(k),
                               color='black',
                               bbox=dict(facecolor='white', alpha=0.85, edgecolor=(0.50, 0.50, 0.50),
                                         pad=0.25, linewidth=0.5), fontsize=9)
                    if plot_alpha_shape:
                        add_alphashape(ax, xy, tuple(col))
                    if plot_hull:
                        add_hull(ax, xy, tuple(col), alpha_hull)
            elif plot_all:
                # Add text with index of cluster.
                ax[i].text(np.mean(xy[:, 0]), np.mean(xy[:, 1]), str(k),
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor=(0.50, 0.50, 0.50),
                                  pad=0.25, linewidth=0.5), fontsize=9)
                if plot_alpha_shape:
                    add_alphashape(ax[i], xy, tuple(col))
                if plot_hull:
                    add_hull(ax[i], xy, tuple(col), alpha_hull)

            if col != [0, 0, 0, 1]:
                # if i!= 0:
                if n_plots > 1:
                    ax[i].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                               markeredgecolor='k', markersize=size_dots, markeredgewidth=0.5)
                else:
                    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                               markeredgecolor='k', markersize=size_dots, markeredgewidth=0.5)
            else:
                if n_plots > 1:
                    ax[i].plot(xy[:, 0], xy[:, 1], '+', markerfacecolor=tuple(col),
                               markeredgecolor='k', markersize=size_dots+1, markeredgewidth=0.5)
                else:
                    ax.plot(xy[:, 0], xy[:, 1], '+', markerfacecolor=tuple(col),
                               markeredgecolor='k', markersize=size_dots+1, markeredgewidth=0.5)
        # for j in range(len(xy[:, 0])):
        #     plt.text(xy[j, 0], xy[j, 1], str(k))

        xy = clustered_dataset[class_member_mask & ~core_samples_mask]

        for i in range(n_plots):
            if n_plots > 1:
                ax[i].plot(xy[:, 0], xy[:, 1], '+', markerfacecolor=tuple(col),
                           markeredgecolor='k', markersize=60, markeredgewidth=0.5)

                # ax[i].set_title(f'Est. num.of clusters: {n_clusters_} (est. num. of noise points: {n_noise_} on {n_models} models)')
                if i == 0:
                    if title1 is not None:
                        ax[i].set_title(title1)
                    else:
                        ax[i].set_title('Clusters with average models')
                elif i == 1:
                    if title2 is not None:
                        ax[i].set_title(title2)
                    else:
                        ax[i].set_title('Clusters with different with true model')
                elif i == 2:
                    if title3 is not None:
                        ax[i].set_title(title3)
                    else:
                        ax[i].set_title('Clusters with different with true model')
                elif i == 3:
                    if title4 is not None:
                        ax[i].set_title(title4)
                    else:
                        ax[i].set_title('Clusters with different with true model')

                ax[i].set_xlabel('t-SNE variable 1')
                ax[i].set_ylabel('t-SNE variable 2')
                add_grid(ax[i])

            else:
                ax.plot(xy[:, 0], xy[:, 1], '+', markerfacecolor=tuple(col),
                           markeredgecolor='k', markersize=60, markeredgewidth=0.5)
                if title1 is not None:
                    ax.set_title(title1)
                else:
                    ax.set_title('Clusters with average models')
                ax.set_xlabel('t-SNE variable 1')
                ax.set_ylabel('t-SNE variable 2')
                add_grid(ax)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def rescale_vector(vector):
    """
    Rescale the input vector to have values between 0 and 1.
    """

    min_val = np.min(vector)
    max_val = np.max(vector)

    # Sanity check.
    if min_val == max_val:
        return np.zeros_like(vector)

    rescaled_vector = (vector - min_val) / (max_val - min_val)

    return rescaled_vector


def calculate_adj_mat(model1):
    """
    Modified by JG from a script developped by Vitaliy Ogarko and Jeremie Giraud to test a topological adjancency matrix
    Jaccard index to compare pairs of models.

    Input:
    model1: 3D model - the model values are categorical, e.g., lithology type, unit type etc.

    Output:
    Resulting adjacency matrix.

    TODO: remove for workshop.
    """

    if not model1.ndim == 3:
        raise ValueError("The model should be 3D!")

    nx, ny, nz = np.shape(model1)

    def get_search_window(ind, size, ind_max):
        """
        Returns the min/max indexes for a search window of a given size around a given index.
        """

        left = max(ind - size, 0)
        right = min(ind + size, ind_max - 1)
        return left, right

    def get_all_neighbour_cells(ic, jc, kc, n_x, n_y, n_z):
        """
        Returns the 3D indexes of all neighbouring cells around a given cell.
        Takes into account the model boundary.
        """

        # Define the search window.
        ii_start, ii_end = get_search_window(ic, 1, n_x)
        jj_start, jj_end = get_search_window(jc, 1, n_y)
        kk_start, kk_end = get_search_window(kc, 1, n_z)

        # Get list of indices from loop over neighbouring cells.
        cells = list((ii, jj, kk)
                     for ii in range(ii_start, ii_end + 1)
                     for jj in range(jj_start, jj_end + 1)
                     for kk in range(kk_start, kk_end + 1)
                     # Skipping the central cell.
                     if (ii != ic or jj != jc or kk != kc)
                     )
        return cells

    def build_adj_matrix(model, rock_A, cells, nlithos):
        """
        Returns the adjacency matrix between the rock_A and the rocks in the given cells.
        """

        adj_matrix = np.zeros(shape=(nlithos, nlithos), dtype=float)
        # Loop over neighbouring cells.
        for cell in cells:
            # Define the rock B.
            rock_B = model[cell[0], cell[1], cell[2]]
            # Add contact to the matrix.
            if (rock_A <= rock_B):
                adj_matrix[rock_A, rock_B] += 1.
            else:
                adj_matrix[rock_B, rock_A] += 1.
        return adj_matrix

    # Count the total number of lithos.
    lithos1 = np.unique(model1)
    nlithos = len(lithos1)

    adj_matrix = np.zeros(shape=(nlithos, nlithos), dtype=float)

    # Loop over all model cells.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Define the rock A.
                rock1_A = model1[i, j, k]

                # Get all neighbouring cells.
                cells = get_all_neighbour_cells(i, j, k, nx, ny, nz)

                # Building the adjacency matrix.
                adj_matrix_tmp = build_adj_matrix(model1, rock1_A, cells, nlithos)
                adj_matrix = adj_matrix + adj_matrix_tmp

    return adj_matrix


def get_upper_triangular(matrix):
    """
    From the input square matrix, extracts the upper triangle including the diagonal and returns its vectorized form.
    """

    # Start with sanity check: check that the matrix is square.
    rows, cols = matrix.shape
    if not rows == cols:
        raise ValueError("The matrix is not square!")

    # Extract the upper triangular part of the input (square) matrix (including the diagonal).
    upper_triangular_indices = np.triu_indices(matrix.shape[0])

    # Vectorize.
    upper_triangular_vector = matrix[upper_triangular_indices]

    return upper_triangular_vector


def plot_clust_colors(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    """ Function slightly modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(X[ci, 0], X[ci, 1], "x" if k == -1 else "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=4 if k == -1 else 1 + 5 * proba_map[ci])

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()

    ax.set_aspect('equal', adjustable='box')