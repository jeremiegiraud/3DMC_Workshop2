import numpy as np
from tomofast_utils import Haar3D, iHaar3D


def calc_fwd(sensitivity, model, unit_conv, use_csr_matrix, indices=None):
    """
    Calculate the forward gravity data of a density model using the provided sensitivity matrix.

    Parameters
    ----------
    sensitivity : numpy.ndarray,
        The sensitivity matrix to be used for the forward calculation
    model : numpy.ndarray
        The 3D model parameters
    unit_conv : bool, optional
        Whether to perform unit conversion to mGal or not, by default True
    use_csr_matrix: Flag controlling whether crs matrix format is used for sparse sensitivity matrix. Generally,
    use_csr_matrix is set to True if use sensitivity matrix from Tomofast-x.
    indices : numpy.ndarray of indices, optional

    Returns
    -------
    data: numpy.ndarray
        The calculated forward gravity data
    """


    # Factor to convert to mGal. For gravity data only.
    mgal_conversion_factor = 1e2

    model = model.flatten()

    # --------- Case with indices provided: use only some elements of sensitivity matrix.
    if indices is not None:
        if use_csr_matrix:
            model_scaled = model.copy()

            # Apply inverse depth weighting to the model.
            model_scaled = model_scaled / sensitivity.weight[indices]

            if sensitivity.compression_type == 1:
                # Transform the model into the wavelet domain.
                n1 = sensitivity.nz
                n2 = sensitivity.ny
                n3 = sensitivity.nx
                model_scaled = model_scaled.reshape((n1, n2, n3))
                Haar3D(model_scaled, n1, n2, n3)
                model_scaled = model_scaled.flatten()

        else:
            model_scaled = model

        # Flatten the 3D model to a 1D array.
        model = model.flatten()

        # Calculates the forward data and perform unit conversion if required.
        if use_csr_matrix:
            data = sensitivity.matrix[:, indices].dot(model_scaled)
        else:
            data = np.matmul(sensitivity[:, indices], model)

        if unit_conv:
            data = data * mgal_conversion_factor

        data = np.ravel(data)

    else:
        # --------- Case with not indices provided: use all elements of sensitivity matrix.
        if use_csr_matrix:
            model_scaled = model.copy()

            # Apply inverse depth weighting to the model.
            model_scaled = model_scaled / sensitivity.weight

            if sensitivity.compression_type == 1:
                # Transform the model into the wavelet domain.
                n1 = sensitivity.nz
                n2 = sensitivity.ny
                n3 = sensitivity.nx
                model_scaled = model_scaled.reshape((n1, n2, n3))
                Haar3D(model_scaled, n1, n2, n3)
                model_scaled = model_scaled.flatten()

        else:
            model_scaled = model

        # Flatten the 3D model to a 1D array.
        model = model.flatten()

        # Calculates the forward data and perform unit conversion if required.
        if use_csr_matrix:
            data = sensitivity.matrix.dot(model_scaled)
        else:
            data = np.matmul(sensitivity, model)

        if unit_conv:
            data = data * mgal_conversion_factor

        data = np.ravel(data)

    return data


def calc_geophy_data(geophy_data, sensitivity, dens_model, bouguer_anomaly, use_csr_matrix, unit_conv,base_model=None):
    """
    Get the forward gravity data.
    if base_model is provided, only the response of the difference with dens_model is calculated.

    :param geophy_data: pd dataframe of gravity data
    :param sensitivity: sensentivity matrix
    :param dens_model: density or density contrast model
    :param bouguer_anomaly: boolean controlling whether Bouguer anomaly is calculated
    :param use_csr_matrix:
    :param base_model:
    :return: None.
    """
    # TODO: make this function simpler and improve the cases where only a subset of the model is considered.
    # Pass the full model and not reduced, and then reduce it inside the function.
    # It could be that using indices array is not efficient and can cause memory issues with big matrices, see below:
    #        https://stackoverflow.com/questions/39500649/sparse-matrix-slicing-using-list-of-int

    # Copy and flatten array.
    model = dens_model.flatten()

    # Indices of the models cells used for calculation.
    # indices = np.arange(0, len(model))

    if base_model is not None:
        # print('Calculate the data for the difference between the two models provided.')

        # Local copy and flatten array
        model_base = base_model.flatten()

        # Calculate the difference between models.
        indices = np.squeeze(np.where(model_base != model))
        diff_values = model - model_base

        if not use_csr_matrix:
            raise Exception("Dense matrix not supported at the moment!")
        else:
            model = diff_values[indices]
            # print('Calculating the forward on the differences')

        # Calculate the forward data.
        if bouguer_anomaly:
            data_calc = calc_fwd(sensitivity, model, unit_conv, use_csr_matrix, indices) - np.array(geophy_data.background)
            geophy_data.data_calc = np.ravel(data_calc)

        elif not bouguer_anomaly:
            # TODO: add var 'indices' here too.
            data_calc = calc_fwd(sensitivity_reduced, model, unit_conv, use_csr_matrix)
            geophy_data.data_calc = np.ravel(data_calc)
        else:
            raise Exception("bouguer_anomaly: should be either True of False")

    elif base_model is None:

        if not use_csr_matrix:
            raise Exception("Dense matrix not supported at the moment!")

        # Calculate the forward data.
        if bouguer_anomaly:
            data_calc = calc_fwd(sensitivity, model, unit_conv, use_csr_matrix) - np.array(geophy_data.background)
            geophy_data.data_calc = np.ravel(data_calc)

        elif not bouguer_anomaly:
            # TODO: add var 'indices' here too.
            data_calc = calc_fwd(sensitivity_reduced, model, unit_conv, use_csr_matrix)
            geophy_data.data_calc = np.ravel(data_calc)
        else:
            raise Exception("bouguer_anomaly: should be either True of False")

    else:
        raise Exception("Wrong input type for model_base!")


def calc_resid_vect(geophy_data):
    """
    Calculate the gravity bouguer difference map between field data and a given density model

    :param geophy_data: panda dataframe containing gravity data for: measured, field, and background density model.
    :return residuals_vect: np.ndarray, the difference between field data and forward data.
    """

    residuals_vect = geophy_data.data_field - geophy_data.data_calc

    return residuals_vect


def calc_data_rms(gravity_data):
    """
    Calculate the root mean square difference (misfit) between calc_geophy_data and data_field
    and the calculated gravity data (data_calc), when geophy_data is a GravData class. Otherwise, when it is
    an array, it calculates the RMS value of geophy_data.

    :param gravity_data: GravData class containing the gravity data
    :return float, the residuals vector and data_misfit
    """

    if isinstance(gravity_data, np.ndarray):
        residuals_vect = gravity_data
    else:
        residuals_vect = calc_resid_vect(gravity_data)

    rms_data_misfit = np.sqrt(np.mean(np.square(residuals_vect)))

    return rms_data_misfit


def calc_grad_misfit(sensitivity, geophy_data, use_csr_matrix):
    """
    Calculate the gradient of the data misfit function 1/2*||d_obs - d_calc||**2 with d_calc = sensitivity * model.
    To apply depth weighting to the nullspace navigation, comment the line applying inverse depth weighting. 

    :param sensitivity: numpy.ndarray, The sensitivity matrix used in the computation of the gradient.
    :param geophy_data: panda.dataframe containing the observed gravity data.
    :param use_csr_matrix: boolean, indicating whether the "sensitivity" in compressed sparse row (CSR) format.
    :return: numpy.ndarray, The gradient of the data misfit function.
    """

    # Gravity data residual vector (d_obs - d_calc).
    residisuals_vector = calc_resid_vect(geophy_data)

    if use_csr_matrix:
        grad_misfit = - sensitivity.matrix.T.dot(residisuals_vector)
        if sensitivity.compression_type == 1:
            # Apply inverse wavelet transform.
            n1 = sensitivity.nz
            n2 = sensitivity.ny
            n3 = sensitivity.nx

            grad_misfit = grad_misfit.reshape(n1, n2, n3)
            iHaar3D(grad_misfit, n1, n2, n3)
            grad_misfit = grad_misfit.flatten()

        # Apply inverse depth weighting.
        grad_misfit = grad_misfit / sensitivity.weight

    else:
        grad_misfit = - np.matmul(sensitivity.T, residisuals_vector)

    # Normalization.
    grad_misfit = grad_misfit / len(residisuals_vector)

    return grad_misfit


def rotate_mesh(mpars, rotation_matrix):

    # TODO: is this used outside of Pyrenees case study?

    """
    Rotates locations in x, y, z coordinates using rotation_matrix

    :param mpars: ModelParameters object (see def in module gloop_nullspace) containing parameters for the model.
    :param rotation_matrix: numpy.ndarray, a 2x2 rotation matrix
    :return: tuple containing rotated x, y, z coordinates of the mesh
    """
    # TODO: move this to a new module called eg gloop_toolkit?

    coord = np.matmul(rotation_matrix, [mpars.x, mpars.y, mpars.z])
    coord_x = coord[0, :].reshape(mpars.dim)
    coord_y = coord[1, :].reshape(mpars.dim)
    coord_z = coord[2, :].reshape(mpars.dim)

    return coord_x, coord_y, coord_z
