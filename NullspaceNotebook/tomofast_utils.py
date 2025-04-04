from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
import struct
import os

@dataclass
class TomofastxSensit:
    nx: int
    ny: int
    nz: int
    compression_type: int
    matrix: object
    weight: object

#=======================================================================================================
def write_sensit_to_tomofastx(sensit_path, matrix, weight, nx, ny, nz, ndata, nbproc, b = None):
    """
    Writes a scipy csr_matrix matrix to the Tomofast-x format.

    sensit_path: output folder path
    matrix: scipy CSR matrix with Nrows = ndata, Ncolumns = nx * ny * nz
    weight: depth weight (1D numpy float array).
    b: right-hand side vector (1D numpy float array) - optional.
    """
    nel_total = nx * ny * nz
    nnz_total = matrix.data.size

    print('nel_total =', nel_total)
    print('nnz_total =', nnz_total)

    # Sanity check.
    if (matrix.shape != (ndata, nel_total)):
        raise Exception('Inconsistent matrix dimensions!')

    if (weight.shape != (nel_total,)):
        raise Exception('Inconsistent weight dimension!')

    # Some additional metadata needed in Tomofast-x.
    MATRIX_PRECISION = 4
    compression_type = 0
    comp_error = 0.
    nmodel_components = 1
    ndata_components = 1

    # Parallel matrix partitioning arrays.
    nnz_at_cpu_new = np.ndarray(shape=(nbproc), dtype=np.int32)
    nelements_at_cpu_new = np.ndarray(shape=(nbproc), dtype=np.int32)

    # TODO: Adjust for nbproc > 1.
    nnz_at_cpu_new[0] = nnz_total
    nelements_at_cpu_new[0] = nel_total

    # Metadata file.
    filename_metadata = sensit_path + "/sensit_grav_" + str(nbproc) + "_meta.dat"
    # Depth weight file.
    filename_weight = sensit_path + "/sensit_grav_" + str(nbproc) + "_weight"
    # Right-hand side file.
    filename_b = sensit_path + "/sensit_grav_" + str(nbproc) + "_b"

    # Create the output sensit folder.
    os.makedirs(os.path.dirname(filename_metadata), exist_ok=True)

    #----------------------------------------------------------
    # Writing the metadata.
    #----------------------------------------------------------
    with open(filename_metadata, "w") as f:
        f.write("{} {} {} {} {} {}\n".format(nx, ny, nz, ndata, nbproc, MATRIX_PRECISION))
        f.write("{} {}\n".format(compression_type, comp_error))
        f.write("{} {}\n".format(nmodel_components, ndata_components))
        np.savetxt(f, (nnz_at_cpu_new,), fmt="%d")
        np.savetxt(f, (nelements_at_cpu_new,), fmt="%d")

    print("Metadata file is written to:", filename_metadata)

    #----------------------------------------------------------
    # Writing the depth weight.
    #----------------------------------------------------------
    with open(filename_weight, "wb") as f:
        depth_weighting_type = 1
        # Write a header.
        f.write(struct.pack('>iiiii', nx, ny, nz, ndata, depth_weighting_type))

        # Convert to big-endian.
        weight = weight.astype('>f8')

        # Write weight to file.
        f.write(weight.tobytes())

    print("Weight file is written to:", filename_weight)

    #----------------------------------------------------------
    # Writing the right-hand side.
    #----------------------------------------------------------
    if (b is not None):
        with open(filename_b, "wb") as f:
            # Write a header.
            f.write(struct.pack('>i', b.size))

            # Convert to big-endian.
            b = b.astype('>f8')

            # Write b to file.
            f.write(b.tobytes())

        print("Right-hand side file is written to:", filename_b)

    #----------------------------------------------------------
    # Writing the matrix.
    #----------------------------------------------------------
    model_component = 1
    data_component = 1

    ndata_all = 0

    # Loop over parallel matrix chunks.
    for myrank in range(nbproc):

        # Sensitivity kernel file.
        filename_sensit = sensit_path + "/sensit_grav_" + str(nbproc) + "_" + str(myrank)

        # Building the matrix arrays.
        with open(filename_sensit, "wb") as f:
            # TODO: Adjust for the parallel case.
            ndata_loc = ndata

            # Write global header.
            f.write(struct.pack('>iiiii', ndata_loc, ndata, nel_total, myrank, nbproc))

            ndata_all += ndata_loc

            # Loop over matrix rows.
            for i in range(ndata_loc):
                # Global data index. 
                # TODO: Adjust for the parallel case.
                idata = i + 1

                # Number of non-zero elements in this row.
                nel = matrix.indptr[i + 1] - matrix.indptr[i]

                # Write local header.
                f.write(struct.pack('>iiii', idata, nel, model_component, data_component))

                # Array start/end indexes corresponding to the current matrix row.
                s = matrix.indptr[i]
                e = matrix.indptr[i + 1]

                # Extract data for one matrix row.
                col = matrix.indices[s:e]
                dat = matrix.data[s:e]

                # Shift column indexes to convert from Python to Fortran array index.
                col = col + 1

                # Convert to big-endian.
                col = col.astype('>i4')
                dat = dat.astype('>f4')

                # Writing one matrix row.
                f.write(col.tobytes())
                f.write(dat.tobytes())

        print("Sensitivity file is written to:", filename_sensit)

#=========================================================================================

def load_sensit_from_tomofastx(sensit_path, nbproc, type="grav", verbose=False):
    """
    Loads the sensitivity kernel from Tomofast-x and stores it in the CSR sparse matrix.
    """

    if type == "grav":
        prefix_sensit_name = "sensit_grav_"
    elif type == "magn":
        prefix_sensit_name = "sensit_magn_"
    else:
        print(type)
        raise Exception('Wrong type of sensitivity matrix!')

    # Metadata file.
    filename_metadata = sensit_path + "/" + prefix_sensit_name + "meta.txt"
    # Depth weight file.
    filename_weight = sensit_path + "/" + prefix_sensit_name + "weight"

    #----------------------------------------------------------
    # Reading the metadata.
    with open(filename_metadata, "r") as f:
        lines = f.readlines()

        # Read model dimensions.
        nx = int(lines[0].split()[0])
        ny = int(lines[0].split()[1])
        nz = int(lines[0].split()[2])

        if verbose:
            print('Tomofastx nx, ny, nz =', nx, ny, nz)

        # Reading the number of data.
        ndata_read = int(lines[0].split()[3])

        if verbose:
            print('ndata_read =', ndata_read)

        # Reading the number of procs.
        nbproc_read = int(lines[1].split()[0])

        if (nbproc != nbproc_read):
            raise Exception('Inconsistent nbproc!')

        if verbose:
            print('nbproc_read =', nbproc_read)

        compression_type = int(lines[2].split()[0])

        if verbose:
            print('compression_type =', compression_type)

        if compression_type > 1:
            raise Exception('Inconsistent compression type!')

        # The number of non-zero values.
        nnz_total = int(lines[4].split()[0])

        if verbose:
            print("nnz_total =", nnz_total)

    #----------------------------------------------------------
    # Reading depth weight.
    nel_total = nx * ny * nz
    with open(filename_weight, "r") as f:
        # Note using '>' for big-endian.
        header = np.fromfile(f, dtype='>i4', count=1)
        weight = np.fromfile(f, dtype='>f8', count=nel_total)

    #----------------------------------------------------------
    # Define spase matrix data arrays.
    # Note we a matrix constructor where the csr_row stores row indexes of all elements: a[row_ind[k], col_ind[k]] = data[k].
    csr_dat = np.ndarray(shape=(nnz_total), dtype=np.float32)
    csr_row = np.ndarray(shape=(nnz_total), dtype=np.int32)
    csr_col = np.ndarray(shape=(nnz_total,), dtype=np.int32)

    nel_current = 0
    ndata_all = 0

    # Loop over parallel matrix chunks.
    for n in range(nbproc):

        # Sensitivity kernel file.
        filename_sensit = sensit_path + "/" + prefix_sensit_name + str(nbproc) + "_" + str(n)

        # Building the matrix arrays.
        with open(filename_sensit, "r") as f:
            # Global header.
            header = np.fromfile(f, dtype='>i4', count=5)
            ndata_loc = header[0]
            ndata = header[1]
            nmodel = header[2]

            if verbose:
                print("ndata_loc =", ndata_loc)
                print("ndata =", ndata)
                print("nmodel =", nmodel)

            ndata_all += ndata_loc

            # Loop over matrix rows.
            for i in range(ndata_loc):
                # Local line header.
                header_loc = np.fromfile(f, dtype='>i4', count=4)

                # Global data index.
                idata = header_loc[0]

                # Number of non-zero elements in this row.
                nel = header_loc[1]

                # Reading one matrix row.
                col = np.fromfile(f, dtype='>i4', count=nel)
                dat = np.fromfile(f, dtype='>f4', count=nel)

                # Array start/end indexes corresponding to the current matrix row.
                s = nel_current
                e = nel_current + nel

                csr_col[s:e] = col
                csr_row[s:e] = idata - 1
                csr_dat[s:e] = dat

                nel_current = nel_current + nel
    #----------------------------------------------------------
    if verbose:
        print('ndata_all =', ndata_all)

    if (ndata_all != ndata_read):
        raise Exception('Wrong ndata value!')

    # Shift column indexes to convert from Fortran to Python array index.
    csr_col = csr_col - 1

    # Convert units from Tomofast to geomos (as we use different gravitational constant).
    if type == 'grav':
        csr_dat = csr_dat * 1.e+3  # TODO: make it all SI.

    # Create a sparse matrix object.
    matrix = csr_matrix((csr_dat, (csr_row, csr_col)), shape=(ndata_all, nmodel))

    sensit = TomofastxSensit(nx, ny, nz, compression_type, matrix, weight)

    # Keep minimal verbose.
    print("Sensitivity matrix from Tomofastx: loaded.")

    return sensit

#=========================================================================================
def Haar3D(s, n1, n2, n3):
    """
    Forward Haar wavelet transform.
    """

    for ic in range(3):
        if ic == 0:
            n_scale = int(np.log(float(n1)) / np.log(2.))
            L = n1
        elif ic == 1:
            n_scale = int(np.log(float(n2)) / np.log(2.))
            L = n2
        else:
            n_scale = int(np.log(float(n3)) / np.log(2.))
            L = n3

        for istep in range(1, n_scale + 1):
            step_incr = 2 ** istep
            ngmin = int(step_incr / 2) + 1
            ngmax = ngmin + int((L - ngmin) / step_incr) * step_incr
            ng = int((ngmax - ngmin) / step_incr) + 1
            step2 = step_incr

            #---------------------------------------------------
            # Predict.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] - s[il, 0:n2, 0:n3]
                elif ic == 1:
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] - s[0:n1, il, 0:n3]
                else:
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] - s[0:n1, 0:n2, il]

                il = il + step2
                ig = ig + step2

            #---------------------------------------------------
            # Update.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] + s[ig, 0:n2, 0:n3] / 2.
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] + s[0:n1, ig, 0:n3] / 2.
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] + s[0:n1, 0:n2, ig] / 2.

                il = il + step2
                ig = ig + step2

            #---------------------------------------------------
            # Normalization.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] * np.sqrt(2.)
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] / np.sqrt(2.)
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] * np.sqrt(2.)
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] / np.sqrt(2.)
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] * np.sqrt(2.)
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] / np.sqrt(2.)

                il = il + step2
                ig = ig + step2

#=========================================================================================
def iHaar3D(s, n1, n2, n3):
    """
    Inverse Haar wavelet transform.
    """

    for ic in range(3):
        if ic == 0:
            n_scale = int(np.log(float(n1)) / np.log(2.))
            L = n1
        elif ic == 1:
            n_scale = int(np.log(float(n2)) / np.log(2.))
            L = n2
        else:
            n_scale = int(np.log(float(n3)) / np.log(2.))
            L = n3

        for istep in reversed(range(1, n_scale + 1)):
            step_incr = 2 ** istep
            ngmin = int(step_incr / 2) + 1
            ngmax = ngmin + int((L - ngmin) / step_incr) * step_incr
            ng = int((ngmax - ngmin) / step_incr) + 1
            step2 = step_incr

            #---------------------------------------------------
            # Normalization.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] / np.sqrt(2.)
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] * np.sqrt(2.)
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] / np.sqrt(2.)
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] * np.sqrt(2.)
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] / np.sqrt(2.)
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] * np.sqrt(2.)

                il = il + step2
                ig = ig + step2

            #---------------------------------------------------
            # Update.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[il, 0:n2, 0:n3] = s[il, 0:n2, 0:n3] - s[ig, 0:n2, 0:n3] / 2.
                elif ic == 1:
                    s[0:n1, il, 0:n3] = s[0:n1, il, 0:n3] - s[0:n1, ig, 0:n3] / 2.
                else:
                    s[0:n1, 0:n2, il] = s[0:n1, 0:n2, il] - s[0:n1, 0:n2, ig] / 2.

                il = il + step2
                ig = ig + step2

            #---------------------------------------------------
            # Predict.
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    s[ig, 0:n2, 0:n3] = s[ig, 0:n2, 0:n3] + s[il, 0:n2, 0:n3]
                elif ic == 1:
                    s[0:n1, ig, 0:n3] = s[0:n1, ig, 0:n3] + s[0:n1, il, 0:n3]
                else:
                    s[0:n1, 0:n2, ig] = s[0:n1, 0:n2, ig] + s[0:n1, 0:n2, il]

                il = il + step2
                ig = ig + step2

#=========================================================================================
def test_write_sensit_to_tomofastx():
    """
    Testing the write_sensit_to_tomofastx() function.
    """
    sensit_path = "./SENSIT"
    nx = 2
    ny = 128
    nz = 32
    ndata = 256
    nbproc = 1

    nel_total = nx * ny * nz

    matrix_np = np.ndarray(shape=(ndata, nel_total), dtype=np.float32)

    # Put some matrix values.
    for i in range(ndata):
        matrix_np[i, :] = float(i + 1)

    matrix_np[:, 0] = 0.
    matrix_np[:, 5] = 0.

    # Create a scipy sparse matrix from dense ndarray.
    matrix = csr_matrix(matrix_np)

    # Depth weight array.
    weight = np.ndarray(shape=(nel_total), dtype=np.float32)
    weight[:] = 1.

    # Right-hand-side.
    b = np.ndarray(shape=(ndata + nel_total), dtype=np.float32)
    b[:] = 3.

    write_sensit_to_tomofastx(sensit_path, matrix, weight, nx, ny, nz, ndata, nbproc, b)

#=========================================================================================
if __name__ == "__main__":
    test_write_sensit_to_tomofastx()
