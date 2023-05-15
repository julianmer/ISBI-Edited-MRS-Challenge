import h5py
import numpy as np


def save_submission(result_spectra,ppm,filename):
    '''
    Save the results in the submission format
    Parameters:
        - results_spectra (np.array): Resulting predictions from test in format scan x spectral points
        - ppm (np.array): ppm values associataed with results, in same format
        - filename (str): name of the file to save results in, should end in .h5
    
    '''

    with h5py.File(filename,"w") as hf:
        hf.create_dataset("result_spectra",result_spectra.shape,dtype=float,data=result_spectra)
        hf.create_dataset("ppm",ppm.shape,dtype=float,data=ppm)


def save_submission_track3(result_spectra_2048, ppm_2048, result_spectra_4096, ppm_4096, filename):
    # create two groups for the two different resolutions
    with h5py.File(filename, "w") as hf:
        hf.create_group("2048")
        hf.create_group("4096")

        # save the 2048 resolution data in the 2048 group
        hf["2048"].create_dataset("result_spectra", result_spectra_2048.shape, dtype=float,
                                  data=result_spectra_2048)
        hf["2048"].create_dataset("ppm", ppm_2048.shape, dtype=float, data=ppm_2048)

        # save the 4096 resolution data in the 4096 group
        hf["4096"].create_dataset("result_spectra", result_spectra_4096.shape, dtype=float,
                                  data=result_spectra_4096)
        hf["4096"].create_dataset("ppm", ppm_4096.shape, dtype=float, data=ppm_4096)
