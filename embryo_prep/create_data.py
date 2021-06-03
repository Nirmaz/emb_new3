import glob
import os
import numpy as np
import tables
import nibabel as nib
import json
from scipy.ndimage import zoom
import data_generation.preprocess
from utils.read_write_data import pickle_load, pickle_dump
from data_curation.helper_functions import move_smallest_axis_to_z, get_resolution

def fetch_data_files(scans_dir, train_modalities, ext, return_subject_ids=False):
    data_files = list()
    subject_ids = list()

    if ';' not in scans_dir:#only one data path
        if(os.path.isdir(os.path.abspath(scans_dir)) == False):
            print('data dir: ' + scans_dir + 'does not exist!')
            return None
        scans_path = glob.glob(os.path.join(scans_dir, "*"))
        all_scans = scans_path
    else: #multiple data pathes
        dirs = scans_dir.split(';')
        all_scans = []
        for dir in dirs:
            if(os.path.isdir(os.path.abspath(dir)) == False):
                print('data dir: ' + dir + 'does not exist!')
                return None
            scans_path = glob.glob(os.path.join(dir, "*"))
            all_scans.extend(scans_path)

    for subject_dir in sorted(all_scans, key=os.path.basename):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in train_modalities:
            modalities = modality.split(';')
            for single_modality in modalities:
                file_path = os.path.join(subject_dir, single_modality + ".nii" + ext)
                if os.path.exists(file_path) or os.path.exists(file_path + '.gz'):
                    subject_files.append(file_path)
                    break
        data_files.append(tuple(subject_files))
    if return_subject_ids:
        return data_files, subject_ids
    else:
        return data_files


def read_img(in_file):
    filepath = os.path.abspath(in_file)
    if(os.path.exists(filepath)==False):
        filepath = filepath + '.gz'
        if(os.path.exists(filepath)==False):
            print('path' + filepath + 'does not exist!')
            return None
    print("Reading: {0}".format(in_file))
    image = nib.load(filepath)
    return image

def add_data_to_storage(storage_dict, subject_id, subject_data):
    storage_dict[subject_id] = {}
    storage_dict[subject_id]['data'] = np.asarray(subject_data[0]).astype(np.float)
    storage_dict[subject_id]['truth'] = np.asarray(subject_data[1]).astype(np.float)
    if len(subject_data) > 2:
        storage_dict[subject_id]['mask'] = np.asarray(subject_data[2]).astype(np.float)



def write_image_data_to_file(image_files, data_storage, subject_ids, scale=None, preproc=None, rescale_res=None, metadata_path=None, default_res= [1.56,1.56,3]):
    for subject_id, set_of_files in zip(subject_ids, image_files):
        images = [read_img(_) for _ in set_of_files]
        subject_data = [image.get_data() for image in images]

        subject_data[0], swap_axis = move_smallest_axis_to_z(subject_data[0])
        subject_data[1], swap_axis = move_smallest_axis_to_z(subject_data[1])
        if len(subject_data)==3: #mask also exists
            subject_data[2], swap_axis = move_smallest_axis_to_z(subject_data[2])

        if rescale_res is not None and metadata_path is not None:
            curr_resolution = get_resolution(subject_id, metadata_path=metadata_path)
            if(curr_resolution is None):
                curr_resolution = default_res
                print('resolution of case ' + subject_id + ' is not in metadata file, using default resolution of ' + str(default_res))
            scale_factor = [curr_resolution[0]/rescale_res[0], curr_resolution[1]/rescale_res[1],
                            curr_resolution[2]/rescale_res[2]]
            subject_data[0] = zoom(subject_data[0], scale_factor)
            subject_data[1] = zoom(subject_data[1], scale_factor, order=0)
            if len(subject_data)==3: #mask also exists
                subject_data[2] = zoom(subject_data[2], scale_factor, order=0)
            if scale is not None:
                print('both rescale_res and scale parameters are given. Using only rescale resolution parameter!')

        elif scale is not None:
            subject_data[0] = zoom(subject_data[0], scale)
            subject_data[1] = zoom(subject_data[1], scale, order=0)
            if len(subject_data)==3: #mask also exists
                subject_data[2] = zoom(subject_data[2], scale, order=0)

        if preproc is not None:
            print(preproc, "preproc")
            subject_data[0] = preproc(subject_data[0])
        print(subject_data[0].shape)
        add_data_to_storage(data_storage, subject_id, subject_data)
    return data_storage

def normalize_data(data, mean, std):
    data -= mean
    data /= std
    return data

def normalize_data_storage(data_dict: dict):
    means = list()
    stds = list()
    for key in data_dict:
        data = data_dict[key]['data']
        means.append(data.mean(axis=(-1, -2, -3)))
        stds.append(data.std(axis=(-1, -2, -3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for key in data_dict:
        data_dict[key]['data'] = normalize_data(data_dict[key]['data'], mean, std)
    return data_dict, mean, std


def normalize_data_storage_each(data_dict: dict):
    for key in data_dict:
        data = data_dict[key]
        mean = data.mean(axis=(-1, -2, -3))
        std = data.std(axis=(-1, -2, -3))
        data_dict[key] = normalize_data(data, mean, std)
    return data_dict, None, None

def write_data_to_file(training_data_files, out_file, subject_ids = None, normalize='all', scale=None, preproc=None, rescale_res=None, metadata_path=None):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image.
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """
    data_dict = {}

    write_image_data_to_file(training_data_files, data_dict, subject_ids, scale=scale, preproc=preproc, rescale_res=rescale_res, metadata_path=metadata_path)

    if isinstance(normalize, str):
        _, mean, std = {
            'all': normalize_data_storage,
            'each': normalize_data_storage_each
        }[normalize](data_dict)
    else:
        mean, std = None, None

    pickle_dump(data_dict, out_file)

    return out_file, (mean, std)

def open_data_file(filename):
    return pickle_load(filename)

def create_load_hdf5(normalization, data_dir, scans_dir, train_modalities, ext, overwrite = False, preprocess=None, scale=None, rescale_res=None, metadata_path=None):
    """
    This function normalizes raw data and creates hdf5 file if needed
    Returns loaded hdf5 file
    """
    data_file = os.path.join(data_dir, "fetal_data.h5")  #path to normalized data hdf5 file
    print('opening data file at: ' + data_file)
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(data_file):
        training_files, subject_ids = fetch_data_files(scans_dir, train_modalities, ext, return_subject_ids=True)
        print(training_files, "trainninh file")
        if preprocess is not None:
            preproc_func = getattr(data_generation.preprocess, preprocess)
        else:
            preproc_func = None

        _, (mean, std) = write_data_to_file(training_files, data_file, subject_ids=subject_ids, normalize=normalization, preproc=preproc_func, scale = scale, rescale_res = rescale_res, metadata_path = metadata_path)

        with open(os.path.join(data_dir, 'norm_params.json'), mode='w') as f:
            json.dump({'mean': mean, 'std': std}, f)

    data_file_opened = open_data_file(data_file)
    return data_file_opened


if __name__ == '__main__':
    create_load_hdf5("all", '/cs/casmip/nirm/embryo_project/DATA-AFTER/plcenta' ,'/cs/casmip/nirm/embryo_project/DATA-RAW/placenta', [ "volume", "truth"], "",False, "window_1_99", [0.5, 0.5, 1])
    create_load_hdf5("all", '/cs/casmip/nirm/embryo_project/DATA-AFTER/TRUFI' ,'/cs/casmip/nirm/embryo_project/DATA-RAW/TRUFI', [ "volume", "truth"], "",False, "window_1_99", [0.5, 0.5, 1])


