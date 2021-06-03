import re
import pandas as pd
import os
import numpy as np
import nibabel as nib


def read_ids(path):

    patient_ids = dict()
    df = pd.read_csv(path)
    dir_names = df['0'].tolist()

    p = re.compile("Fetus(?P<patient_id>[\d]+)")

    for name in dir_names:
        patient_id = p.findall(name)

        patient_ids[patient_id[0]] = name

    return patient_ids


def origin_id_from_filepath(filename):
    removed_extension = os.path.splitext(filename)[0]
    if '.nii' in removed_extension: #for .nii.gz extension
        removed_extension = os.path.splitext(removed_extension)[0]
    basename = os.path.basename(removed_extension)
    return basename


def id_from_filepath(filename):
    basename = os.path.basename(filename)
    p = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series_id>[\d]+)")
    ids = p.findall(basename)[0]
    patient_id = ids[0]
    series_id = ids[1]
    return patient_id + '_' + series_id


def resolution_from_scan_name(filename):
    try:
        basename = os.path.basename(filename)
        p = re.compile("Res(?P<x_res>[-+]?[0-9]*\.?[0-9]+)_(?P<y_res>[-+]?[0-9]*\.?[0-9]+)_Spac(?P<z_res>[+-]?([0-9]*[.])?[0-9]+)")
        res = p.findall(basename)[0]
        x_res = res[0]
        y_res = res[1]
        z_res = res[2]
    except:
        print("error in parsing resolution from name for file: " + filename)
        return None

    return [float(x_res), float(y_res), float(z_res)]


def patient_id_from_filepath(filename):
    patient_id, series_id = patient_series_id_from_filepath(filename)
    return patient_id


def patient_series_name_from_filepath(filename):
    res = patient_series_id_from_filepath(filename)
    if res is None:
        return None
    return res[0] + '_' + res[1]


def patient_series_id_from_filepath(filename):
    basename = os.path.basename(filename)
    p = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series_id>[\d]+)")
    find_res = p.findall(basename)

    if len(find_res)!=0:
        ids = p.findall(basename)[0]
        patient_id = ids[0]
        series_id = ids[1]
    else:
        p = re.compile("Fetus(?P<patient_id>[\d]+)_St(?P<st_id>[\d]+)_Se(?P<series_id>[\d]+)")
        find_res = p.findall(basename)[0]
        if len(find_res)!=0:
            patient_id = find_res[0]
            series_id = find_res[2]
        else:
            p = re.compile("Pat(?P<patient_id1>[\d]+)_(?P<patient_id2>[\d]+)_Se(?P<series_id>[\d]+)")
            find_res = p.findall(basename)
            if(len(find_res)!=0):
                find_res=find_res[0]
                patient_id = find_res[0] + '_' + find_res[1]
                series_id = find_res[2]
            else:
                return None,None

    return patient_id, series_id


def move_smallest_axis_to_z(vol):
    shape = vol.shape
    min_index = shape.index(min(shape))

    if(min_index != 2):
        vol = np.swapaxes(vol, min_index, 2)

    return vol, min_index


def swap_to_original_axis(swap_axis, vol):
    if(swap_axis != 2):
        new_vol = np.swapaxes(vol, swap_axis, 2)
        return new_vol
    return vol


def get_spacing_between_slices(scan_id, metadata_path=None, df=None, extract_scan_series_id=True):

    return get_metadata_value(scan_id, 'SpacingBetweenSlices', metadata_path, df, extract_scan_series_id=extract_scan_series_id)


def get_FOV(scan_id, metadata_path=None, df=None):

    fov_str = get_metadata_value(scan_id, 'FOV', metadata_path, df)
    if(fov_str==None):
        return None

    fov_str = fov_str.replace('[',"")
    fov_str = fov_str.replace(']',"")
    fov_list = fov_str.split(',')
    map_object=map(int,fov_list)
    return list(map_object)


def get_resolution(scan_id, metadata_path=None, df=None, extract_scan_series_id=True, in_plane_res_name='PixelSpacing'):
    if (df is not None) or (';' not in metadata_path):#only one metadata
        res_str = get_metadata_value(scan_id, column_name=in_plane_res_name, metadata_path=metadata_path, df=df, extract_scan_series_id=extract_scan_series_id)
        spacing = get_metadata_value(scan_id, 'SpacingBetweenSlices', metadata_path, df=df, extract_scan_series_id=extract_scan_series_id)
    else: #multiple data pathes
        met_pathes = metadata_path.split(';')
        for metadata in met_pathes:
            res_str = get_metadata_value(scan_id, column_name=in_plane_res_name, metadata_path=metadata, df=df, extract_scan_series_id=extract_scan_series_id)
            if res_str is not None:
                spacing = get_metadata_value(scan_id, 'SpacingBetweenSlices', metadata, df=df, extract_scan_series_id=extract_scan_series_id)
                break

    if(res_str==None):
        return None
    res_str = res_str.replace('[',"")
    res_str = res_str.replace(']',"")
    res_list = res_str.split(',')
    res_list[0] = float(res_list[0])
    res_list[1] = float(res_list[1])

    res_list.append(spacing)
    return res_list


def get_metadata(scan_id, metadata_path=None, df=None): #specify either metadata path or df
    try:
        subject_id, series_id = patient_series_id_from_filepath(scan_id)
    except:
        print('subject id and series id cannot be extracted!')
        return None
    try:
        if(df is None):
            df = pd.read_csv(metadata_path, encoding ="unicode_escape")
        row_df = df[(df['Subject'] == int(subject_id)) & (df['Series'] == int(series_id))]
    except:
        print("no information in scan " + scan_id)
        return None
    return row_df


def get_metadata_by_subject_id(subject_id, metadata_path=None, df=None): #specify either metadata path or df
    try:
        if(df is None):
            df = pd.read_csv(metadata_path, encoding ="unicode_escape")
        row_df = df[df['Subject'] == int(subject_id)]
    except:
        print("no information in scan " + subject_id)
        return None
    return row_df


def get_metadata_value(scan_id, column_name='', metadata_path=None, df=None, extract_scan_series_id=True):
    subject_id = None
    series_id = None

    if extract_scan_series_id:
        try:
            subject_id, series_id = patient_series_id_from_filepath(scan_id)
        except:
            print('subject id and series id cannot be extracted! Subject id will be scan_id')

    if subject_id is None:
        subject_id = scan_id
    print('scan id is: ' + str(subject_id))

    try:
        if(df is None):
            df = pd.read_csv(metadata_path, encoding ="unicode_escape")
    except:
        print('metadata path not correct')
        return None

    if (series_id is not None) and ('Series' in df.columns):
        row_df = df[(df['Subject'] == int(subject_id)) & (df['Series'] == int(series_id))]
    else: #subject_id is string
        try:
            row_df = df[(df['Subject'] == subject_id)]
        except:
            row_df = df[(df['Subject'] == int(subject_id))]
    try:
        value = row_df.iloc[0][column_name]
    except:
        print("no information about " + column_name + " in scan " + subject_id)
        return None

    return value

def get_vol_sizes(vol_ids, eval_folder, gt_filename):
    sizes_dict = {}

    for id in vol_ids:
        vol_path = os.path.join(eval_folder, str(id), gt_filename)
        vol = nib.load(vol_path).get_data()
        sizes_dict[id] = vol.shape

    return sizes_dict


def get_volumes_info(vol_ids, metadata_path, info_list):
    info_dict={}
    df = pd.read_csv(metadata_path, encoding ="unicode_escape")

    for id in vol_ids:
        info_dict[id] = {}
        for info in info_list:
            try:
                scan_info = get_metadata_value(id, info, metadata_path, df)
            except:
                print('no information about ' + info + ' for scan ' + id)
                info_dict[id][info] = ""
                continue
            info_dict[id][info] = scan_info

    return info_dict