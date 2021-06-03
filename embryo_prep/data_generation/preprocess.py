import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from skimage.exposure import equalize_adapthist, rescale_intensity
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis


def window_1_99(data, min_percent=2, max_percent=99):
    print(data,"data")
    image = sitk.GetImageFromArray(data)
    image = sitk.IntensityWindowing(image,
                                    np.percentile(data, min_percent),
                                    np.percentile(data, max_percent))
    return sitk.GetArrayFromImage(image)


def window_1_99_2D(data, min_percent=2, max_percent=99):
    """
    Perform windowing for each 2D slice separately
    :param data:
    :param min_percent:
    :param max_percent:
    :return:
    """
    for i in range(0,data.shape[2]):
        print(data, "data")
        image = sitk.GetImageFromArray(data[:,:,i])
        image = sitk.IntensityWindowing(image,
                                        np.percentile(data, min_percent),
                                        np.percentile(data, max_percent))
        data[:,:,i] = sitk.GetArrayFromImage(image)

    return data


def adapt_hist(data):
    data, swap_axis = move_smallest_axis_to_z(data)
    data_adapthist = np.empty(data.shape)
    for i in range(0,data.shape[2]):
        min_intensity = np.min(data[:,:,i])
        max_intensity = np.max(data[:,:,i])
        scaled_intensity_img = rescale_intensity(data[:,:,i],out_range=(0,1))
        adaptist_res = equalize_adapthist(scaled_intensity_img)
        data_adapthist[:,:,i] = rescale_intensity(adaptist_res, out_range=(min_intensity,max_intensity))#rescale back to original intensity

    data_adapthist = swap_to_original_axis(swap_axis, data_adapthist)
    return data_adapthist


def correct_bias(in_file, out_file):
    """
    Applies the new itk N4BiasFieldCorrectionImageFilter
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    inputImage = sitk.ReadImage( in_file)
    maskImage = sitk.OtsuThreshold( inputImage, 0, 1, 200 )

    inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )

    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    output = corrector.Execute( inputImage, maskImage )

    sitk.WriteImage( output, out_file )


#def hist_match(data, matching_hist):



def normalize_data(data, mean, std):
    data -= mean
    data /= std
    return data


def norm_minmax(d):
    return -1 + 2 * (d - d.min()) / (d.max() - d.min())


def laplace(d):
    return ndimage.laplace(d)


def laplace_norm(d):
    return norm_minmax(laplace(d))


def grad(d):
    return ndimage.gaussian_gradient_magnitude(d, sigma=(1,1,1))


def grad_norm(d):
    return norm_minmax(grad(d))