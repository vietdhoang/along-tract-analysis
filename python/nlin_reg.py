'''
Viet Hoang
July 14, 2020

This script non-linearly registers a set of images (all in the same image space)
to another space (ex. MNI152). Registration is performed in two steps:
    
    1. Initial linear registration for rough alignment. This is done using
    FSL's FLIRT.
    
    2. Non-linear registration (Symmetric Diffeomorphic Registration) for fine-
    tuning. For more information, please read the tutorial available on DIPY:
    https://dipy.org/documentation/1.4.1./examples_built/syn_registration_3d

    3. Subsequent images are transformed into MNI space using the computed 
    transform of the first image.

Required input:
Reference image (either T1- or T2-weighted)
At least one image to register. The first image (img1) has to be T1 or T2.

Example usage:
python nlinreg.py PATH/to/ref PATH/to/img1 PATH/to/img2 ...                          

Cost function for FLIRT will be mutual info, but it can be changed in the code
'''

# I/O
from compress_pickle import dump
from dipy.io.image import load_nifti, save_nifti
import nibabel as nib

# Registration
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.viz import regtools
from nipype.interfaces import fsl

# Utilities
import numpy as np
from os.path import join as pjoin
from os.path import basename as basename
from pathlib import Path
from re import split as split
import sys

def main():
    
    path_static_data = sys.argv[1] # Path of atlas is the first arg
    type_moving_data = sys.argv[2] # The type of img1 is the second arg    
    path_moving_data = sys.argv[3] # Path of img1 (moving data) is the third arg

    # Extract the basename of the moving image to use later for saving other images
    moving_data_basename = split("\.", basename(path_moving_data))[0]
    
    # Perform an initial linear registration using FLIRT and update the path
    # to the new transformed image
    path_moving_data, path_aff_mat = flirt(path_static_data, 
                                            path_moving_data, 
                                            type_moving_data,
                                            moving_data_basename)
    
    # Load the moving data and atlas
    moving_data, moving_affine = load_nifti(path_moving_data)    
    static_data, static_affine = load_nifti(path_static_data)  

    # Perform non-linear registration
    warped_moving, mapping = syn_registration(static_data, static_affine, 
                                                moving_data, moving_affine)

    # Saving the registration results
    path_warped_moving = pjoin(Path(path_moving_data).parent, 
                                moving_data_basename + "_nlinreg.nii.gz")

    nib.save(nib.Nifti1Image(warped_moving.astype(np.float32), static_affine), 
            path_warped_moving)

    # Save the optimized mapping object for future use
    dump(mapping, 
        pjoin(Path(path_moving_data).parent, moving_data_basename + "_map.gz"), 
        compression="gzip", 
        set_default_extension=True)
    
    # Apply the affine transformation and warp to all the other images
    for i in range (4, len(sys.argv)):
        img_basename = split("\.", basename(sys.argv[i]))[0]
        path_img_flrt = apply_mat(path_static_data, sys.argv[i], path_aff_mat, img_basename)

        img_data, img_affine = load_nifti(path_img_flrt)

        warped_img = apply_syn_registration(img_data, mapping)
        
        path_warped_img = pjoin(Path(sys.argv[i]).parent, 
                                    img_basename + "_nlinreg.nii.gz")

        nib.save(nib.Nifti1Image(warped_img.astype(np.float32), static_affine), 
                path_warped_img)
    
def flirt(path_static_data, path_moving_data, type_moving_data, basename
          cost="mutualinfo"):
    
    # Initialize transform object and selecting the paramaters for FLIRT
    transform = fsl.FLIRT() 
    transform.inputs.in_file = path_moving_data
    transform.inputs.reference = path_static_data
    transform.inputs.out_file = pjoin(Path(path_moving_data).parent, 
                                        basename + "_flirt.nii.gz")
    transform.inputs.out_matrix_file = pjoin(Path(path_moving_data).parent, 
                                            basename + "_flirt_omat.nii.gz")
    transform.inputs.dof = 12

    transform.inputs.cost = cost
    
    # print(transform.cmdline) # Print the FLIRT command that will be executed
    
    transform.run() # Run FLIRT
    
    # Return the path to the output files
    return transform.inputs.out_file, transform.inputs.out_matrix_file

def apply_mat(path_static_data, path_moving_data, path_aff_mat, basename): 
    
    # Initialize transform object and selecting the paramaters for FLIRT
    transform = fsl.FLIRT()
    transform.inputs.in_file = path_moving_data
    transform.inputs.reference = path_static_data
    transform.inputs.apply_xfm = True
    transform.inputs.in_matrix_file = path_aff_mat
    transform.inputs.out_file = pjoin(Path(path_moving_data).parent, 
                                        basename + "_flirt.nii.gz")
    transform.inputs.out_matrix_file = pjoin(Path(path_moving_data).parent, 
                                            basename + "_flirt_omat.nii.gz")
    
    # print(transform.cmdline) # Print the FLIRT command that will be executed

    transform.run() # Run FLIRT

    return transform.inputs.out_file

def syn_registration(static_data, static_affine, moving_data, moving_affine):
    
    # Terminology for comments:
    #   1. Static image = Reference image
    #   2. Moving image = Image that will be registered to the static image

    # Prealignment has been done using FLIRT, so the identity matrix is used since
    # no further transformation is necessary
    pre_align = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])    
    
    # We want to find an invertible map that transforms the moving image into the
    # static image. We will use the Cross Correlation metric. 
    metric = CCMetric(3)

    # Now we define an instance of the registration class. The SyN algorithm uses
    # a multi-resolution approach by building a Gaussian Pyramid. We instruct the
    # registration object to perform at most [n_0, n_1, ..., n_k] iterations at
    # each level of the pyramid. The 0-th level corresponds to the finest resolution. 
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    # Execute the optimization, which returns a DiffeomorphicMap object,
    # that can be used to register images back and forth between the static and
    # moving domains. We provide the pre-aligning matrix that brings the moving
    # image closer to the static image    
    mapping = sdr.optimize(static_data, moving_data, 
                            static_affine, moving_affine, pre_align)

    # Warp the moving image and see if it gets similar to the static image
    warped_moving = mapping.transform(moving_data)

    return warped_moving, mapping

def apply_syn_registration(img_data, mapping):
    
    # Apply mapping generated previously
    warped_img_data = mapping.transform(img_data)

    return warped_img_data

if __name__ == "__main__":
    main()