'''
Viet Hoang
ID: 260789801

This script samples diffusion metrics along the CST using tools provided by 
DIPY. It executes the following steps:

    1. For each patient,
        
        a. It loads the left nad right CST and calculates the centroid streamline 
        for the both tracts. Thisc an be though of as the 'average streamline'.
        
        b. It reorients all the streamlines so that they all follow the same 
        direction, either from the brainstem to the motor cortex or the other 
        way around. The direction is depends on the centroid streamline and will 
        vary from patient to patient,so you will have to verify that all patients 
        follow the same direction. For example, if one patient has streamlines going
        from the motor cortex to the brainstem while all other patients have
        streamlines going from the brain stem to the motor cortex, then you will
        have to reverse the streamlines for that one patient

        c. Calculate the weight of each streamline at every point along the 
        tract. Streamlines closer to the centroid bundle are given greater 
        weight than streamlines further away.

        d. Sample the metric along the tract.
    
    2. Save the patient in a pandas dataframe. Each column of the dataframe
    represents the diffusion metric profile of one patient

    3. Output the pandas dataframe using pickle

Usage:
python sample_metric.py PATH/to/base_dir metric_name PATH/to/output_dir

'''

# I/O
import sys
from dipy.data import pjoin
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tck

# Along tract profiling
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import (AveragePointwiseEuclideanMetric,
                                 ResampleFeature)
from dipy.tracking.streamline import orient_by_streamline

# Misc.
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


def get_tract_profile(bundle, metric_img, metric_affine, use_weights=False, 
                      flip=True, num_points=100):
    '''
    This function reorients the streamlines and extracts the diffusion metrics 
    along the tract. It essentiall performs step 1. The default number of points
    along a tract is 100, which can be thought of as %-along a tract.

    The flip variable signals if you would like to flip the direction of the
    streamlines after reorientation. For example if after reorientation all the
    streamlines were motor cortex -> brainstem and you actually wanted 
    brainstem -> motor cortex, then you set flip to True. The default is True
    because generally we see reorientation result in motor cortex -> brainstem.
    For the honours project, we were looking for the opposite
    '''


    # Reorient all the streamlines so that they are follwing the same direction
    feature = ResampleFeature(nb_points=num_points)
    d_metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(np.inf, metric=d_metric)
    centroid_bundle = qb.cluster(bundle).centroids[0]
    oriented_bundle = orient_by_streamline(bundle, centroid_bundle)
    
    # Calculate weights for each streamline/node in a bundle, based on a 
    # Mahalanobis distance from the core the bundle, at that node
    w_bundle = None
    if use_weights:        
        w_bundle = gaussian_weights(oriented_bundle)
    
    # Sample the metric along the tract. The implementation of this function
    # is based off of work by Yeatman et al. in 2012
    profile_bundle = afq_profile(metric_img, oriented_bundle, 
                                 metric_affine, weights=w_bundle)
    
    # Reverse the profile bundle if the direction is not desired
    if flip:
        profile_bundle = np.flip(profile_bundle)

    return profile_bundle

def main():

    '''
    For this script, we have assume the following subject directory structure,
    although the code can be modified for different structures.
    
    base/
    |── subject1/
    |   |── dwi/
    |   |   └── b0 image
    |   |── lesion/
    |   |   └── lesion mask
    |   |── cst/
    |   |   |── left cst
    |   |   └── right cst
    |   └── metrics/
    |       |── ad map
    |       |── fa map
    |       |── md map
    |       └── rd map
    |── subject2/
    |   .
    |   .
    |   .
    
    Each subject has a code name of the form xxx_yyy_m00_MS where xxx is a three
    letter code, yyy is a 3 digit code, m00 is the month 0 scane, and MS is
    the MS phenotype (PP, SP, RR, HC)
    '''

    # Get the base directory containing all the patient data and output directory
    # The output directory can be anywhere
    base = sys.argv[1] 
    out = sys.argv[3]
    
    # List of subjects. These subject names are not real and are randomly
    # generated.
    subjects = ["anb_022_m00_PP", "bwc_037_m00_HC", 
                "czd_021_m00_RR", "xyt_023_m00_SP"] 
    
    # This script can only process one diffusion metric at any point. Run 
    # this script 4 times to get diffusion profiles for all 4 diffusion metrics
    metric_name = sys.argv[2]
    
    # Initialize output dataframes
    lh_metric_df = pd.DataFrame()
    lh_lesion_df = pd.DataFrame()
    rh_metric_df = pd.DataFrame()
    rh_lesion_df = pd.DataFrame()
    
    # Loop through each subject
    for subj in subjects: 

        print(f"Processing {subj}...")

        b0_path = pjoin(base, subj, 'dwi', subj[:-3]+'_dwi_b0_noskull.nii.gz')
        
        lh_path = pjoin(base, subj, 'cst', subj[:-3]+'_lh_cst_edit.tck')
        rh_path = pjoin(base, subj, 'cst', subj[:-3]+'_rh_cst_edit.tck')
        
        lh = load_tck(lh_path, b0_path).streamlines        
        rh = load_tck(rh_path, b0_path).streamlines

        metric_path = pjoin(base, subj, 'metrics', 
                            subj[:-3]+'_dwi_noskull_'+metric_name+'.nii.gz')
        metric_img, metric_aff = load_nifti(metric_path)

        lesion_path = pjoin(base, subj, 'lesion', subj[:-3]+'_ct2f_dtisp.nii.gz')
        lesion_img, lesion_aff = load_nifti(lesion_path)

        
        # Get the along tract profile of the metric (FA, MD, etc.)
        lh_profile = get_tract_profile(lh, metric_img, metric_aff, 
                                        use_weights=True)
        # Get the along tract profile of the metric (FA, MD, etc.)
        rh_profile = get_tract_profile(rh, metric_img, metric_aff, 
                                        use_weights=True)
        
        lh_metric_df[subj] = lh_profile
        rh_metric_df[subj] = rh_profile

        # Find out which part of the tract intesects the roi
        lh_lesion_profile = get_tract_profile(lh, lesion_img, lesion_aff)
        rh_lesion_profile = get_tract_profile(rh, lesion_img, lesion_aff)

        lh_lesion_df[subj] = lh_lesion_profile        
        rh_lesion_df[subj] = rh_lesion_profile
    
    # Fill in NaN values
    lh_metric_df.fillna(method='ffill', inplace=True)
    lh_lesion_df.fillna(method='ffill', inplace=True)
    rh_metric_df.fillna(method='ffill', inplace=True)
    rh_lesion_df.fillna(method='ffill', inplace=True)
    
    # Save and output the pandas dataframes as pickle files.

    output_metric = [lh_metric_df, rh_metric_df]
    output_metric_path = pjoin(out, f'{metric_name}.pkl')
    
    with open(output_metric_path, 'wb') as handle:
        pickle.dump(output_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)

    output_lesion = [lh_lesion_df, rh_lesion_df]
    output_lesion_path = pjoin(out, 'lesion.pkl')
    
    with open(output_lesion_path, 'wb') as handle:
        pickle.dump(output_lesion, handle, protocol=pickle.HIGHEST_PROTOCOL) 


if __name__ == "__main__":
    main()