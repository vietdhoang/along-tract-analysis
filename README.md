# Along Tract Analysis Pipeline
## COMP 402D1 & D2: Honours Project in Computer Science and Biology
Supervisor: Dr. David Rudko

Multiple sclerosis (MS) is a chronic, neuroinflammatory disease that affects the central nervous system. It is an autoimmune condition that results in demyelination and axonal damage, disrupting the nervous system’s ability to transmit electrical signals and often causing progressive neurodegeneration. Despite being one of the most common neurological diseases among adults in Western countries (Filippi & Rocca, 2011) and with an estimated 250 000 to 350 000 people diagnosed with MS in the U.S. (Goldenberg, 2012), the mechanisms underlying MS pathology are not fully understood.

Diffusion weighted imaging (DWI) is a form of MR imaging that is sensitive to the diffusion of water molecules in the brain. The diffusion of water is hindered by obstacles such as membranes. This allows for the extraction of useful information regarding tissue structure. In a healthy axon, the direction of diffusion is primarily along the length of the axon. In MS, however, axonal damage and demyelination can result in a reduction in diffusion along the axon length and radially outward from the centre of the axon (Inglese & Bester, 2010). By quantifying differences in diffusion MRI parameters between healthy and MS patients, it is possible to gain deeper insight into the pathology of MS. 

In this project, we attempted to characterize the variability of along-tract MS pathology in a cohort of RRMS and SPMS subjects. Using probabilistic tractography to reconstruct the corticospinal tract (CST) we sampled diffusion metrics along the CST. Specifically, we computed fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD), and axial diffusivity (AD) at finely sampled vertex points along the CST.

## Requirements
### Python
* DIPY >= 1.1.1
* niblabel >= 3.1
* nipype >= 1.5
* matplotlib
* pandas
* numpy
* scipy
* seaborn
* compress-pickle
### MRI Tools
* FSL
* MINC Toolkit
* MRtrix3
* Freesurfer

## Pipeline

For this tutorial, we have copied all the scripts into `~/bin/`. This allaws us to use the scripts from anywhere without needing to explicitly write out the path to the script. Equivalently, you can also add the folder containing all the bash scripts to `$PATH`.

### Required data
* **DWI images**. If they are in the DICOM format, convert them to NIFTI. Tools like [dcm2niix](https://people.cas.sc.edu/rorden/mricron/dcm2nii.html) can be used.
* **aparc+aseg.mgz images**. These images are outputs after running Freesurfer's [recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all/). Convert these images to NIFTI format using Freesurfer's [mri_convert](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert)
* **brain.mgz images** These images are outputs after running Freesurfer's [recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all/). Convert these images to NIFTI format using Freesurfer's [mri_convert](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert). It is important to note that brain.mgz is in the same space as aparc+aseg.mgz and so we will be using brain.mgz to help transform aparc+aseg.mgz to DWI space.
* **Lesion masks**. These images will likely be in MINC format. Use the command from MINC toolkit [minc2nii](http://bic-mni.github.io/man-pages/man/mnc2nii.html) to convert to NIFTI.
* **T2-weighted images in the same image space as the lesion masks**. These images will likely be in MINC format. Use the command from MINC toolkit [minc2nii](http://bic-mni.github.io/man-pages/man/mnc2nii.html) to convert to NIFTI. Like brain.mgz, these images will be used to help transform the lesions masks to DWI space. 

### Brain extraction
Brain extraction is performed using FSL's Brain Extraction Tool (BET). Use the script `dwi_bet`, which performs brain extraction for DWI images, which contain multiple volumes per image. 
```bash
dwi_bet PATH/to/DWI fractional_intensity_threshold
```
Fractional intensity threshold determines how aggressive brain extraction will be. Higher values increases the probability of accidently removing brain tissue, while lower values might leave portions of the skull untouched. Values range between 0 and 1 with the default being 0.5  

### ROI extraction
To perform tractography of the CST, we need the endpoints of the CST, the brainstem and motor cortex. We use the script `aparc+aseg_2_roi` for this step.
```bash
aparc+aseg_2_roi PATH/to/aparc+aseg.nii.gz
```

### Registration
The ROIs (brainstem and motor cortex) have to be registered to DWI space. We will use the script `reg_roi` for this.
```bash
reg_roi PATH/to/T1w PATH/to/b0_img PATH/to/brainstem PATH/to/left_motor_cortex PATH/to/right_motor_cortex
```
The lesion masks also have to be registered to DWI space. These lesion masks are in STX152 space, which means we have to employ non-linear registration. More specifically, registration is done using Symmetric Normalization (SyN) algorithm proposed by Avants et al. This is the same algorithm used by ANTS. We will use the script `nlin_reg.py` for this step, which performs Symmetric Diffeomorphic Registration using DIPY tools.
```bash
python nlinreg.py PATH/to/b0_img PATH/to/t2w_img PATH/to/lesion_mask
```
### Tractography
The script `trk_gen` is used to reconstruct the CST using probabilistic tractograhpy. The script uses tools from MRtrix to perform tractography. In order to reconstruct the CST fiber bundle, the start point (brainstem) and endpoint (motor cortex) are provided as input.
```bash
trk_gen -n subject_name -d dwi_img -l bval_file -c bvec_file -b b0_img -r dir_roi
```
The output tractgrams can be view with MRtrx's [mrview](https://mrtrix.readthedocs.io/en/latest/reference/commands/mrview.html)

### Diffusion metrics
FA, MD, AD, and RD need to be calculated from diffusion weighted images. We use the script `dwi_2_metrics`.
```bash
dwi_2_metrics PATH/to/bval_file PATH/to/bvec_file PATH/to/DWI
```
We will now sample the diffusion metrics at 100 points along the CST. By using 100 points we can consider the points as '%-along CST'. For this step we will use the script `sample_metric.py`. Please read the documentation in the script as your directory structure may differ from the one that this script uses.
```bash
python sample_metric.py PATH/to/base_dir metric_name PATH/to/output_dir
```
The directory `base_dir` refers to the directory containing all subjects. This script will output two pickled files. Each file contains a pandas dataframe for the left and right CST. Each column in the dataframe is the diffusion metric profile for one patient. The script will have to be run four times, once for each metric.
```bash
python sample_metric.py PATH/to/base_dir ad PATH/to/output_dir
python sample_metric.py PATH/to/base_dir fa PATH/to/output_dir
python sample_metric.py PATH/to/base_dir md PATH/to/output_dir
python sample_metric.py PATH/to/base_dir rd PATH/to/output_dir
```

### Statistics and graphs
The script `stasts.py` the required statistical analysis for the honours project.
```bash
python stats.py PATH/to/base_dir
```
The directory `base_dir` refers to the directory containing all the pickled pandas dataframes of all the diffusion metrics.
