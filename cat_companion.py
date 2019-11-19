# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:42:08 2018

@author: federico nemmi
"""
from pathlib import Path
import pandas as pd
from glob import glob
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img, math_img, resample_to_img
import numpy as np
from nipype.interfaces.spm.preprocess import Normalize12, Coregister
from nibabel import Nifti1Image
from os.path import isfile, isdir
import nibabel as nib
from warnings import warn
from os import mkdir
import os
import warnings

def check_for_multiple_match_ask_input(pattern):
    matches = sorted(glob(pattern))
    if len(matches) == 0:
        warnings.warn("There is no match for the pattern --{}--, this subject will be skipped")
        return(None) 
    elif len(matches) == 1:
        return(matches[0])
    else:
        os.system('spd-say "cat companion requires your attention"')
        print("Multiple files match the patterns --{}--".format(pattern))
        for n,m in enumerate(matches):
            print("match -{}- = {}".format(n,m))
        index = input("Please select the correct file by pressing the corresponding number: ")
        return(matches[int(index)])
        
    
def create_csv_lut_for_atlases(atlases_dir, atlases_csv_dir):
    atlases_files = sorted(glob("{}/*.nii".format(atlases_dir)))
    atlases_csv_files = sorted(glob("{}/*".format(atlases_csv_dir)))
    atlases_names = [Path(el).stem for el in atlases_csv_files]
    atlases_dict = {}
    for atlas_file, atlas_name in zip(atlases_csv_files, atlases_names):
        roi_numbers = pd.read_table(atlas_file,  sep = ";").loc[:,"ROIid"].values
        roi_names = pd.read_table(atlas_file,  sep = ";").loc[:,"ROIname"]
        roi_names_array = roi_names.values
        missing_name = np.where(roi_names.isna())[0]
        if missing_name.shape[0] != 0:
            substitute = ["unknown_" + str(el) for el in missing_name]
            roi_names_array[missing_name] = substitute
        valid_roi_names = [el.translate(str.maketrans(" àéèïôùç","_aeeiouc")) for el in roi_names_array]
        atlases_dict[atlas_name] = pd.DataFrame(np.transpose(np.array([roi_numbers, valid_roi_names])),
                    columns = ["ROIid", "ROIname"])
    for file in atlases_files:
        atlas = next((x for x in atlases_names if x in file), False)
        dict_to_write = atlases_dict[atlas]
        dict_to_write.to_csv("{}/{}.txt".format(atlases_dir, str(Path(file).stem)), 
                             header = None, index = None, sep = " ")
        
        
def list_duplicates(seq):
        """Find elements which appear more than one in the list.
        
        This function take as input a list and return any and all elements 
        that appear more than once. Note that what is returned is the element, only once
        independently from the number of times, and not the index
        
        Parameters
        ----------
        seq : a list
        """
        
        seen = set()
        seen_add = seen.add
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set( x for x in seq if x in seen or seen_add(x) )
        # turn the set into a list (as requested)
        return list( seen_twice )
            
        
def remove_duplicate_name_from_columns(names_source, sep = ";"):
        """Modify repeated element in set of string by appending the repetition index to them
        
        This function take as input either the path to a csv file and work on "ROIname" column
        or directly a list of names (supposed to be the names of the ROIs at hand), search for 
        any repeated name and change repeated name by appendinf the repetition index
        
        Parameters
        ----------
        names_source : either an absolute path pointing to a csv or any other tabular file
        or a list of strings
        sep : a string, the separator used in the tabular file
        
        """
        if type(names_source) == str:
            names = pd.read_table(names_source, sep = sep).loc[:,"ROIname"].tolist()
        else:
            names = names_source
        valid_names = [str(el).replace(" ", "_") for el in names]
        duplicates = list_duplicates(valid_names)
        for n, dup in enumerate(duplicates):
            dup_index = np.where(np.isin(valid_names,dup))[0]
            for n, index in enumerate(dup_index):
                valid_names[index] = "{}_{}".format(dup, n)
        return (valid_names)

def find_exisiting_rois_in_subject_atlas(path_to_atlas, path_to_atlas_csv, sep = ";"):
        """Find which ROIs are actaully present in an atlas
        
        When an atlas is masked by e.g. a thresholded grey or white matter image
        some of the ROIs present in the original atlas may be lost. This function find out which 
        ROIs are actually present and return them as a list of string with the name of said ROIs
        
        Parameters
        ----------
        path to atlas : absolute path to the atlas image whose ROIs need to be checked
        path_to_atlas_csv : absolute path (as a string) to the atlas-associated csv. This is a csv or 
        any other tabular structured file that MUST have two columns: ROIid, containing the indexes
        of the ROIs as they appear in the original images and ROIname, containing the names of sais ROIs
        sep: string, the separator used in the csv-associated atlas file
        
        """
        atlas_masked_img = load_img(path_to_atlas)
        atlas_masked_array = np.unique(atlas_masked_img.get_data())
        current_df = pd.read_table(path_to_atlas_csv, sep = sep).loc[:,["ROIid","ROIname"]]
        subject_atlas_specific_name = current_df[current_df["ROIid"].isin(atlas_masked_array[1:])].loc[:,"ROIname"].values.tolist()
        valid_names = remove_duplicate_name_from_columns(subject_atlas_specific_name)
        return (valid_names)
    
def check_and_equalize_affine(path_to_atlas, path_to_image_to_extract):
    """Check if two images have EXACTLY the same affine, if True than it returns the images as they are, in False
        it registers the atlas image onto the image to extract. The two images SHOULD BE already in alignment, 
        as this function is taught to correct for decimal differences in the affine that can create problems
        with the NiftiLabelMasker class
        
    Parameters
    ----------
    path to atlas : absolute path to the atlas image containing the ROIs from which values are to be extracted
    path_to_image_to_extract : absolute path to the image whose values are to be extracted. This image have to be in the same
    space as the atlas
    
    
    """
    atlas_img = load_img(path_to_atlas)
    image_to_extract = load_img(path_to_image_to_extract)
    if np.array_equal(atlas_img.affine, image_to_extract.affine):
        return (atlas_img, image_to_extract)
    else:
        atlas_img = Nifti1Image(atlas_img.get_data(), image_to_extract.affine, header = image_to_extract.header)
    return (atlas_img, image_to_extract)

    
class CatCompanion(object):

     def __init__(self, main_cat_dir):
        self.main_cat_dir = str(Path(main_cat_dir))
        if not(isdir(self.main_cat_dir)):
            raise FileNotFoundError("main_cat_dir must be an existing directory but {} does not exist or is not a directory".format(self.main_cat_dir))
        self.volume_dir = self.main_cat_dir + "/mri"
        if not(isdir(self.volume_dir)):
            warn("No /mri directory found in the main CAT directory")
        self.atlas_dir = self.main_cat_dir + "/mri_atlas"
        if not(isdir(self.atlas_dir)):
            warn("No /mri_atlas directory found in the main CAT directory")
        self.masked_atlas_dir = self.main_cat_dir + "/mri_atlas_masked"
        self.other_modalities_dir = self.main_cat_dir + "/other_modalities"
        if not(isdir(self.other_modalities_dir)):
            warn("No /other_modalities directory found in the main CAT directory")
        self.atlases_csv_dir = self.main_cat_dir + "/atlases_csv"
        if not(isdir(self.atlases_csv_dir)):
            warn("No /atlases_csv directory found in the main CAT directory")
        self.other_modalities_groups = []
        self.n_subjects = len(glob("{}/*.nii".format(self.main_cat_dir)))
        self.text_sep = ","
        
        
        
        

     def send_atlases_to_single_subject_space(self, original_atlases):
        """Send atlas(es) in single subject space using previously calculated spm-like deformation field ("iy_*")
               
        Parameters
        ----------
        deformation_fields_dir : abolute path of the directories were the deformation field(s) are stored
        output_dir = absolute path of the directory where the warped atlas are to be written
        
        The function return number of atlases * number of deformation fields nii file named [atlas]_[subject]
        
        """    
        nrm = Normalize12()
        nrm.inputs.apply_to_files = original_atlases
        def_fields = sorted(glob("{}/iy*".format(self.volume_dir)))
        t1_dir = str(Path(self.volume_dir).parents[0])
        for n, def_field in enumerate(def_fields):
            subject_name = Path(def_field).stem.split("iy_")[1]
            t1_file = check_for_multiple_match_ask_input("{}/*{}*".format(t1_dir, subject_name))
            print("working on subject {}, {}% treated".format(subject_name, round(((n + 1)/len(def_fields))*100, 2)))
            nrm.inputs.deformation_file = def_field
            nrm.inputs.jobtype = "write"
            nrm.inputs.write_interp = 0
            nrm.run()
            for atlas in original_atlases:
                atlas_dir = "/".join(Path(atlas).parts[0:len(Path(atlas).parts) -1])
                atlas_name = Path(atlas).stem.split("[.]")[0]
                res_atlas = resample_to_img(load_img("{}/w{}.nii".format(atlas_dir, atlas_name)), 
                                load_img(t1_file), interpolation = "nearest")
                res_atlas.to_filename("{}/{}_{}.nii".format(self.atlas_dir, atlas_name, subject_name))
                
                
                
    
     def mask_atlases(self, atlases_to_mask, tissue_for_masking, tissue_thr = .1):
        """Mask atlas(es) for the selected tissue type
               
        Parameters
        ----------
        atlases_dir : a string with the absolute path to the directoty where the atlas in single subject space are stored
        atlases_to_mask : a string or a list of strings with the name of the atlas(es) to be treated (with no extension)
        tissue_for_masking : a string in ["gm", "wm", "csf"]
        tissue_thr : threshold value for the tissue type - default at .1
        output_dir :  a string with the absolute path to the directory where the masked atlases should be written
        
        The function return number of subject * number of atlas nii file named [atlas]_masked_[subject]
        
        """
        if not(isdir(self.masked_atlas_dir)):
            mkdir(self.masked_atlas_dir)
        if type(atlases_to_mask) == str:
            atlases_to_mask = [atlases_to_mask]
        if tissue_for_masking == "gm":
            tissue_prefix = "p1"
        elif tissue_for_masking == "wm":
            tissue_prefix = "p2"
        elif tissue_for_masking =="csf":
            tissue_prefix = "p3"
        for atlas_name in atlases_to_mask:
            single_subject_atlases = sorted(glob("{}/{}*nii".format(self.atlas_dir, atlas_name)))
            for n, f in enumerate(single_subject_atlases):
                subject_name = Path(f).stem.split(atlas_name + "_")[1]
                print("working on subject {}, atlas {}, {}% treated".format(subject_name, atlas_name, round(((n+1)/((len(single_subject_atlases) * len(atlases_to_mask))))*100)))
                tissue_dir = str(Path(f).parents[1]) + "/mri"
                atlas = load_img(f)
                tissue_filename = check_for_multiple_match_ask_input("{}/{}{}*".format(tissue_dir, tissue_prefix, subject_name))
                tissue_image = load_img(tissue_filename)
                tissue_image = math_img("i > {}".format(str(tissue_thr)), i = tissue_image)
                tissue_array = tissue_image.get_data()
                atlas_array = atlas.get_data()
                masked_array = tissue_array * atlas_array
                masked_atlas = Nifti1Image(masked_array, tissue_image.affine, header = tissue_image.header)
                masked_atlas.to_filename("{}/{}_masked_{}.nii".format(self.masked_atlas_dir, atlas_name, subject_name))
                
     def send_other_modalities_to_t1(self, source_modality, modalities = None, prefix = None, suffix = None):
        """Register other modalities than t1 (supposed to be in single subject space) to t1
               
        Parameters
        ----------
        source_modality: string, the name of the modality to be used as source for the registration (as it
        appears in the "other_modalities" directory)
        modalities: string or list of strings, the name(s) of the modality(ies) to be registered (as it appears
        in the "other_modalities" directory). Note that if you want to only register the source modality you need to 
        specify once again it here.
        prefix: string, part of the filename that appears before the conventional CAT12 prefix (e.g. a0, p1 etc.) to 
        be discarded when retrieving the subject name (e.g. a0my_prefix_SUBJECTNAME.nii)
        suffix: string, part of the filename that appears after the conventional CAT12 prefix (e.g. a0, p1 etc.) to 
        be discarded when retrieving the subject name (e.g. a0SUBJECTNAME_my_suffix.nii)
        """
        value_to_be_zeroed = [8,7]
        source_modality_dir = "{}/other_modalities/{}".format(self.main_cat_dir, source_modality)
        brain_parcels = sorted(glob("{}/a0*".format(self.main_cat_dir + "/mri")))
        for el in brain_parcels:
            if prefix == None and suffix == None:
                subject = Path(el).stem.split("a0")[1]
            elif not(prefix == None) and suffix == None:
                subject = Path(el).stem.split("a0")[1].split(prefix)[1]
            elif prefix == None and not(suffix == None):
                subject = Path(el).stem.split("a0")[1].split(suffix)[0]
            else:
                subject = Path(el).stem.split("a0")[1].split(prefix)[1].split(suffix)[0]
                
            source = check_for_multiple_match_ask_input("{}/*{}*".format(source_modality_dir, subject))
            if source is None:
                continue
            if source_modality == modalities:
                file_to_check = "{}/r{}.nii".format(Path(source).parent,Path(source).stem)
                if isfile(file_to_check):
                    continue
            if modalities != source_modality:
                images_to_align = []
                for mod in modalities:
                    img_to_append = check_for_multiple_match_ask_input("{}/other_modalities/{}/*{}*".format(self.main_cat_dir, mod, subject))
                    images_to_align.append(img_to_append)
                all_files_to_check = [source] + images_to_align
                all_files_to_check = ["{}/r{}.nii".format(Path(el).parent,Path(el).stem) for el in all_files_to_check]
                if all([isfile(el) for el in  all_files_to_check]):
                    continue
            
            if not isfile("{}/mri/t1_masked_{}.nii".format(self.main_cat_dir, subject)):
                vol_parcel = nib.load(el)
                t1 = check_for_multiple_match_ask_input("{}/*{}*.nii".format(self.main_cat_dir, subject))
                t1 = load_img(t1)
                brain_parcel_array = vol_parcel.get_data()
                brain_parcel_array[np.isin(brain_parcel_array, value_to_be_zeroed)] = 0
                brain_parcel_array[brain_parcel_array > 0] = 1
                t1_array = t1.get_data()
                masked_t1_array = t1_array * brain_parcel_array
                masked_t1 = nib.Nifti1Image(masked_t1_array, t1.affine, header = t1.header)
                masked_t1.to_filename("{}/mri/t1_masked_{}.nii".format(self.main_cat_dir, subject))
            crg = Coregister()
            crg.inputs.target = "{}/mri/t1_masked_{}.nii".format(self.main_cat_dir, subject)
            crg.inputs.source = source
            if modalities != source_modality:
                crg.inputs.apply_to_files = images_to_align
            crg.run()
    
    
     
     def extract_values_from_atlas(self, path_to_atlas, path_to_image_to_extract, subject):
        """Extract values from ROIs in the provided atlas and return this value, together with 
        the name of the subjects, as a list
               
        Parameters
        ----------
        path to atlas : absolute path to the atlas image containing the ROIs from which values are to be extracted
        path_to_image_to_extract : absolute path to the image whose values are to be extracted. This image have to be in the same
        space as the atlas
        subject: string, subject name to be appended as first element of the list
        
        """
        atlas, image_to_extract = check_and_equalize_affine(path_to_atlas, path_to_image_to_extract)
        masker = NiftiLabelsMasker(atlas)
        data = np.squeeze(masker.fit_transform([image_to_extract])).tolist()
        roi_csv = pd.read_csv("{}/{}.csv".format(self.atlases_csv_dir,Path(path_to_atlas).stem.split("_masked")[0]), sep = self.text_sep)
        roi_names = roi_csv.loc[:,"ROIname"].values[roi_csv.loc[:,"ROIid"].isin(masker.labels_)]
        extracted_line = [subject] + data
        cols = ["subject"] + list(roi_names)
        df = pd.DataFrame([extracted_line], columns = cols)
        return (df) 

     
        
    
     def create_csv_with_atlas_values(self, atlases, modalities, image_filename_pattern = "*{}*",  prefix = None, suffix = None):
        """Create a csv file with ROIs values for a set of subjects
        This function requires that the atlas-related csv file(s) be in the main CAT directory
               
        Parameters
        ----------
        atlases: string or list of strings containing the name of the atlas(es) to use
        image_filename_pattern: string, the pattern that the filenames in image_dir follow. It is assumed that this images all
            follow the same pattern. E.G. if the images to be extracted are grey matter from an SPM style segmentation,
            they will all be called p1[rest of the pattern]. In that case the image_filename_pattern would be 'p1{}*'
            A string equal to "*{}*" will select any file with the subject name in it
        modalities: a string or a list of strings, the modality(ies) to extract
        prefix, suffix : string, any prefix or suffix that can be found in the file name (e.g. [atlas_name]_[subject_name]_T1) that you may want to remove 
        in order to get the subject name alone
        
        The function return a csv file (comma separated) whose filename is [atlas]_[modality]_[len(subjects)]
        
        """
        if type(atlases) == str:
            atlases = [atlases]
        if type(modalities) == str:
            modalities = [modalities]
        if prefix == None and suffix == None:
            subjects = [Path(el).stem.split(atlases[0] + "_masked_")[1] for el in sorted(glob("{}/mri_atlas_masked/{}*".format(self.main_cat_dir, atlases[0])))]
        elif not(prefix == None) and suffix == None:
            subjects = [Path(el).stem.split(atlases[0] + "_masked_")[1].split(prefix)[1] for el in sorted(glob("{}/mri_atlas_masked/{}*".format(self.main_cat_dir, atlases[0])))]
        elif prefix == None and not(suffix == None):
            subjects = [Path(el).stem.split(atlases[0] + "_masked_")[1].split(suffix)[0] for el in sorted(glob("{}/mri_atlas_masked/{}*".format(self.main_cat_dir, atlases[0])))]
        else:
            subjects = [Path(el).stem.split(atlases[0] + "_masked_")[1].split(prefix)[1].split(suffix)[0] for el in sorted(glob("{}/mri_atlas_masked/{}*".format(self.main_cat_dir, atlases[0])))]
            
        total_number_of_iteration = len(subjects) * len(atlases) * len(modalities)
        counter = 0
        for atlas in atlases:
            csv_file = "{}/{}.csv".format(self.atlases_csv_dir, atlas)
            all_atlas_rois = remove_duplicate_name_from_columns(csv_file)[1:]
            all_colnames = ["subject"] + all_atlas_rois
            for modality in modalities:
                df = pd.DataFrame(columns = all_colnames)
                for subject in subjects:
                    counter += 1
                    print("working on atlas {} and modalities {}, {}% treated".format(atlas, modality, round((counter/total_number_of_iteration)*100)))
                    atlas_filename  = check_for_multiple_match_ask_input("{}/{}*{}*".format(self.main_cat_dir + "/mri_atlas_masked", atlas, subject))
                    image_to_extract_filename = "{}/other_modalities/{}/{}".format(self.main_cat_dir, modality, image_filename_pattern.format(subject))
                    path_to_image = check_for_multiple_match_ask_input(image_to_extract_filename)
                    if (atlas_filename is None) or (path_to_image is None):
                         continue
                    df_subject = self.extract_values_from_atlas(atlas_filename, path_to_image, subject)
                    df = pd.concat([df, df_subject])
                    df.sort_values(by=["subject"])
                csv_filename = "{}/{}_{}_{}_subjects.csv".format(self.main_cat_dir, atlas, modality, df.shape[0])
                df.to_csv(csv_filename, index = False)
                    
                    
                    
