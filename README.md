# [VISOR visualization](https://epic-kitchens.github.io/VISOR)
This repository is to visualize VISOR dataset (both sparse and dense). VISOR website: (https://epic-kitchens.github.io/VISOR)


## How to use
Open `demo.py` script and edit the paths with your local paths. `demo.py` script will call `folder_of_jsons_to_masks` function from vis.py the function support has those arguments:

`json_files_path`: a path to JSON files you want to visualize.<br /> The path file structure would be like:<br>
-- /path/to/json/files/folder<br>
---- P01_01.json<br>
---- P01_102.json<br>
---- PXX_XXX.json<br>
 
`output_directory`: a path where the output masks/overlay would be generated.<br />
`is_overlay` (optional): this flag determine if you want to generate masks (is_overlay=False) only or overlaid images (is_overlay=True). Default is False.<br />
`rgb_frames` (optional): a path where VISOR images are in case you want to visualize the output as overlaid images.<br /> The path file structure would be like:<br>
-- /path/to/rbg_frames/folder<br>
---- P01_01<br>
------ P01_01_frame_0000xxxxxx.jpg<br>
---- PXX_XXX<br>
------ PXX_XXX_frame_0000xxxxxx.jpg<br>

`output_resolution` (optional): output resolution of the generated masks/overlaid images. Default is Full HD (1920x1080)<br />
`generate_video` (optional): whether you want to create videos from the output images. Default is True

After identifying all arguments, just run `python demo.py` and you should get the visualizations in your `output_directory`.

You'll also get a `data_mapping.csv` file in the `output_directory`, this file maps each colour in the PNG images into the object name.



# About EPIC-KITCHENS VISOR dataset


## Authors
Ahmad Darkhalil* (1) 
Dandan Shan* (2) 
Bin Zhu* (1) 
Jian Ma* (1) 
Amlan Kar (3) 
Richard Higgins (2) 
Sanja Fidler (3) 
David Fouhey (2) 
Dima Damen (1)<br>

1: University of Bristol, United Kingdom <br>
2: University of Michigan, United States <br>
3: University of Toronto, Canada <br>

### Release Date: Aug 2022

## Citing
When using VISOR annotations, kindly reference both the annotations paper (VISOR) and the EPIC-KITCHENS-100 paper as follows:

- Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, Dima Damen (2022). EPIC-KITCHENS VISOR Benchmark
VIdeo Segmentations and Object Relations. Early Access OpenReview.

- Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, Michael Wray (2022). Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100. International Journal of Computer Vision (IJCV) vol 130, pp 33-55.

## Ownership and License

All annotations are owned by the University of Bristol under the research license agreement 2021 - 3107, signed by all parties on Jan 2022.

All files in this dataset are copyright by us and published under the Creative Commons Attribution-NonCommerial 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc/4.0/). This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes.

For commercial licenses, contact the University of Bristol at: uob-epic-kitchens@bristol.ac.uk 

## Dataset Details

This deposit contains manually collected pixel-level segmentations for a subset of videos from the dataset EPIC-KITCHENS-100. These represent 30 recorded hours (train/val) and 6 hours of test (only images released).

This README contains information about annotations format.  Please see [epic-kitchens VISOR website](https://github.com/epic-kitchens/VISOR) for additional details, open challenges and the latest starter code.


## Videos and RGB Images

The original videos can be downloaded from two separate DOIs:
http://dx.doi.org/10.5523/bris.3h91syskeag572hl6tvuovwv4d 
and
http://dx.doi.org/10.5523/bris.2g1n6qdydwa9u22shpxqzp0t8m

However, we also release sparse annotation frames for all splits of VISOR (train/val/test) with the following structure:<br>
sparse <br>
-- train<br>
---- P01<br>
-------P01_01.zip<br>
-- val<br>
-- test<br>

We provide a mapping from these frames to the originally released rgb_frames in EPIC-KITCHENS. This avoids any differences in the frame extraction if any.

## Ground Truth - Sparse Annotations

As explained in Sec 2.1, we provide manually annotated sparse segmentations, at the rate of (roughly) 2 frames per action clip. We use JSON formatting annotations, one per video, with the following fields<br>
*image*<br>
-- `image_path`: the folder and file name in the VISOR file structure. Note that this might not exactly match the EPIC-KITCHENS frame numbers (see Sec frame extraction for correspondences)<br>
-- `name`: image filename<br>
-- `subsequence`: Refer to Sec 2.1 where we define subsequences within videos with consistent set of entities. This field refers to the sequence number and is used for the VOS benchmark<br>
-- `video`: video filename - matches EPIC-KITCHENS-100 video file name<br>
*annotations* [one per mask]<br>
-- `id`: Unique mask ID<br>
-- `name`: open vocabulary entity name<br>
-- `class_id`: key for EPIC-KITCHENS class number reflecting closed vocabulary entity name (see Sec classes for correspondences)<br>
-- `exhaustive`: flag (y/n) to indicate whether all instances of the entity have been segmented. When 'n' is selected, background non-active instances of the same entity are present. We use 'inconclusive' when a consensus amongst annotators could not be found<br>
-- `in_contact_object`: [only for hand and on-hand gloves] this field offers one of several options: the ID of the object that the hand is in contact with in this frame, or hand-not-in-contact when the hand is not touching any mask, none-of-the-above when the hand is touching an object that is not segmented, or inconclusive when a consensus amongst annotators could not be found.<br>
-- `on_which_hand`: [only for gloves] indicating whether the glove is worn on a hand and which hand side is it. <br>
-- `segments`: polygon-based representation of each mask<br>

## Interpolations - Dense Annotations

As explained in Sec 2.4, two consecutive ground truth segmentations are used to interpolate intermediate masks. The interpolations are filtered and only high J&F scored interpolations are provided.

In addition to the flags above, additional flags are available in interpolations:<br>

-- `type`: 1: start/end ground-truth frames, that have been filtered to only include entities that are present at both the start and end frames. These are replicated here. 0: interpolated frames<br>
-- `interpolation`: unique incremental ID for each interpolation in the dataset

## Frame Extraction (frame_mapping.json)

Note that we had to re-extract frames from the dataset due to a change in ffmpeg versions. We provide a file that maps frame numbers between VISOR and publicly released rgb_frames in EPIC-KITCHENS-55 and EPIC-KITCHENS-100.
You can find this at: frame_mapping.json

## Classes (EPIC_100_noun_classes_v2.csv)

As new open-vocabulary entities have been added with the VISOR segmentations, we provide an updated noun classes for EPIC-KITCHENS.
These include all new entities, mapped to the EPIC-KITCHENS-100 classes.
We also add 4 additional classes: left hand, right hand, left glove, right glove - as prior classes didn't separate hand or glove sides. Note that we do not remove the original hand and glove classes for consistency with other benchmarks.
