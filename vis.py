import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import glob
import csv
import pandas as pd
import os
from tqdm import tqdm
import json
import collections, functools, operator
from PIL import Image
from scipy.stats import norm
from numpy import asarray

def generate_masks_for_image(image_name, image_path, masks_info, output_directory,object_keys=None,is_overlay=False, images_root_directory= '.', input_resolution=(1920,1080), output_resolution=(1920,1080)):

    non_empty_objects = []

    img = np.zeros([input_resolution[1],input_resolution[0]],dtype=np.uint8)
    ###masks_info = sorted(masks_info, key=lambda k: k['name'])
    index = 1
    #if (image_path == 'P30_107/Part_008/P30_107_seq_00072/frame_0000038529/frame_0000038529.jpg'): #'P30_107/Part_008/P30_107_seq_00067/frame_0000037246/frame_0000037246.jpg'

    entities = masks_info
    i = 1
    for entity in entities:
        object_annotations = entity["segments"]
        polygons = []
        polygons.append(object_annotations)
        non_empty_objects.append(entity["name"])
        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))
        if object_keys:
            if (entity['name'] in object_keys.keys()):
                #print(ps)
                #print(entity['name'])
                #print(object_keys[entity['name']])
                cv2.fillPoly(img, ps, (object_keys[entity['name']], object_keys[entity['name']], object_keys[entity['name']]))
                #cv2.polylines(img, ps, True, (255,255,255), thickness=1)
        else:
            cv2.fillPoly(img, ps, (i, i, i))
        i += 1
        #visualize(img)
        #if (not np.all(img == 0)):  # image_path.__contains__("P03_120_seq_00064")
    image_name = image_name.replace("jpg", "png")
    if (not np.all(img == 0)):
        image_name = image_name.replace("jpg", "png")
        #print(output_directory + image_name)
        # cv2.imwrite(output_directory+image_name,img)
        image_data = asarray(img)
        if (input_resolution != output_resolution):
            out_image = cv2.resize(image_data, (output_resolution[0],
                                        output_resolution[1]),
                                 interpolation=cv2.INTER_NEAREST)
            out_image = (np.array(out_image)).astype('uint8')
        else:
            out_image = image_data
            
        imwrite_indexed_2(os.path.join(output_directory,image_name), out_image)
        
        if is_overlay:
            image1 = image_name.replace("png", "jpg")
            video = '_'.join(image_name.split('_')[:2])
            image1_overlay = Image.open(os.path.join(images_root_directory, video+'/'+image1))

            if (image1_overlay.size != output_resolution):
                image1_overlay = image1_overlay.resize(output_resolution)
                
            davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
            davis_palette[:104, :] = [[0,0,0],[200, 0, 0], [0, 200, 0],[200, 128, 0], [0, 0, 200], [200, 0, 200], [0, 200, 200], [200, 200, 200],[252,93,82], [160,121,99], [164,188,119], [0,60,29], [75,237,255], [148,169,183], [96,74,207], [255,186,255], [255,218,231], [136,30,23], [231,181,131], [219,226,216], [0,196,107], [0,107,119], [0,125,227], [153,134,227], [91,0,56], [86,0,7], [246,207,195], [87,51,0], [125,131,122], [187,237,218], [46,57,59], [164,191,255], [37,29,57], [144,53,104], [79,53,54], [255,163,128], [255,233,180], [68,100,62], [0,231,199], [0,170,233], [0,20,103], [195,181,219], [148,122,135], [200,128,129], [46,20,10], [86,78,24], [180,255,188], [0,36,33], [0,101,139], [50,60,111], [188,81,205], [168,9,70], [167,91,59], [35,32,0], [0,124,28], [0,156,145], [0,36,57], [0,0,152], [89,12,97], [249,145,183],[255,153,170], [255,153,229], [184,143,204], [208,204,255], [11,0,128], [69,149,230], [82,204,194], [77,255,136], [6,26,0], [92,102,41], [102,85,61], [76,45,0], [229,69,69], [127,38,53], [128,51,108], [41,20,51], [25,16,3], [102,71,71], [77,54,71], [143,122,153], [42,41,51], [4,0,51], [31,54,77], [204,255,251], [51,128,77], [61,153,31], [194,204,143], [255,234,204], [204,119,0], [204,102,102],[64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],[64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],[0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],[0, 64, 128], [128, 64, 128]] # first 90 for the regular colors and the last 14 for objects having more than one segment
    
            a = overlay_semantic_mask(image1_overlay, out_image, colors=davis_palette, alpha=0.2, contour_thickness=1)
            img2 = Image.fromarray(a, 'RGB')
            img2.save(os.path.join(output_directory,image_name))

    	

    
    #imwrite_indexed(output_directory + image_name, img,non_empty_objects)
    




def folder_of_jsons_to_masks(json_files_path,output_directory,is_overlay=False, rgb_frames= '.', output_resolution=(1920,1080),generate_video=True):

    
    if os.path.exists(os.path.join(output_directory,'data_mapping.csv')):
        os.remove(os.path.join(output_directory,'data_mapping.csv'))
        
    for json_file in tqdm(sorted(glob.glob(os.path.join(json_files_path ,'*.json')))):
        if 'interpolation' in os.path.basename(json_file):
            input_resolution =(854,480)
            frame_rate = 50
        else:
            input_resolution =(1920,1080)
            frame_rate = 3
        objects_keys = json_to_masks_new(json_file,output_directory,is_overlay=is_overlay, images_root_directory=rgb_frames,input_resolution=input_resolution, output_resolution=output_resolution)
        if generate_video:
            generate_video_from_images(os.path.join(output_directory,'_'.join(os.path.basename(json_file).split('.')[0].split('_')[:2])), frame_rate)
        data = pd.DataFrame(objects_keys.items(), columns=['object_name', 'unique_index'])
        data['video_id'] = json_file.split('/')[-1].split('.')[0]
        
        data = data[['video_id', 'object_name','unique_index']]
        if not os.path.isfile(os.path.join(output_directory,'data_mapping.csv')):
            data.to_csv(os.path.join(output_directory,'data_mapping.csv'), index=False,header=['video_id','object_name', 'unique_index'])
        else:
            data.to_csv(os.path.join(output_directory,'data_mapping.csv'),mode='a', header=False,index=False)
        



def json_to_masks_new(filename,output_directory,is_overlay=False, images_root_directory= '.',input_resolution=(1920,1080), output_resolution=(1920,1080)):
    os.makedirs(output_directory, exist_ok=True)

    object_keys = {}
    objects = do_stats_stage2_jsons_single_file_new(filename)
    #print('objects: ',objects)
    i = 1
    for key,_ in objects:
        object_keys[key] = i
        i=i+1
    max_count = max(object_keys.values())
    #print(f'unique object count of {filename.split("/")[-1]} is {max_count}')
    
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data["video_annotations"], key=lambda k: k['image']['image_path'])
    # Iterating through the json list
    full_path=""
    for datapoint in data:
        image_name = datapoint["image"]["name"]
        image_path = datapoint["image"]["image_path"]
        masks_info = datapoint["annotations"]
        full_path =os.path.join(output_directory,image_path.split('/')[0]+'/') #until the end of sequence name
        #print(full_path)
        os.makedirs(full_path,exist_ok= True)
        #generate_masks_stage3(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
        #generate_masks_per_seq(image_name, image_path, masks_info, full_path)
        generate_masks_for_image(image_name, image_path, masks_info, full_path,object_keys=object_keys,is_overlay=is_overlay,images_root_directory=images_root_directory,input_resolution=input_resolution, output_resolution=output_resolution) #this is for unique id for each object throughout the video
    return object_keys


def do_stats_stage2_jsons_single_file_new(file):

    total_number_of_images=0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image=[]
    objects=[]
    infile=file
    f = open(infile)
    # returns JSON object as a dictionary
    data = json.load(f)

    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data["video_annotations"], key=lambda k: k['image']['image_path'])

    total_number_of_images = total_number_of_images + len(data)

    # Iterating through the json list
    index = 0
    full_path=""
    prev_seq = ""
    obj_per_image=0
    for datapoint in data:
        obj_per_image=0 # count number of objects per image
        image_name = datapoint["image"]["name"]
        image_path = datapoint["image"]["image_path"]
        masks_info = datapoint["annotations"]
        #generate_masks(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
        entities = masks_info
        for entity in entities: #loop over each object
            object_annotations = entity["segments"]
            if not len(object_annotations) == 0: #if there is annotation for this object, add it
                total_number_of_objects = total_number_of_objects + 1
                objects.append(entity["name"])
                obj_per_image = obj_per_image + 1
        total_number_objects_per_image.append(obj_per_image)


    #print(objects)
    objects_counts = collections.Counter(objects)

    df = pd.DataFrame.from_dict(objects_counts, orient='index').reset_index()
    #print(df)
    #print("Number of sequences: ", (total_number_of_seq))
    #print("Number of images (masks): ", (total_number_of_images))
    #print("Number of unique objects: ", len(set(objects)))

    return objects_counts.most_common()




def generate_video_from_images (input_directory,frame_rate):

    out_file = input_directory
    x = "ffmpeg -y -pattern_type glob -r " + str(frame_rate)+ " -i '" +input_directory + "/*.png' -vcodec mpeg4 -qmin 1 -qscale:v 1 " + out_file + ".mp4"
    os.system(x)

def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):

    d = colors
    colors = colors.ravel()
    a = ann
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    #ann2 = a.convert('RGB')
    ann2 = np.array(a)

    colors = np.asarray(colors, dtype=np.uint8)

    #mask = colors[ann2]
    mask = d[ann2]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, d[obj_id].tolist(),
                             contour_thickness)
    return img




def imwrite_indexed(filename, im,non_empty_objects=None):

    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                             [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                             [0, 64, 128], [128, 64, 128]]
    #davis_palette[:79, :] = [[0,0,0],[255, 0, 0], [255, 255, 0],[255,153,170], [255,153,229], [184,143,204], [208,204,255], [11,0,128], [69,149,230], [82,204,194], [77,255,136], [6,26,0], [92,102,41], [102,85,61], [76,45,0], [229,69,69], [127,38,53], [128,51,108], [41,20,51], [65,61,102], [21,0,255], [0,77,153], [31,77,73], [0,51,17], [64,255,0], [44,51,10], [229,182,115], [0,0,0], [51,0,0], [153,0,26], [26,10,22], [122,61,153], [111,102,204], [230,242,255], [0,127,255], [51,128,121], [204,204,204], [19,77,0], [199,230,46], [153,108,46], [179,125,125], [204,0,0], [77,69,75], [204,61,168], [176,69,230], [57,51,128], [143,173,204], [0,13,26], [0,26,23], [87,102,82], [251,255,229], [133,153,31], [51,34,10], [51,36,36], [255,0,0], [230,207,224], [77,0,57], [71,10,102], [82,69,230], [77,115,153], [161,179,177], [143,179,155], [143,204,122], [48,51,36], [204,196,184], [25,16,3], [102,71,71], [77,54,71], [143,122,153], [42,41,51], [4,0,51], [31,54,77], [204,255,251], [51,128,77], [61,153,31], [194,204,143], [255,234,204], [204,119,0], [204,102,102]]
    #davis_palette[:154, :]= [[227,246,225], [178,155,102], [255,35,255], [0,105,68], [249,239,225], [152,142,255], [82,9,105], [0,184,99], [255,225,226], [81,125,144], [143,116,10], [0,255,128], [145,163,143], [255,193,139], [132,221,13], [0,255,69], [255,221,255], [255,167,141], [130,0,4], [0,61,0], [166,157,144], [255,213,137], [177,0,38], [0,103,0], [227,219,255], [236,250,133], [255,0,69], [0,105,0], [214,232,255], [255,237,135], [127,0,40], [0,141,0], [167,152,185], [71,35,71], [246,0,110], [0,147,0], [71,87,69], [35,49,70], [156,0,73], [0,148,0], [89,81,70], [101,127,64], [122,0,72], [0,191,0], [185,150,144], [123,60,107], [252,0,147], [0,191,0], [245,183,184], [162,90,184], [178,0,108], [114,175,0], [106,120,144], [212,141,103], [241,0,186], [209,255,0], [205,187,255], [103,63,144], [136,0,106], [113,127,0], [146,106,145], [134,58,71], [179,0,183], [54,51,0], [90,77,106], [240,104,227], [134,0,142], [174,160,0], [255,205,183], [153,110,66], [137,0,221], [94,85,0], [255,182,255], [243,122,105], [90,0,179], [255,202,0], [160,205,226], [81,33,38], [0,60,178], [208,147,0], [255,238,181], [118,222,91], [0,97,220], [161,109,0], [175,255,224], [255,100,186], [0,62,140], [116,74,0], [54,47,37], [209,100,255], [0,56,104], [255,160,0], [171,255,179], [255,162,99], [0,125,223], [75,45,0], [180,208,140], [255,97,146], [0,113,180], [133,65,0], [105,75,70], [44,39,105], [0,161,255], [189,92,0], [205,136,185], [183,94,68], [0,69,104], [255,120,0], [70,83,106], [64,90,29], [0,104,142], [86,38,0], [160,104,106], [255,243,80], [0,187,255], [96,29,0], [83,128,105], [255,76,107], [0,53,70], [255,72,0], [163,250,255], [76,99,255], [0,153,182], [156,41,0], [160,161,102], [47,177,142], [0,92,106], [192,0,0], [219,134,145], [218,140,58], [0,223,255], [202,0,0], [184,138,226], [126,66,33], [0,179,183], [205,0,0], [139,154,226], [255,74,66], [0,70,69], [215,0,0], [255,153,186], [248,121,61], [0,229,224], [243,0,0], [152,201,255], [57,51,221], [0,107,105], [255,0,0], [122,108,184], [9,56,36], [0,255,222], [116,172,100], [44,21,141], [0,255,177]]
    #davis_palette[:105, :] = [[0,0,0],[200, 0, 0], [0, 200, 0],[200, 128, 0], [0, 0, 200], [200, 0, 200], [0, 200, 200], [200, 200, 200],[252,93,82], [160,121,99], [164,188,119], [0,60,29], [75,237,255], [148,169,183], [96,74,207], [255,186,255], [255,218,231], [136,30,23], [231,181,131], [219,226,216], [0,196,107], [0,107,119], [0,125,227], [153,134,227], [91,0,56], [86,0,7], [246,207,195], [87,51,0], [125,131,122], [187,237,218], [46,57,59], [164,191,255], [37,29,57], [144,53,104], [79,53,54], [255,163,128], [255,233,180], [68,100,62], [0,231,199], [0,170,233], [0,20,103], [195,181,219], [148,122,135], [200,128,129], [46,20,10], [86,78,24], [180,255,188], [0,36,33], [0,101,139], [50,60,111], [188,81,205], [168,9,70], [167,91,59], [35,32,0], [0,124,28], [0,156,145], [0,36,57], [0,0,152], [89,12,97], [249,145,183],[255,153,170], [255,153,229], [184,143,204], [208,204,255], [11,0,128], [69,149,230], [82,204,194], [77,255,136], [6,26,0], [92,102,41], [102,85,61], [76,45,0], [229,69,69], [127,38,53], [128,51,108], [41,20,51], [25,16,3], [102,71,71], [77,54,71], [143,122,153], [42,41,51], [4,0,51], [31,54,77], [204,255,251], [51,128,77], [61,153,31], [194,204,143], [255,234,204], [204,119,0], [204,102,102],[64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],[64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],[0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],[0, 64, 128], [128, 64, 128],[255,255,255]] # first 90 for the regular colors and the last 14 for objects having more than one segment
    #davis_palette[:104, :] = [[0,0,0],[200, 0, 0], [0, 200, 0],[200, 128, 0], [0, 0, 200], [200, 0, 200], [0, 200, 200], [200, 200, 200],[252,93,82], [160,121,99], [164,188,119], [0,60,29], [75,237,255], [148,169,183], [96,74,207], [255,186,255], [255,218,231], [136,30,23], [231,181,131], [219,226,216], [0,196,107], [0,107,119], [0,125,227], [153,134,227], [91,0,56], [86,0,7], [246,207,195], [87,51,0], [125,131,122], [187,237,218], [46,57,59], [164,191,255], [37,29,57], [144,53,104], [79,53,54], [255,163,128], [255,233,180], [68,100,62], [0,231,199], [0,170,233], [0,20,103], [195,181,219], [148,122,135], [200,128,129], [46,20,10], [86,78,24], [180,255,188], [0,36,33], [0,101,139], [50,60,111], [188,81,205], [168,9,70], [167,91,59], [35,32,0], [0,124,28], [0,156,145], [0,36,57], [0,0,152], [89,12,97], [249,145,183],[255,153,170], [255,153,229], [184,143,204], [208,204,255], [11,0,128], [69,149,230], [82,204,194], [77,255,136], [6,26,0], [92,102,41], [102,85,61], [76,45,0], [229,69,69], [127,38,53], [128,51,108], [41,20,51], [25,16,3], [102,71,71], [77,54,71], [143,122,153], [42,41,51], [4,0,51], [31,54,77], [204,255,251], [51,128,77], [61,153,31], [194,204,143], [255,234,204], [204,119,0], [204,102,102],[64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],[64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],[0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],[0, 64, 128], [128, 64, 128]] # first 90 for the regular colors and the last 14 for objects having more than one segment
    color_palette = davis_palette
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    im = torch.from_numpy(im)
    im = Image.fromarray(im.detach().cpu().squeeze().numpy(), 'P')
    im.putpalette(color_palette.ravel())
    im.save(filename)


def imwrite_indexed_2(filename, im,non_empty_objects=None):

    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    davis_palette[:104, :] = [[0,0,0],[200, 0, 0], [0, 200, 0],[200, 128, 0], [0, 0, 200], [200, 0, 200], [0, 200, 200], [200, 200, 200],[252,93,82], [160,121,99], [164,188,119], [0,60,29], [75,237,255], [148,169,183], [96,74,207], [255,186,255], [255,218,231], [136,30,23], [231,181,131], [219,226,216], [0,196,107], [0,107,119], [0,125,227], [153,134,227], [91,0,56], [86,0,7], [246,207,195], [87,51,0], [125,131,122], [187,237,218], [46,57,59], [164,191,255], [37,29,57], [144,53,104], [79,53,54], [255,163,128], [255,233,180], [68,100,62], [0,231,199], [0,170,233], [0,20,103], [195,181,219], [148,122,135], [200,128,129], [46,20,10], [86,78,24], [180,255,188], [0,36,33], [0,101,139], [50,60,111], [188,81,205], [168,9,70], [167,91,59], [35,32,0], [0,124,28], [0,156,145], [0,36,57], [0,0,152], [89,12,97], [249,145,183],[255,153,170], [255,153,229], [184,143,204], [208,204,255], [11,0,128], [69,149,230], [82,204,194], [77,255,136], [6,26,0], [92,102,41], [102,85,61], [76,45,0], [229,69,69], [127,38,53], [128,51,108], [41,20,51], [25,16,3], [102,71,71], [77,54,71], [143,122,153], [42,41,51], [4,0,51], [31,54,77], [204,255,251], [51,128,77], [61,153,31], [194,204,143], [255,234,204], [204,119,0], [204,102,102],[64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],[64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],[0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],[0, 64, 128], [128, 64, 128]] # first 90 for the regular colors and the last 14 for objects having more than one segment
    color_palette = davis_palette
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    im = torch.from_numpy(im)
    im = Image.fromarray(im.detach().cpu().squeeze().numpy(), 'P')
    im.putpalette(color_palette.ravel())
    im.save(filename)
