import tqdm
import re
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
os.environ['OPENCV_IO_ENABLE_JASPER']= 'True'
import cv2


import fitz # PyMuPDF
import io
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

import pandas as pd


def pdf_to_img(filename):
    file_path=''
    # open file
    with fitz.open(file_path+filename) as my_pdf_file:

        #loop through every page
        imgs = []
        for page_number in range(1, len(my_pdf_file)+1):

            # acess individual page
            page = my_pdf_file[page_number-1]

            # accesses all images of the page
            images = page.getImageList()

            # check if images are there
            '''
            if images:
                print(f"There are {len(images)} image/s on page number {page_number}[+]")
            else:
                print(f"There are No image/s on page number {page_number}[!]")
            '''
            # loop through all images present in the page
            for image_number, image in enumerate(page.getImageList(), start=1):

                #access image xerf
                xref_value = image[0]

                #extract image information
                base_image = my_pdf_file.extractImage(xref_value)

                # access the image itself
                image_bytes = base_image["image"]

                #get image extension
                ext = base_image["ext"]

                #load image
                image = Image.open(io.BytesIO(image_bytes))

                #save image locally
                #image.save(open(image_path+filename+f"Page{page_number}Image{image_number}.{ext}", "wb"))
                
                numpy_image=np.array(image)  

                # convert to a openCV2 image and convert from RGB to BGR format
                opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                imgs.append(numpy_image)
            
        return imgs
#display(*imgs)


def template_load(template_dir) : 
    from os import listdir
    from os.path import isfile, join
    template_filenamelist = [f for f in listdir(template_dir) if isfile(join(template_dir, f))]
    template_img = []
    for template_name in tqdm.tqdm(template_filenamelist):
        fullname = template_dir + '/'+template_name
        img = cv2.imread(fullname,0)
        template_img.append(img)
    return template_filenamelist, template_img
    
    
    
#pick one of method ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
method_name = 'cv2.TM_SQDIFF'
method = eval(method_name)

paper_info_df =  pd.read_csv("Jilin University.csv", encoding='cp949')

def get_original_paper_name(name) : 
    return int(name.split('/')[-1].split('.')[0])

def parse_paper_num(name) :
    namestr = ''.join( x for x in name[:name.find('Page')] if x.isdigit())
    return int(namestr)

def match_image_with_template(template_filenamelist, template_img, select_paper_img, paper_name, N=5, score_limit=180000):
    match_info_dict={}
    for i, (template_name, template) in enumerate(tqdm.tqdm(list(zip(template_filenamelist, template_img)))) : 
        
        if parse_paper_num(template_name) == paper_name : continue
        
        if select_paper_img.shape[0] < template.shape[0] or select_paper_img.shape[1] < template.shape[1] : continue
        res = cv2.matchTemplate(select_paper_img,template,method)
        min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            score = min_val
            if score <=  score_limit: 
                match_info_dict[i]=(template_name,score,top_left)
        else:
            top_left = max_loc
            score = max_val
            if score >= score_limit:
                match_info_dict[i]=(template_name,score,top_left)
        
        #score 순으로 정렬합니다. 메소드마다 오름차순/내림차순이 달라 순서를 맞춤
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        sorted_by_score_info = sorted(tuple(match_info_dict.items()), key=lambda x : x[1][1], reverse=False)
    else :
        sorted_by_score_info = sorted(tuple(match_info_dict.items()), key=lambda x : x[1][1], reverse=True)

    #top N개를 표시합니다
    min_show = N

    fig_list = []
    for select_index, (top_score_check_img_name, top_score, top_left) in sorted_by_score_info[:min_show]:     

        top_score_check_img = select_paper_img.copy()
        template = template_img[select_index]
        w,h = template.shape[::-1]
        bottom_right =  (top_left[0]+w,top_left[1]+h)
        #cv2.rectangle(top_score_check_img,top_left,bottom_right,(0,255,0),5)


        fig = plt.figure( figsize=(6,2)) 

        ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
        ax1.add_patch(
         patches.Rectangle(
            top_left,
            w,
            h,
            edgecolor = 'blue',
            facecolor = 'red',
            fill=False
         ) )
        ax1.imshow(top_score_check_img,cmap='gray'), ax1.set_title("input paper", fontsize=8),ax1.set_yticks([]),ax1.set_xticks([])
        ax2.imshow(top_score_check_img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w],cmap='gray'), ax2.set_title("input paper\n[zoom detected western blot]", fontsize=8),ax2.set_yticks([]),ax2.set_xticks([])
        ax2.spines['bottom'].set_color('blue'), ax2.spines['top'].set_color('blue'), ax2.spines['left'].set_color('blue'), ax2.spines['right'].set_color('blue')
        ax3.imshow(template,cmap='gray'), ax3.set_title("papermills\nwesternblot",fontsize=8),ax3.set_yticks([]),ax3.set_xticks([])
        ax3.spines['bottom'].set_color('red'), ax3.spines['top'].set_color('red'), ax3.spines['left'].set_color('red'), ax3.spines['right'].set_color('red')

        
        #plt.show()
        
        infoPage = top_score_check_img_name[top_score_check_img_name.find('Page'):top_score_check_img_name.find('Image')]
        infoImage = top_score_check_img_name[top_score_check_img_name.find('Image'):top_score_check_img_name.find('.')]
        infoXY = top_score_check_img_name[top_score_check_img_name.find('(')+1:top_score_check_img_name.find(')')]
        
        paper_num = parse_paper_num(top_score_check_img_name)
        #''.join( x for x in top_score_check_img_name[:top_score_check_img_name.find('Page')] if x.isdigit())
        paper_info = paper_info_df.iloc[int(paper_num)-2]
        name, journal, year, autors, email = paper_info[9], paper_info[10], paper_info[11], paper_info[22], paper_info[23]
        info_str = "%s(%d). %s. \"%s.\" - %s %s [%s] western blot" % (journal, int(year), autors, name, infoPage, infoImage, infoXY)
        
        fig_list.append((fig, info_str))
        #print(top_score_check_img_name)
    
    if fig_list == [] : 
        fig = plt.figure( figsize=(6,2)) 

        ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)

        ax1.imshow(select_paper_img.copy(),cmap='gray'), ax1.set_title(paper_name, fontsize=8),ax1.set_yticks([]),ax1.set_xticks([])
        ax2.text(0, 0, "no matched", fontsize=10), ax2.set_title("[no matched]", fontsize=8),ax2.set_yticks([]),ax2.set_xticks([])
        ax2.spines['bottom'].set_color('blue'), ax2.spines['top'].set_color('blue'), ax2.spines['left'].set_color('blue'), ax2.spines['right'].set_color('blue')
        ax3.text(0, 0, "no matched", fontsize=10), ax3.set_title("[no matched]", fontsize=8),ax3.set_yticks([]),ax3.set_xticks([])
        ax3.spines['bottom'].set_color('red'), ax3.spines['top'].set_color('red'), ax3.spines['left'].set_color('red'), ax3.spines['right'].set_color('red')
        fig_list.append((fig, "no matched"))

    return fig_list
