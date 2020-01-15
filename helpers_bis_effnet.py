import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

#Image manipulation
from PIL import Image
from PIL import ImageFilter

#Librairies used in astronomical data processing
from astropy.visualization import LogStretch, MinMaxInterval,ImageNormalize
from astropy.convolution import Gaussian2DKernel , convolve
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import LogStretch, MinMaxInterval,ImageNormalize

# ==============================================================================
# Data Loading
# ==============================================================================
def load_data (path, extensions ,label_name, load_length , IR  , test = False, start = 0):

    print('Data loading :')
    file_n=load_csv_data(path+label_name,1)
    filelist_y = str(np.zeros(len(file_n)))
    filelist_j = str(np.zeros(len(file_n)))		#Read the file number and initialize
    filelist_h = str(np.zeros(len(file_n)))		#empty string lists
    filelist = str(np.zeros(len(file_n)))

    if IR == 'IR':		#Load the deeper infrared channel on its own

        transform =  MinMaxInterval()
        #Fill the list with the files path and name
        filelist_h = [path + 'EUC_H/' + 'imageEUC_H-' + str(int(file_n[x])) + extensions for x in np.arange(len(file_n))]


        Images = np.empty( ( load_length,200,200 ) ,dtype=np.float32)

        for k in range(load_length):
            if k%300==0:
                print(str(k) + ', loading file ' + str(k+start))
			#Interpolate the IR images to match the VIS resolution
			#and load the data
            Im_h = np.array(fits.getdata( filelist_h[k+start]))
            Im_h = transform(Im_h)
            Im_h= np.array( cv2.resize(Im_h, dsize=(200, 200))  )

            Images[k,:,:] = Im_h

    elif IR == 'mid':	#Load mid IR channels to combine them into one image

        transform =  MinMaxInterval()

        filelist_y = [path + 'EUC_Y/' + 'imageEUC_Y-' + str(int(file_n[x])) + extensions for x in np.arange(len(file_n))]
        filelist_j = [path + 'EUC_J/' + 'imageEUC_J-' + str(int(file_n[x])) + extensions for x in np.arange(len(file_n))]

        Images = np.empty( ( load_length,200,200 ) ,dtype=np.float32)

        for k in range(load_length):
            if k%300==0:
                print(str(k) + ', loading file ' + str(k+start))

            Im_y = np.array(fits.getdata( filelist_y[k+start]))
            Im_y = transform(Im_y)
            Im_y= np.array( cv2.resize(Im_y, dsize=(200, 200))  )

            Im_j = np.array(fits.getdata( filelist_j[k+start]))
            Im_j = transform(Im_j)
            Im_j= np.array( cv2.resize(Im_j, dsize=(200, 200))  )

            Images[k,:,:] = 0.5 * Im_y + 0.5*Im_j

    else:
		#Load the visible images for the third channel
        filelist = [path + 'EUC_VIS/' + 'imageEUC_VIS-' + str(int(file_n[x])) + extensions for x in np.arange(len(file_n))]

        Images = np.empty( ( load_length,200,200 ) ,dtype=np.float32)

        for k in range(load_length):
            if k%1000==0:
                print(str(k) + ', loading file ' + str(k+start))
            Images[k,:,:] = np.array( fits.getdata( filelist[k+start    ] ) )

    print('Done loading data')

    return Images


# ==============================================================================
# Method to open and read the catalogue
# ==============================================================================

def load_csv_data(data_path, col ):
	
    #Loads data for a certain column
    x = np.genfromtxt(data_path, delimiter=",")
    input_data = x[1:, col]

    return input_data

# ==============================================================================
# Image combination over 3 channels
# ==============================================================================

def Img_combine_3(Vis_set , mid_set ,IR_set):

    Images = np.empty( (Vis_set.shape[0],200,200,3) ,dtype=np.float32)

    for k in range(Vis_set.shape[0]):
        Images[k,:,:,0] = Vis_set[k,:,:]
        Images[k,:,:,1] = mid_set[k,:,:]
        Images[k,:,:,2] = IR_set[k,:,:]

    return Images

# ==============================================================================
# Generate more background images by loading further into the set
# ==============================================================================
def background_gen_extended_3(Image_set, Labels, path , load_length, rate_extension):

    file_n=load_csv_data(path+'image_catalog2.0train.csv',1)
    filelist_y = str(np.zeros(len(file_n)))
    filelist_j = str(np.zeros(len(file_n)))		#Initiate the file lists
    filelist_h = str(np.zeros(len(file_n)))
    filelist = str(np.zeros(len(file_n)))
    
    Images_BG = Image_set[Labels == 0]
    temp = Images_BG.shape[0]
    Images_BG_temp_vis = np.empty((temp*rate_extension,200,200))
    Images_BG_temp_mid = np.empty((temp*rate_extension,200,200))	#Initiate the 	
    Images_BG_temp_IR = np.empty((temp*rate_extension,200,200))		#array for the images

    transform =  MinMaxInterval()
	
	#Load the image path and name
    filelist = [path + 'EUC_VIS/' + 'imageEUC_VIS-' + str(int(file_n[x])) + '.fits' for x in np.arange(len(file_n))]
    filelist_y = [path + 'EUC_Y/' + 'imageEUC_Y-' + str(int(file_n[x])) + '.fits' for x in np.arange(len(file_n))]
    filelist_j = [path + 'EUC_J/' + 'imageEUC_J-' + str(int(file_n[x])) + '.fits' for x in np.arange(len(file_n))]
    filelist_h = [path + 'EUC_H/' + 'imageEUC_H-' + str(int(file_n[x])) + '.fits' for x in np.arange(len(file_n))]
	
	#Load the label of the images
    Labels_all = load_csv_data(path+'image_catalog2.0train.csv',26)

    completion = 0	#Indicates the generation completion
    k = 0			#Indicate the number of file checked
	
	#Iterate while generation is incomplete
    while completion < temp*rate_extension:

        if Labels_all[k+load_length]==0:
			
            if completion%300==0:
				
				#Print the state of the generation
                print(str(completion) + ', generating file ' + str(k+load_length))
                
            #When a 0 is found, load the file    
            Im_y = np.array(fits.getdata( filelist_y[k+load_length]))
            Im_y = transform(Im_y)
            Im_y= np.array( cv2.resize(Im_y, dsize=(200, 200))  )

            Im_j = np.array(fits.getdata( filelist_j[k+load_length]))
            Im_j = transform(Im_j)
            Im_j= np.array( cv2.resize(Im_j, dsize=(200, 200))  )

            Im_h = np.array(fits.getdata( filelist_h[k+load_length]))
            Im_h = transform(Im_h)
            Im_h= np.array( cv2.resize(Im_h, dsize=(200, 200))  )

            Images_BG_temp_IR[completion,:,:] =  Im_h
            Images_BG_temp_mid[completion,:,:] =  0.5*Im_y+ 0.5* Im_j
            Images_BG_temp_vis[completion,:,:] = np.array( fits.getdata( filelist[k+load_length   ] ) )
            completion +=1
        k+=1
	#Preprocess and log stretch of the new images
    Images_BG_temp_vis = logarithmic_scale(Images_BG_temp_vis,Images_BG_temp_vis.shape[0])
    Images_BG_temp_mid = logarithmic_scale(Images_BG_temp_mid,Images_BG_temp_IR.shape[0])
    Images_BG_temp_IR = logarithmic_scale(Images_BG_temp_IR,Images_BG_temp_IR.shape[0])
    Images_BG_temp_vis = preprocessing(Images_BG_temp_vis,Images_BG_temp_vis.shape[0])
    Images_BG_temp_mid = preprocessing(Images_BG_temp_mid,Images_BG_temp_IR.shape[0])
    Images_BG_temp_IR = preprocessing(Images_BG_temp_IR,Images_BG_temp_IR.shape[0])
	
	#Combine the channels of the new images
    Images_BG_temp = Img_combine_3(Images_BG_temp_vis,Images_BG_temp_mid,Images_BG_temp_IR)
    Labels_temp = np.zeros(temp*rate_extension)

    return Images_BG_temp , Labels_temp

# ==============================================================================
# Generate new background by flipping and mirroring the images
# ==============================================================================

def background_images_gen (Image_set, Labels):
	
    Images_BG = Image_set[Labels == 0]
    temp = Images_BG.shape[0]
    Images_BG_temp = np.empty((temp*7,200,200,3))
	
	#For every image, apply all the possible combination of
	#mirroring and flipping
    for i in range(temp):

        Images_BG_temp[7*i,:,:,:] = cv2.rotate(Images_BG[i,:,:,:], cv2.ROTATE_90_CLOCKWISE)
        Images_BG_temp[7*i+1,:,:,:] = cv2.rotate(Images_BG[i,:,:,:], cv2.ROTATE_180)
        Images_BG_temp[7*i+2,:,:,:] = cv2.rotate(Images_BG[i,:,:,:], cv2.ROTATE_90_COUNTERCLOCKWISE)
        Images_BG_temp[7*i+3,:,:,:] = cv2.flip(cv2.rotate(Images_BG[i,:,:,:], cv2.ROTATE_90_CLOCKWISE) , 0)
        Images_BG_temp[7*i+4,:,:,:] = cv2.flip(cv2.rotate(Images_BG[i,:,:,:], cv2.ROTATE_180) , 0)
        Images_BG_temp[7*i+5,:,:,:] = cv2.flip(cv2.rotate(Images_BG[i,:,:,:], cv2.ROTATE_90_COUNTERCLOCKWISE) , 0)
        Images_BG_temp[7*i+6,:,:,:] = cv2.flip(Images_BG[i,:,:,:], 0)

    Labels_temp = np.zeros(temp*7, dtype=int)

    return Images_BG_temp , Labels_temp


def logarithmic_scale (Images, Images_load_length):

    transform = LogStretch() + MinMaxInterval()  # def of transformation

    print('Logarithmic strech :')

    for k in range(Images_load_length):
        if k%1000==0:
            print(k)
        Images[k,:,:] = transform(Images[k,:,:])

    print('Logarithmic data strech done')

    return Images

# ==============================================================================
# Image plotting
# ==============================================================================

def plot_images_test (train_images,norm,N_im) :


    plt.imshow(train_images[N_im],origin='lower', norm=norm)
    plt.show()


# ==============================================================================
# Multiple image plotting
# ==============================================================================

def plot_images_test_mosaic (train_images, norm, start, predictions = [20]) :
    if len(predictions) == 1:
        predictions = np.ones(train_images.shape[0])

    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[start+i,:,:],origin='lower', norm=norm)
        plt.title("1" if predictions[start+i] == 1 else "0")
    plt.show()

# ==============================================================================
# Preprocessing of the images
# ==============================================================================

def preprocessing(Images, Images_load_length):
	
    centered_data = np.zeros(200*200*Images_load_length).reshape(Images_load_length,200,200)
    std_data = np.zeros(200*200*Images_load_length).reshape(Images_load_length,200,200)
    
    #For each image, simply substract the mean and divide by its variance
    for k in range(Images_load_length):
        centered_data[k,:,:] = Images[k,:,:] - np.mean(Images[k,:,:])
        std_data[k,:,:] = centered_data[k,:,:] / np.std(centered_data[k,:,:])

    return std_data
