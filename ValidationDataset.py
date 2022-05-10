from skimage import io, color
import numpy as np
import Augmentor
import skimage
import cv2
import os


#directory where we will save our output Before Cropping
pathoforignalValidation = 'Dataset/orignalValidation'
pathofCropping = pathoforignalValidation + '/Before_Cropping'
pathNotCancerBefore = pathofCropping + '/Not_Cancer'
pathCancerBefore = pathofCropping + '/Cancer' #make folder for each image where all white blood cells in that image will be stored

alreadyExist = 0 #0 mean already exists, 1 means newly created
if not os.path.isdir(pathofCropping):
    os.mkdir(pathofCropping)
    os.mkdir(pathCancerBefore)
    os.mkdir(pathNotCancerBefore)
    alreadyExist += 1
else:   
    if not os.path.isdir(pathCancerBefore):
        os.mkdir(pathCancerBefore)
        alreadyExist += 1
    if not os.path.isdir(pathNotCancerBefore):
        os.mkdir(pathNotCancerBefore)
        alreadyExist += 1

if alreadyExist != 0:
    noOfImages = 183
    idx =0 
    CancerCount, NotCancerCount = 0,0
    for i in range(1,noOfImages):
        print('Run #', i)
        if i<=100:
            if i < 10:
                path = pathoforignalValidation + '/Orignal/im00' + str(i) + '_1.tif' #path to img
            elif i<100:
                path = pathoforignalValidation + '/Orignal/im0' + str(i) + '_1.tif' #path to img
            else:
                path = pathoforignalValidation + '/Orignal/im' + str(i) + '_1.tif'
        else:
            if i < 10:
                path = pathoforignalValidation + '/Orignal/im00' + str(i) + '_0.tif' #path to img
            elif i < 100: 
                path = pathoforignalValidation + '/Orignal/im0' + str(i) + '_0.tif'
            else:
                path = pathoforignalValidation + '/Orignal/im' + str(i) + '_0.tif'
        
        rgb = io.imread(path)
        lab = color.rgb2lab(rgb) #coverting to lab colorspace for kmean clustering
        #cv2.imwrite( 'labImage'+str(i)+'.png',lab)

        #kmean clustering
        singlePrecisionImage = skimage.img_as_float32(lab)
        twoDimage = singlePrecisionImage.reshape((-1, 3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 3
        attempts = 10
        ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((rgb.shape))
        #cv2.imwrite('outClusteredImg' + str(i) + '.png', result_image)
        
        #coverting the img to grayscale
        gray=cv2.cvtColor(result_image,cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('outGray'+str(i)+'.png', gray)
        
        bw_img = cv2.inRange(gray, 89, 95)
        #cv2.imwrite('outbinary'+str(i)+'.png', bw_img)
        
        #making the orignal image with binary image
        bitwiseImg1 = cv2.bitwise_and(rgb, rgb, mask=bw_img)
        #cv2.imwrite('Dataset/validation/Cancer/New/out'+str(i)+'.png', bitwiseImg1)
        
        if i <=100:
            cv2.imwrite(pathCancerBefore + '/' + str(i) + '.png', bitwiseImg1)
            CancerCount +=1
        else:
            cv2.imwrite(pathNotCancerBefore+ '/' + str(i) + '.png', bitwiseImg1)
            NotCancerCount +=1


basdir = os.path.abspath(os.path.dirname(__file__))
pathNotCancerAfter = os.path.join(basdir, 'Dataset', 'processedValidation', 'Not_Cancer')
pathCancerAfter = os.path.join(basdir, 'Dataset', 'processedValidation', 'Cancer')
pathofProcessedValidation = 'Dataset/processedValidation'

alreadyExist = 0 #0 mean already exists, 1 means newly created
if not os.path.isdir(pathofProcessedValidation):
    os.mkdir(pathofProcessedValidation)
    os.mkdir(pathCancerAfter)
    os.mkdir(pathNotCancerAfter)
    alreadyExist +=1
else:
    if not os.path.isdir(pathCancerAfter):
        os.mkdir(pathCancerAfter)
        alreadyExist +=1
    if not os.path.isdir(pathNotCancerAfter):
        os.mkdir(pathNotCancerAfter)
        alreadyExist +=1

if alreadyExist != 0:
    # Augmenting Cancererous Cells
    p = Augmentor.Pipeline(pathCancerBefore, output_directory=pathCancerAfter ,save_format='png')
    p.crop_by_size(probability=1.0, width= 130, height=130)
    p.sample(CancerCount)

    # Cropping Testing Images to our desired Width and Height for better accuracy
    p = Augmentor.Pipeline(pathNotCancerBefore, output_directory=pathNotCancerAfter ,save_format='png')
    p.crop_by_size(probability=1.0, width= 130, height=130)
    p.sample(NotCancerCount)