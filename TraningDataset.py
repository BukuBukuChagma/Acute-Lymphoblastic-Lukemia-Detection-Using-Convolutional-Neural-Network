from skimage import io, color
import numpy as np
import Augmentor
import skimage
import cv2
import os


#directory where we will save our output Before Augmentation
pathofTrain = 'Dataset/orignalTrain'
pathofAugmentation = pathofTrain + '/Before_Augmentation'
pathNotCancerBefore = pathofTrain + '/Before_Augmentation/Not_Cancer'
pathCancerBefore = pathofTrain + '/Before_Augmentation/Cancer' #make folder for each image where all white blood cells in that image will be stored

alreadyExist = 0 #0 mean already exists, 1 means newly created
if not os.path.isdir(pathofAugmentation):
    os.mkdir(pathofAugmentation)
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
    noOfImages = 5
    idx =0 
    for i in range(1,noOfImages):
        print('Run #', i)
        if i<=31:
            if i < 10:
                path = pathofTrain + '/Orignal/Cancer/im00' + str(i) + '_1.jpg'
            else:
                path = pathofTrain + '/Orignal/Cancer/im0' + str(i) + '_1.jpg'
        else:
            if i < 10:
                path = pathofTrain + '/Orignal/Not_Cancer/im00' + str(i) + '_0.jpg'
            else:
                path = pathofTrain + '/Orignal/Not_Cancer/im0' + str(i) + '_0.jpg'
        
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
        
        gray=cv2.cvtColor(result_image,cv2.COLOR_BGR2GRAY)#coverting the grayscale
        #cv2.imwrite('outGray'+str(i)+'.png', gray)
        
        bw_img = cv2.inRange(gray, 80, 100) #converting to binary image/black and white
        #cv2.imwrite('outbinary'+str(i)+'.png', bw_img)
        bitwiseImg1 = cv2.bitwise_and(rgb, rgb, mask=bw_img) #making the orignal image with binary mask
        #cv2.imwrite('outBitwise'+str(i)+'.png', bitwiseImg1)

        #applying erosion to seperate the white blood cells
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 30))
        erosion = cv2.erode(bw_img, kernel)

        #masking the orignal image with eroded image//will be using this to get the bounding box
        bitwiseImg2 = cv2.bitwise_and(rgb, rgb, mask=erosion)
        
        
        #extracting cells
        gray=cv2.cvtColor(bitwiseImg2,cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)

            #doing this to make sure the x,y coordinate after manually dilating remain in image size
            a = 35
            while(True):
                if(x-a <= np.shape(bitwiseImg1)[1]) and (x-a >= 0):
                    x -=a
                    break
                else:
                    a -=1
            a = 35
            while(True):
                if(y-a <= np.shape(bitwiseImg1)[0]) and (y-a >= 0):
                    y -=a
                    break
                else:
                    a -=1
            a = 70
            while(True):
                if(w+a <= np.shape(bitwiseImg1)[1]) and (w+a >= 0):
                    w +=a
                    break
                else:
                    a -=1
            a = 70
            while(True):
                if(h+a <= np.shape(bitwiseImg1)[0]) and (h+a >= 0):
                    h +=a
                    break
                else:
                    a -=1

            #cv2.rectangle(rgb, (x, y), (x + (w), y + (h)), (36,255,12), 2)
            roi=bitwiseImg1[y:y+h,x:x+w]
            if i<=31:
                cv2.imwrite(pathCancerBefore +'/' + str(idx) + '.png', roi)
            else:
                cv2.imwrite(pathNotCancerBefore +'/' + str(idx) + '.png', roi)
            idx += 1
        #cv2.imwrite('outLocalization'+str(i)+'.png', rgb) 

#directory where we will save out final training data output After Augmentation
basdir = os.path.abspath(os.path.dirname(__file__))
pathNotCancerAfter = os.path.join(basdir, 'Dataset', 'processedTrain', 'Not_Cancer')
pathCancerAfter = os.path.join(basdir, 'Dataset', 'processedTrain', 'Cancer')
pathofProcessedTrain = 'Dataset/processedTrain'

alreadyExist = 0 #0 mean already exists, 1 means newly created
if not os.path.isdir(pathofProcessedTrain):
    os.mkdir(pathofProcessedTrain)
    os.mkdir(pathCancerAfter)
    os.mkdir(pathNotCancerAfter)
    alreadyExist += 1
else:
    if not os.path.isdir(pathCancerAfter):
        os.mkdir(pathCancerAfter)
        alreadyExist += 1
    if not os.path.isdir(pathNotCancerAfter):
        os.mkdir(pathNotCancerAfter)
        alreadyExist += 1

if alreadyExist != 0:
    # Augmenting Cancererous Cells
    p = Augmentor.Pipeline(pathCancerBefore, output_directory=pathCancerAfter ,save_format='png')
    p.rotate_random_90(probability=0.5)
    p.flip_random(probability=0.5)
    p.sample(1000)

    # Augmenting Non Cancererous Cells
    p = Augmentor.Pipeline(pathNotCancerBefore, output_directory=pathNotCancerAfter ,save_format='png')
    p.rotate_random_90(probability=0.8)
    p.flip_random(probability=0.8)
    p.sample(2000)


