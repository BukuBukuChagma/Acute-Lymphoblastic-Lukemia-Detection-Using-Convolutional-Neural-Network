from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from skimage import io, color
import numpy as np
import skimage
import cv2
import os

def Predict_Image(model, img):
        img = cv2.resize(img, dsize=(130, 130), interpolation=cv2.INTER_AREA)#because neural networks are trained on this size
        img = img/255
        X = np.expand_dims(img, axis=0)
        val = model.predict(X)
        if val < 0.5:
            return True
        else:
            return False

def mainPredict(dir):
    count = 0
    modelpath = "Model/"
    model = load_model(modelpath, compile = True)
    i = 0
    for path in os.listdir(dir):
        rgb = io.imread(dir + '/' + path)
        lab = color.rgb2lab(rgb) #coverting to lab colorspace for kmean clustering
        #cv2.imwrite( 'labImage'+str(i)+'.png',lab)

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
        #cv2.imwrite('outClusteredImg.png', result_image)
        
        #coverting the img to binary
        gray=cv2.cvtColor(result_image,cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('outGray'+str(i)+'.png', gray)
        bw_img = cv2.inRange(gray, 80, 100)
        #cv2.imwrite('outbinary.png', bw_img)
        
        #making the orignal image with threshold otus image
        bitwiseImg1 = cv2.bitwise_and(rgb, rgb, mask=bw_img)
        #cv2.imwrite('outBitwise.png', bitwiseImg1)


        #applying erosion to seperate the white blood cells
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        erosion = cv2.erode(bw_img, kernel)

        #masking the orignal image with eroded image//will be using this to get the bounding box
        bitwiseImg2 = cv2.bitwise_and(rgb, rgb, mask=erosion)
        #cv2.imwrite('outErosion.png', bitwiseImg2)
        
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
            a = 50
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

            x1, y1 = x, y
            roi=bitwiseImg1[y:y+h,x:x+w]
            if Predict_Image(model, roi):
                if(w > 80 and h > 80):
                    cv2.rectangle(rgb, (x1, y1), (x1 + (w+50), y1 + (h+50)), (36,255,12), 6)
                count +=1
        os.remove(dir + '/' + path)
        if count ==0:
            result = "Congrats! You Don't Have Cancer"
            cv2.imwrite('static/Displayed/result_img.png', rgb)
            return result
        else:
            result = "Alas! You Have Cancer."
            cv2.imwrite('static/Displayed/result_img.png', rgb)
            return result
