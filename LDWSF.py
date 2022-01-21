import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
def display(image,title='Image',YUV=False):
    if YUV:
        rows = 2
        columns = 2 
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image[:, :, 0])
        plt.title('Y Channel')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(image[:, :, 1])
        plt.title('U Channel')
        fig.add_subplot(rows, columns, 3)
        plt.imshow(image[:, :, 2])
        plt.title('V Channel')
        fig.add_subplot(rows, columns, 4)
        plt.imshow(image)
        plt.title('YUV')
        plt.show()
        exit(0)
    else:
        cv.imshow(title,image)
def mask(image,):
    D=650 #Car ka samne wala hisa ka length
    imshape = image.shape
    mask = np.ones_like(image)
    ignore_mask_color = (255,0,0)  
    # vertices = np.array([[(645,520),(0, D), (1280, D)]])
    vertices = np.array([[(0,900),(1916,900),(1140,595)]])
    mask=cv.fillPoly(mask,vertices, ignore_mask_color)
    masked_edges = cv.bitwise_and(image, mask)
    # display(masked_edges,"Mask Applied to Image")
    return masked_edges




def lDWSF(image,YUV=False):
    if YUV:
        image=cv.cvtColor(frame,cv.COLOR_YUV2BGR)
        image=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # ret = cv.mean(cv.mean(image))[0]*(4/3)
    # if ret>99:
    #     print('Waiting...')
    #     cv.waitKey(1000)
    # display(image)

    # image=cv.equalizeHist(image)
    th, image = cv.threshold(image,190,255, cv.THRESH_TOZERO)
    # display(image,"Global Thresh")
    # image=cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 7, 11)
    # image=cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 7, 11)
    kernel1 = np.ones((2,2),np.uint8)
    # image = cv.erode(image,kernel1,iterations = 1)
    kernel2 = np.ones((2,2),np.uint8)
    # opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel2)
    # kernel = np.ones((2,2),np.uint8)
    # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel2)
    image = cv.dilate(image,kernel1,iterations = 1)
    # display(image,"Local Thresh")

    # image = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    # image = cv.convertScaleAbs(image);

    image = cv.Canny(image,50,150,edges=4)
    # display(image,'Canny')
    # image=cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 7, 11)
    # image=cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 7, 11)
    # ret, image=cv.threshold(image,thresh=100,maxval=255,type=cv.THRESH_BINARY)
    # kernel = np.ones((5,5),np.uint8)
    # image = cv.erode(image,kernel,iterations = 1)
    # image = cv.dilate(image,kernel,iterations = 2)
    # for row in range(image.shape[0]):
        # for col in range(image.shape[1]):
            # print(image[row][col],end='')
        # print()

    # exit(0)
    # image=mask(image)
  
    # mask(image)
    
    lines = cv.HoughLinesP(image,1,np.pi/180,80,minLineLength=100,maxLineGap=70)
    image=cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    line_image = np.copy(image)     
    for line in lines:
        x1,y1,x2,y2 = line[0]
        # if y2-y1>30 and (x1-x2)<50:
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    
    # a=[]
    # b=[]
    # c=[]
    # d=[]
    # # # Iterate over the output "lines" and draw lines on a blank image
    # for line in lines:
    #     for x1,y1,x2,y2 in line:        #taking points
    #         a.append(x1)
    #         b.append(x2)
    #         c.append(y1)
    #         d.append(y2)
    #         # cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    # # plt.imshow(line_image)
    # a=int(np.median(a))
    # b=int(np.median(b))
    # c=int(np.median(c))
    # d=int(np.median(d))
    # cv.line(line_image,(a,c),(b,d),(255,0,0),10)       #right lane
    # am=a
    # bm=b
    # a=[]
    # b=[]
    # c=[]
    # d=[]
    # for line in lines:                      
    #     for x1,y1,x2,y2 in line:
    #         if x1<=am and x2<=bm:                       #taking points
    #             a.append(x1)
    #             b.append(x2)
    #             c.append(y1)
    #             d.append(y2)
    # a=int(np.median(a))
    # b=int(np.median(b))
    # c=int(np.median(c))
    # d=int(np.median(d))           

    # cv.line(line_image,(a,c),(b,d),(255,0,0),10) 

    # # image=cv.line(image,(660,495),(0,600),color=(0,0,255))
    # # image=cv.line(image,(660,495),(1280,600),color=(0,0,255))

    return image

def preFiltering(image):
    # To preprocess the image with differnt algorithms
    # Gaussian Blur to remove the noise
    # Gamma Equilization 
    # Histogram filter to equilize the lights

    smoothenImage = cv.GaussianBlur(image,(5,5),0.1)
    gamma=0.9
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gammaCorrectedImage = cv.LUT(smoothenImage, lookUpTable)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(40,40))
    histEqiImage =clahe.apply(gammaCorrectedImage)

    return histEqiImage,gammaCorrectedImage,smoothenImage


def colorTest(image):
    # image=cv.cvtColor(image,cv.COLOR_RGB2HSV)
    image=cv.cvtColor(image,cv.COLOR_RGB2YUV)
    image=cv.cvtColor(image,cv.COLOR_RGB2HLS)
    # display(image[:,:,0],"h")
    # display(image[:,:,1],"l")
    display(image[:,:,2]/255,"s")
    # print(image[:,:,2])
    return image
if __name__ == "__main__":
    inpt=int(sys.argv[1])
    dataDir='/Users/alpha/Downloads/dataset/08/image_0/'
    videos={1:'/Users/alpha/Downloads/twitchslam-master/videos/test_countryroad_reverse.mp4',2:'test.mp4',3:"/Users/alpha/Downloads/twitchslam-master/videos/test_countryroad.mp4"
                ,4:"/Users/alpha/Downloads/twitchslam-master/videos/test_nyc.mp4",5:'/Users/alpha/Downloads/twitchslam-master/videos/test_ohio.mp4',6:'/Users/alpha/Downloads/tt.mp4'
                ,7:'/Users/alpha/Downloads/aa.mp4'}
    extension='*.png'
    leftDir='image_0'
    filepaths=glob.glob(dataDir+extension)
    filepaths.sort()
    image=[]
    count=0
    # for filepath in filepaths:
    cap = cv.VideoCapture(videos[inpt])
    while(cap.isOpened()):
        ret,frame = cap.read()
        # ret=True
        # frame=cv.imread(filepath)
        if ret == True:
                count+=1
                # image=cv.imread(filepath)
                # display(frame,'orignal')
                # plt.imshow(frame)
                # plt.show()
                # exit(0)
                # image=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
                # image=cv.cvtColor(frame,cv.COLOR_BGR2YUV)
                # image=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                image=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                # display(image)
                image,_,_=preFiltering(image)
                # display(image,'Processed')
                # image=lDWSF(image[:,:,2])
                # image=colorTest(image)
                # image=mask(image)
                # image=image[:,:,2]
                image=image[400:,:]
                image=lDWSF(image)
                # image,_,_=preFiltering(image)
                display(image,'Detected')
                # plt.imshow(image)
                # plt.show()
                # exit(1)

                if cv.waitKey(30)==27:
                    cv.i
                    cv.destroyAllWindows()
                    break
        else:
            break
    print('Processed images: ',count)
