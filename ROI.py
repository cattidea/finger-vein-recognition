import numpy as np
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os

'''loG边缘检测'''
def edge_detectio(grayImage):
    laplacian = cv2.Laplacian(grayImage,cv2.CV_64F)
    absX= cv2.convertScaleAbs(laplacian)
    absX= cv2.GaussianBlur(absX,(3,3),100)
    h, w=absX.shape
    # cv2.imwrite('pictures/2.jpg',absX1)
    # plt.imshow(absX,cmap='gray')
    # # plt.figure()
    # plt.imshow(absX,cmap='gray')
    # plt.show()
    return absX, h, w


'''取边缘检测后边界的点'''
def get_bound_piont(absX):
    h, w=absX.shape
    ret,absX=cv2.threshold(absX, 0, 255, cv2.THRESH_OTSU)
    plt.imshow(absX,cmap='gray')
    # plt.show()
    t, t1 =0, h-1
    bound_up_x, bound_down_x, bound_up_y, bound_down_y = [], [], [], []
    for i in range (w):
        y_min, y_max = 0, h-1
        for j in range (h):
            if absX[j, i] > ret:
                if j < h // 2 and j > y_min:
                    y_min = j
                if j > h // 2 and j < y_max:
                    y_max = j
                    break
        if t < y_min:
            t=y_min
        if t1 > y_max:
            t1=y_max
        if y_max != h-1:
            bound_up_x.append(i)
            bound_up_y.append(y_max)
        if y_min != 0:
            bound_down_x.append(i)
            bound_down_y.append(y_min)
    t2=(t+t1)/2.
    bound_up_x = np.array(bound_up_x)
    bound_down_x = np.array(bound_down_x)
    bound_up_y = np.array(bound_up_y)
    bound_down_y = np.array(bound_down_y)
    return bound_up_x, bound_down_x, bound_up_y, bound_down_y

def fun2ploy(x,n):
    '''
    数据转化为[x^0,x^1,x^2,...x^n]
    首列变1
    '''
    lens = len(x)
    X = np.ones([1,lens])
    for i in range(1,n):
        X = np.vstack((X,np.power(x,i)))#按行堆叠
    return X


def leastseq_byploy(x,y,ploy_dim):
    '''
    最小二乘求解
    '''
    #散点图
    # plt.scatter(x,y,color="r",marker='o',s = 50)

    X = fun2ploy(x,ploy_dim)
    #直接求解
    Xt = X.transpose()#转置变成列向量
    XXt=X.dot(Xt)#矩阵乘
    XXtInv = np.linalg.inv(XXt)#求逆
    XXtInvX = XXtInv.dot(X)
    coef = XXtInvX.dot(y.T)

    y_est = Xt.dot(coef)

    return y_est,coef


'''拟合直线'''
def fitting_line(absX):
    ploy_dim =2#拟合参数个数，即权重数量
    ## 数据准备
    bound_up_x, bound_down_x, bound_up_y, bound_down_y = get_bound_piont(absX)
    x = bound_up_x
    y = bound_up_y
    x1 = bound_down_x
    y1 = bound_down_y
    # # 最小二乘拟合
    [y_est,coef] = leastseq_byploy(x,y,ploy_dim)
    [y_est1,coef1] = leastseq_byploy(x1,y1,ploy_dim)
    # print(coef,coef1)
    # 显示拟合结果
    # est_data = plt.plot(x,y_est,color="g",linewidth= 4)
    # est_data = plt.plot(x1,y_est1,color="g",linewidth= 4)
    b1,k1 = coef
    b2,k2 = coef1
    k = (k1+k2)/2.0
    b = (b1+b2)/2.0
    return k, b

'''得到角度'''
def get_theta(k,b,w):
    w_c = w/2
    h_c = k * w_c + b
    x = np.arange(w)
    # y = h_c.repeat(w)
    y = k*x+b
    # est_data = plt.plot(x,y,color="r",linewidth= 3)
    theta =math.atan(k)
    return theta,w_c,h_c


'''旋转图片'''
def Rotating_picture(img,theta,h_c,w_c):
    h, w,_=img.shape
    matRotate = cv2.getRotationMatrix2D((h_c, w_c), 180*theta/math.pi, 1)
    img1=cv2.warpAffine(img, matRotate, (w, h))
    print(180*theta/math.pi)
    return img1

'''返回图像旋转一定角度后的坐标'''
def get_point(x,y,angle,w_c,h_c):
    (cX, cY) = (w_c, h_c)
    if (cX-x)== 0:
        theta1=math.atan((cY-y)/1)
        l=1/math.cos(theta1)
    else:
        theta1=math.atan((cY-y)/(cX-x))
        l=(cX-x)/math.cos(theta1)
    # print(h,w)
    new_x = cX-math.cos(angle +theta1)*l
    new_y = cY-math.sin(angle +theta1)*l
    # print(l, math.cos(theta1),cX,cY)
    # print((new_x-cX)**2+(new_y-cY)**2,(x-cX)**2+(y-cY)**2)
    return round(new_x), round(new_y)

''''数值边界的范围最大Y最小Y'''
def Vertical_boundary(theta,bound_down_x, bound_down_y,bound_up_x, bound_up_y, w_c, h_c):
    locs = list(zip(bound_down_x, bound_down_y))
    locs1 = list(zip(bound_up_x, bound_up_y))
    new_locs = list(map(lambda loc: get_point(loc[0], loc[1],   theta,w_c,h_c), locs))
    new_locs1 = list(map(lambda loc: get_point(loc[0], loc[1],   theta,w_c,h_c), locs1))
    # new_locs = list(map(fun, locs))
    # print(180*theta/math.pi)
    new_locs = list(zip(*new_locs))
    new_locs1 = list(zip(*new_locs1))
    # yu=max(bound_down_y)
    # yu1=min(bound_up_y)
    Y_max=int(max(new_locs[1]))
    Y_min=int(min(new_locs1[1]))
    print(Y_max, Y_min)
    Y_max = np.array(Y_max)
    Y_min = np.array(Y_min)
    Y1=Y_max.repeat(len(bound_up_x))
    Y2=Y_min.repeat(len(bound_down_x))
    plt.plot(bound_down_x,Y2+2, color='g', linewidth=2)
    plt.plot(bound_up_x, Y1+2, color='g', linewidth=2)
    # plt.plot(bound_down_x, bound_down_y, color='g', linewidth=1)
    # plt.plot(bound_up_x, bound_up_y, color='g', linewidth=1)
    # plt.plot(new_locs[0], new_locs[1],color="r",linewidth= 1)
    # plt.plot(new_locs1[0], new_locs1[1],color="r",linewidth= 1)
    return(Y_max, Y_min)

'''水平边界的范围截取'''
def Horizontal_boundary(cropped):
    max_1, max_2=0, 0
    height, width,_ = cropped.shape
    kernel_width = width // 20
    # print(height, width)
    x = [i for i in range(width)]
    light = [0 for i in range(width)]
    for w in range(width - kernel_width): #步长是1
        light[w+kernel_width//2] = np.sum(cropped[:, w: w+kernel_width])
        if w+kernel_width//2>width/3:
            if light[w+kernel_width//2]>light[w+kernel_width//2+1] and light[w+kernel_width//2]>light[w+kernel_width//2-1]:
               if max_1< light[w+kernel_width//2]:
                    max_1=light[w+kernel_width//2]
                    t800=w+kernel_width//2
    cut_1, cut_2=int(t800/3), int(2/3*(width-t800)+t800)
    plt.figure()
    plt.plot(x, light, color="b",linewidth= 2)
    plt.show()
    print(cut_1, cut_2)
    return cut_1,cut_2

""" 灰度归一化 """
def gray_normalization(img):
    g1, g2 = np.min(img), np.max(img)
    img = ((img - g1) / (g2 - g1) * 255).astype(np.uint8)
    return img
'''高斯去噪'''
def convertScaleAbs(img):
    img = cv2.convertScaleAbs(img,alpha=1.5,beta=0)
    return img

def main_(path):
    i=0
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for human_id in os.listdir(path):
        human_dir = os.path.join(path, human_id)
        for hand_id in os.listdir(human_dir):
            hand_dir = os.path.join(human_dir, hand_id)
            for finger_name in os.listdir(hand_dir):
                if os.path.splitext(finger_name)[-1]==".bmp":
                    img_path = os.path.join(hand_dir, finger_name)
                    img = cv2.imread(img_path)
                    i=i+1
                    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    grayImage = cv2.GaussianBlur(grayImage,(3,3),0)
                    absX, h, w=edge_detectio(grayImage)
                    bound_up_x, bound_down_x, bound_up_y, bound_down_y=get_bound_piont(absX)
                    k, b=fitting_line(absX)
                    theta,w_c,h_c=get_theta(k,b,w)
                    img=Rotating_picture(img,theta,h_c,w_c)
                    Y_max, Y_min=Vertical_boundary(-theta, bound_down_x,bound_down_y,bound_up_x,bound_up_y,w_c,h_c)
                    cropped = img[Y_max:Y_min, :]
                    X_min,X_max=Horizontal_boundary(cropped)
                    # ncrease_contrast(img)
                    cropped = img[Y_max:Y_min, X_min:X_max]
                    cropped = gray_normalization(cropped)
                    cropped = cv2.resize(cropped, (320,128))
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    # cv2.imwrite(img_path, cropped)
# path = 'data/Shandong University'
# per_test(path)

def per_test(img_path):
    img = cv2.imread(img_path)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.GaussianBlur(grayImage,(3,3),0)
    absX, h, w=edge_detectio(grayImage)
    bound_up_x, bound_down_x, bound_up_y, bound_down_y=get_bound_piont(absX)
    k, b=fitting_line(absX)
    theta,w_c,h_c=get_theta(k,b,w)
    img=Rotating_picture(img,theta,h_c,w_c)
    plt.imshow(img,cmap='gray')
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # absX1,h,w = edge_detectio(grayImage)
    # bound_up_x, bound_down_x, bound_up_y, bound_down_y=get_bound_piont(absX1)
    # plt.imshow(img,cmap='gray')
    Y_max, Y_min=Vertical_boundary(-theta, bound_down_x,bound_down_y,bound_up_x,bound_up_y,w_c,h_c)
    cropped = img[Y_max:Y_min, :]
    # cv2.imwrite('pictures/5.jpg',cropped)
    X_min,X_max=Horizontal_boundary(cropped)
    # ncrease_contrast(img)
    cropped = img[Y_max:Y_min, X_min:X_max]
    cropped = gray_normalization(cropped)
    cropped = cv2.resize(cropped, (320,128))
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('pictures/4.jpg',cropped)
path = 'data/Shandong University/074/left/ring_1.bmp'
per_test(path)
