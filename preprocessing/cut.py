import numpy as np
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
# 读取图像
img = cv2.imread('data/Shandong University/004/left/middle_4.bmp')
# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(grayImage,cmap='gray')
grayImage = cv2.GaussianBlur(grayImage,(5,5),0)
plt.imshow(grayImage, cmap="gray")

# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernelx1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)
absX = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
absX1 = cv2.filter2D(grayImage, cv2.CV_16S, kernelx1)
# absX = cv2.convertScaleAbs(absX)
plt.imshow(absX, cmap="gray")
plt.figure()
plt.imshow(absX1, cmap="gray")
plt.show()
h, w=absX.shape
jihe=[]
# print(absX * (absX > 200))
print(8*np.mean(absX))
print(absX.shape)
ret,absX=cv2.threshold(absX, 0, 255, cv2.THRESH_OTSU)
print(ret) #测试图：absX边缘检测，mask阈值
plt.imshow(absX,cmap="gray")
plt.show() #测试图：absX边缘检测，mask阈值
# plt.imshow(mask,cmap="gray"),plt.title('原始图像'), plt.axis('off') #坐标轴关闭
# plt.figure()
# plt.imshow(absX,cmap="gray")
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
print(t, t1)
t2=(t+t1)/2.
    # print('x: {}, y_min: {}, y_max: {}'.format(i, y_min, y_max))

bound_up_x = np.array(bound_up_x)
bound_down_x = np.array(bound_down_x)
bound_up_y = np.array(bound_up_y)
bound_down_y = np.array(bound_down_y)
# img = cv2.imread(r"C:\Users\Administrator\Desktop\players.png")
# cv2.imshow("sss", img[0:375, 240:480])

# """最小二乘法"""
# import numpy as np
# import matplotlib.pyplot plt

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

def fit_fun(x):
    '''
    要拟合的函数
    '''
    #return np.power(x,5)
    return np.sin(x)
#    return 5*x+3
#    return np.sqrt(25-pow(x-5,2))


if __name__ == '__main__':
    data_num = 100
    ploy_dim =2#拟合参数个数，即权重数量
    # noise_scale = 0.2
    ## 数据准备
    # x = np.array(np.linspace(-2*np.pi,2*np.pi,data_num))   #数据
    # y = fit_fun(x)+noise_scale*np.random.rand(1,data_num)  #添加噪声
    x = bound_up_x
    y = bound_up_y
    x1 = bound_down_x
    y1 = bound_down_y
    # 最小二乘拟合
    [y_est,coef] = leastseq_byploy(x,y,ploy_dim)
    [y_est1,coef1] = leastseq_byploy(x1,y1,ploy_dim)
    # print(coef,coef1)
    #显示拟合结果
    # org_data = plt.scatter(x,y,color="r",marker='o',s = 50)
    est_data = plt.plot(x,y_est-5,color="g",linewidth= 2)
    # org_data = plt.scatter(x1,y1,color="r",marker='o',s = 50)
    est_data = plt.plot(x1,y_est1+5,color="g",linewidth= 2)
    b1,k1 = coef
    b2,k2 = coef1
    k = (k1+k2)/2.0
    b = (b1+b2)/2
    w_c = w/2
    h_c = k * w_c + b
    theta =math.atan(k)
    # print(theta)
    matRotate = cv2.getRotationMatrix2D((h_c, w_c), 180*theta/math.pi, 1)
    dst = cv2.warpAffine(img, matRotate, (w, h))
    # plt.imshow (dst)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Fit funtion with leastseq method")
    # plt.legend(["Fited function"])
    # # plt.show()
    # print('x: {}, y_min: {}, y_max: {}'.format(i, y_min, y_max))
    # y_min, y_max, t, t1 = 0, h-1, 0, h-1
    # # h, w=dst.shape()
    # for i in range (w):
    #     if t < y_min:
    #         t=y_min
    #     if t1 > y_max:
    #         t1=y_max
    #     for j in range (h):
    #         if dst[j, i] > A:
    #             if j < h // 2 and j > y_min:
    #                 y_min = j
    #             if j > h // 2 and j < y_max:
    #                 y_max = j
    #                 break
    # print(t, t1)
#参数x y 图像  角度
#返回图像旋转一定角度后的坐标
def get_point(x,y,image,angle):
    (h, w) = image.shape[:2]
    # 将图像中心设为旋转中心
    (cX, cY) = (w // 2, h // 2)
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
# new_x,new_y= get_point(,,img,30)

def fun(loc):
    return get_point(loc[0], loc[1], img, 10)

locs = list(zip(bound_down_x, bound_down_y))
locs1 = list(zip(bound_up_x, bound_up_y))
new_locs = list(map(lambda loc: get_point(loc[0], loc[1], img,  theta), locs))
new_locs1 = list(map(lambda loc: get_point(loc[0], loc[1], img,  theta), locs1))
# new_locs = list(map(fun, locs))
# print(180*theta/math.pi)
new_locs = list(zip(*new_locs))
new_locs1 = list(zip(*new_locs1))
yu=max(bound_down_y)
yu1=min(bound_up_y)
Y_max=int(max(new_locs[1]))
Y_min=int(min(new_locs1[1]))
# print(Y_max, Y_min)
#划线
# green = (0,255,0)
# cv2.line(dst,(0,Y_min),(320,Y_min),green,1)
# cv2.line(dst,(0,Y_max),(320,Y_max),green,1)
# cv2.imshow("image",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure()
plt.plot(bound_down_x, bound_down_y, color='g', linewidth=2)
plt.plot(bound_up_x, bound_up_y, color='g', linewidth=2)
plt.plot(new_locs[0], new_locs[1],color="r",linewidth= 2)
plt.plot(new_locs1[0], new_locs1[1],color="r",linewidth= 2)
img = cv2.convertScaleAbs(img,alpha=1.5,beta=0)
img1 = cv2.imread('data/Shandong University/004/left/middle_4.bmp',0)
clahe = cv2.createCLAHE(3,(8,8))
img = clahe.apply(img1)
max_1, max_2=0, 0
height, width = grayImage.shape
kernel_width = width // 20
# print(height, width)
x = [i for i in range(width)]
light = [0 for i in range(width)]
for w in range(width - kernel_width): #步长是1
    light[w+kernel_width//2] = np.sum(grayImage[:, w: w+kernel_width])
    if w+kernel_width//2>width/3:
        if light[w+kernel_width//2]>light[w+kernel_width//2+1] and light[w+kernel_width//2]>light[w+kernel_width//2-1]:
            if max_1< light[w+kernel_width//2]:
                max_1=light[w+kernel_width//2]
                t800=w+kernel_width//2
            # if max_1>light[w+kernel_width//2]and max_2<light[w+kernel_width//2]:
            #     max_2 =light[w+kernel_width//2]
            #     t800=w+kernel_width//2
            # if max_1< light[w+kernel_width//2]:
            #     max_1=light[w+kernel_width//2]
            #     t1000=w+kernel_width//2
# t800=max(t800,t1000)
print(t800)
    # print(max_1, max_2,t800, t1000)
cut_1, cut_2=int(t800/3), int(2/3*(width-t800)+t800)
#     light[w+kernel_width//2] = np.sum(grayImage[:, w: w+kernel_width])
#     if light[w+kernel_width//2]>light[w+kernel_width//2+1] and light[w+kernel_width//2]>light[w+kernel_width//2-1]:
#         if max_1>light[w+kernel_width//2]and max_2<light[w+kernel_width//2]:
#             max_2 =light[w+kernel_width//2]
#             t800=w+kernel_width//2
#         if max_1< light[w+kernel_width//2]:
#             max_1=light[w+kernel_width//2]
#             t1000=w+kernel_width//2
# t800=max(t800,t1000)
plt.figure()
plt.plot(x, light, color="g",linewidth= 2)
print(cut_1, cut_2)
cropped = img[Y_max:Y_min, cut_1:cut_2]
plt.figure()
plt.imshow(cropped) # 裁剪坐标为[y0:y1, x0:x1]
print(Y_max,Y_min, cut_1,cut_2)
cv2.imwrite("processed_data/2.jpg", cropped)
plt.figure()
plt.imshow(cropped)
plt.show()
