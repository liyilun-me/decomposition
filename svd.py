# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as la
#1. SVD分解
A= [[1,1,3,6,1],[5,1,8,4,2],[7,9,2,1,2]]
A=np.array(A)
U,s,VT = la.svd(A)
# 为节省空间，svd输出s只有奇异值的向量
print('奇异值：',s)
# 根据奇异值向量s，生成奇异值矩阵
Sigma = np.zeros(np.shape(A))
Sigma[:len(s),:len(s)] = np.diag(s)

print("左奇异值矩阵：\n",U)
print('奇异值矩阵：\n',Sigma)
print('右奇异矩阵的转置：\n',VT)

#2.SVD重构
B = U.dot(Sigma.dot(VT))
print('重构后的矩阵B：\n', B)

print('原矩阵与重构矩阵是否相同？',np.allclose(A,B))

# 3. SVD矩阵压缩（降维）
for k in range(3,0,-1):  # 3,2,1
    # U的k列，VT的k行
    D = U[:,:k].dot(Sigma[:k,:k].dot(VT[:k,:]))
    print('k=',k,"压缩后的矩阵：\n",np.round(D,1))  # round取整数

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
from itertools import count
from PIL import Image
import numpy as np

def img_compress(img,percent):
    U,s,VT=np.linalg.svd(img)
    Sigma = np.zeros(np.shape(img))
    Sigma[:len(s),:len(s)] = np.diag(s)
    # 根据压缩比 取k值

    # 方法1 # k是奇异值数值总和的百分比个数，（奇异值权重）
    count = (int)(sum(s))*percent
    k = -1
    curSum = 0
    while curSum <= count:
        k+=1
        curSum += s[k]

    # 方法2
    # k = (int)(percent*len(s)) # k是奇异值个数的百分比个数

    # 还原后的数据D
    D = U[:,:k].dot(Sigma[:k,:k].dot(VT[:k,:]))
    D[D<0] = 0
    D[D>255] = 255
    return np.rint(D).astype('uint8')

# 图像重建
def rebuild_img(filename,percent):
    img = Image.open(filename,'r')
    a = np.array(img)
    R0 = a[:,:,0]
    G0 = a[:,:,1]
    B0 = a[:,:,2]
    R = img_compress(R0,percent)
    G = img_compress(G0,percent)
    B = img_compress(B0,percent)
    re_img = np.stack((R,G,B),2)
    # 保存图片
    newfile = filename+str(percent*100)+'.jpg'
    Image.fromarray(re_img).save(newfile)
    img = Image.open(newfile)
    img.show()

rebuild_img('cat.jpg',0.58)
rebuild_img('cat.jpg',0.8)
rebuild_img('cat.jpg',0.9)