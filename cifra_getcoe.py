import numpy as np
import torch
import torch.nn as nn
import numpy as np

torch.set_printoptions(threshold=np.inf)
# (-4,4)
def quantify_1_2_5(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*1)):
            a = str('00')
            return a
        elif(object<=-4):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 2*16*256/64)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=4):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 2*16*256/64)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-2,2)
def quantify_1_1_6(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*2)):
            a = str('00')
            return a
        elif(object<=-2):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/64)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=2):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/64)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-1,1)
def quantify_1_0_7(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32)):
            a = str('00')
            return a
        elif(object<=-1):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=1):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d#
# (-0.5,0.5)
def quantify_1_1_8(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*2)):
            a = str('00')
            return a
        elif(object<=-0.5):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*2)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.5):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*2)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-0.25,0.25)
def quantify_1_2_9(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*4)):
            a = str('00')
            return a
        elif(object<=-0.25):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*4)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.25):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*4)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-0.125,0.125)
def quantify_1_3_10(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*8)):
            a = str('00')
            return a
        elif(object<=-0.125):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*8)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.125):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*8)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-0.0625,0.0625)
def quantify_1_4_11(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*16)):
            a = str('00')
            return a
        elif(object<=-0.0625):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*16)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.0625):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*16)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-0.03125,0.03125)        1/32
def quantify_1_5_12(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*32)):
            a = str('00')
            return a
        elif(object<=-0.03125):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*32)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.03125):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*32)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-0.015625,0.0.015625)    1/64
def quantify_1_6_13(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*64)):
            a = str('00')
            return a
        elif(object<=-0.015625):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*64)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.015625):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*64)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d
# (-0.0078125,0.0078125)    1/128
def quantify_1_7_14(object):#目前使用的量化方法
    if(object<0):
        if(object>-1/(4096*32*128)):
            a = str('00')
            return a
        elif(object<=-0.0078125):
            a = str('80')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*128)), 'b')
            b = a[1:9]
            c = format(2 ** 7 + int(b, 2), 'x')
            return c
    else:
        if(object>=0.0078125):
            a = str('7f')
            return a
        else:
            a = format(2 ** 8 + (int(object * 16*256/32*128)), 'b')
            #print(a)
            b = a[1:9]
            #print(b)
            c = format(2 ** 8 + int(b, 2), 'x')
            #print(c)
            d = c[1:7]
        return d

feature = torch.tensor([

]).float()

for i in range(0, len(feature), 1):

    if(abs(feature[i])<=0.015625):
        print(quantify_1_6_13(feature[i]), end='')
    elif(abs(feature[i])<=0.03125):
        print(quantify_1_5_12(feature[i]), end='')
    elif(abs(feature[i])<=0.0625):
        print(quantify_1_4_11(feature[i]), end='')
    elif(abs(feature[i])<=0.125):
        print(quantify_1_3_10(feature[i]), end='')
    elif(abs(feature[i])<=0.25):
        print(quantify_1_2_9(feature[i]), end='')
    elif(abs(feature[i])<=0.5):
        print(quantify_1_1_8(feature[i]), end='')
    elif(abs(feature[i])<=1):
        print(quantify_1_0_7(feature[i]), end='')
    print(",")
