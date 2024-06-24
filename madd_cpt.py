conv1_conv=10838016
conv1_bn=802816
conv1_relu=401408
conv2_1depthConv_Conv2d=3612672
conv2_1depthConv_BatchNorm2d=802816
conv2_1depthConv_ReLU=401408
conv2_1pointConv_Conv2d=26492928
conv2_1pointConv_BatchNorm2d=1605632
conv2_1pointConv_ReLU=802816
conv2_2depthConv_Conv2d=1806336
conv2_2depthConv_BatchNorm2d=401408
conv2_2depthConv_ReLU=200704
conv2_2pointConv_Conv2d=26091520
conv2_2pointConv_BatchNorm2d=802816
conv2_2pointConv_ReLU=401408
conv2_3depthConv_Conv2d=3612672
conv2_3depthConv_BatchNorm2d=802816
conv2_3depthConv_ReLU=401408
conv2_3pointConv_Conv2d=51781632
conv2_3pointConv_BatchNorm2d=802816
conv2_3pointConv_ReLU=401408
conv2_4depthConv_Conv2d=903168
conv2_4depthConv_BatchNorm2d=200704
conv2_4depthConv_ReLU=100352
conv2_4pointConv_Conv2d=25890816
conv2_4pointConv_BatchNorm2d=401408
conv2_4pointConv_ReLU=200704
conv2_5depthConv_Conv2d=1806336
conv2_5depthConv_BatchNorm2d=401408
conv2_5depthConv_ReLU=200704
conv2_5pointConv_Conv2d=51580928
conv2_5pointConv_BatchNorm2d=401408
conv2_5pointConv_ReLU=200704
conv2_6depthConv_Conv2d=451584
conv2_6depthConv_BatchNorm2d=100352
conv2_6depthConv_ReLU=50176
conv2_6pointConv_Conv2d=25790464
conv2_6pointConv_BatchNorm2d=200704
conv2_6pointConv_ReLU=100352
conv3_1depthConv_Conv2d=903168
conv3_1depthConv_BatchNorm2d=200704
conv3_1depthConv_ReLU=100352
conv3_1pointConv_Conv2d=51480576
conv3_1pointConv_BatchNorm2d=200704
conv3_1pointConv_ReLU=100352
conv3_2depthConv_Conv2d=903168
conv3_2depthConv_BatchNorm2d=200704
conv3_2depthConv_ReLU=100352
conv3_2pointConv_Conv2d=51480576
conv3_2pointConv_BatchNorm2d=200704
conv3_2pointConv_ReLU=100352
conv3_3depthConv_Conv2d=903168
conv3_3depthConv_BatchNorm2d=200704
conv3_3depthConv_ReLU=100352
conv3_3pointConv_Conv2d=51480576
conv3_3pointConv_BatchNorm2d=200704
conv3_3pointConv_ReLU=100352
conv3_4depthConv_Conv2d=903168
conv3_4depthConv_BatchNorm2d=200704
conv3_4depthConv_ReLU=100352
conv3_4pointConv_Conv2d=51480576
conv3_4pointConv_BatchNorm2d=200704
conv3_4pointConv_ReLU=100352
conv3_5depthConv_Conv2d=903168
conv3_5depthConv_BatchNorm2d=200704
conv3_5depthConv_ReLU=100352
conv3_5pointConv_Conv2d=51480576
conv3_5pointConv_BatchNorm2d=200704
conv3_5pointConv_ReLU=100352
conv4_1depthConv_Conv2d=225792
conv4_1depthConv_BatchNorm2d=50176
conv4_1depthConv_ReLU=25088
conv4_1pointConv_Conv2d=25740288
conv4_1pointConv_BatchNorm2d=100352
conv4_1pointConv_ReLU=50176
conv4_2depthConv_Conv2d=230400
conv4_2depthConv_BatchNorm2d=51200
conv4_2depthConv_ReLU=25600
conv4_2pointConv_Conv2d=26240000
conv4_2pointConv_BatchNorm2d=51200
conv4_2pointConv_ReLU=25600
avgpool=0
fc=10240
softmax=0

# mobilenetv1
conv1_zero_all     =  0.001853125

conv2_1_dw_zero_all= 0.004296875
conv2_1_pw_zero_all= 0.000171875

conv2_2_dw_zero_all= 0.0026734375
conv2_2_pw_zero_all= 8.359375e-05

conv2_3_dw_zero_all= 0.00072734375
conv2_3_pw_zero_all=0.0

conv2_4_dw_zero_all= 0.000284375
conv2_4_pw_zero_all= 4.6875e-06

conv2_5_dw_zero_all= 0.0004171875
conv2_5_pw_zero_all=3.90625e-07

conv2_6_dw_zero_all= 0.00173046875
conv2_6_pw_zero_all= 0.000409375

conv3_1_dw_zero_all= 0.0034931640625
conv3_1_pw_zero_all= 2.87109375e-05

conv3_2_dw_zero_all= 0.001465625
conv3_2_pw_zero_all= 1.953125e-06

conv3_3_dw_zero_all= 0.0005775390625
conv3_3_pw_zero_all=0.0

conv3_4_dw_zero_all= 0.0003
conv3_4_pw_zero_all=0.0

conv3_5_dw_zero_all= 0.000509765625
conv3_5_pw_zero_all=3.90625e-07

conv4_1_dw_zero_all= 0.00350234375
conv4_1_pw_zero_all= 9.658203125e-05

conv4_2_dw_zero_all= 0.0239294921875
conv4_2_pw_zero_all= 0.00684423828125









a = (conv1_zero_all*(conv2_1depthConv_Conv2d+conv2_1depthConv_BatchNorm2d+conv2_1depthConv_ReLU)

      +conv2_1_dw_zero_all*(conv2_1pointConv_Conv2d+conv2_1pointConv_BatchNorm2d+conv2_1pointConv_ReLU)
      +conv2_1_pw_zero_all*(conv2_2depthConv_Conv2d+
conv2_2depthConv_BatchNorm2d+
conv2_2depthConv_ReLU)

      +conv2_2_dw_zero_all*(conv2_2pointConv_Conv2d+
conv2_2pointConv_BatchNorm2d+
conv2_2pointConv_ReLU)
      +conv2_2_pw_zero_all*(conv2_3depthConv_Conv2d+
conv2_3depthConv_BatchNorm2d+
conv2_3depthConv_ReLU)

      +conv2_3_dw_zero_all*(conv2_3pointConv_Conv2d+
conv2_3pointConv_BatchNorm2d+
conv2_3pointConv_ReLU)
      +conv2_3_pw_zero_all*(conv2_4depthConv_Conv2d+
conv2_4depthConv_BatchNorm2d+
conv2_4depthConv_ReLU)

      +conv2_4_dw_zero_all*(conv2_4pointConv_Conv2d+
conv2_4pointConv_BatchNorm2d+
conv2_4pointConv_ReLU)
      +conv2_4_pw_zero_all*(conv2_5depthConv_Conv2d+
conv2_5depthConv_BatchNorm2d+
conv2_5depthConv_ReLU)

     + conv2_5_dw_zero_all * (conv2_5pointConv_Conv2d +
                              conv2_5pointConv_BatchNorm2d +
                              conv2_5pointConv_ReLU)
     + conv2_5_pw_zero_all * (conv2_6depthConv_Conv2d +
                              conv2_6depthConv_BatchNorm2d +
                              conv2_6depthConv_ReLU)

      +conv2_6_dw_zero_all*(conv2_6pointConv_Conv2d+
conv2_6pointConv_BatchNorm2d+
conv2_6pointConv_ReLU)
      +conv2_6_pw_zero_all*(conv3_1depthConv_Conv2d+
conv3_1depthConv_BatchNorm2d+
conv3_1depthConv_ReLU)



      + conv3_1_dw_zero_all * (conv3_1pointConv_Conv2d +
                               conv3_1pointConv_BatchNorm2d +
                               conv3_1pointConv_ReLU)
      + conv3_1_pw_zero_all*(conv3_2depthConv_Conv2d +
                            conv3_2depthConv_BatchNorm2d +
                            conv3_2depthConv_ReLU)

      + conv3_2_dw_zero_all * (conv3_2pointConv_Conv2d +
                               conv3_2pointConv_BatchNorm2d +
                               conv3_2pointConv_ReLU)
      + conv3_2_pw_zero_all*(conv3_3depthConv_Conv2d +
                            conv3_3depthConv_BatchNorm2d +
                            conv3_3depthConv_ReLU)

      + conv3_3_dw_zero_all * (conv3_3pointConv_Conv2d +
                               conv3_3pointConv_BatchNorm2d +
                               conv3_3pointConv_ReLU)
      + conv3_3_pw_zero_all*(conv3_4depthConv_Conv2d +
                            conv3_4depthConv_BatchNorm2d +
                            conv3_4depthConv_ReLU)

      + conv3_4_dw_zero_all * (conv3_4pointConv_Conv2d +
                               conv3_4pointConv_BatchNorm2d +
                               conv3_4pointConv_ReLU)
      + conv3_4_pw_zero_all*(conv3_5depthConv_Conv2d +
                            conv3_5depthConv_BatchNorm2d +
                            conv3_5depthConv_ReLU)

      + conv3_5_dw_zero_all * (conv3_5pointConv_Conv2d +
                               conv3_5pointConv_BatchNorm2d +
                               conv3_5pointConv_ReLU)
      + conv3_5_pw_zero_all*(conv4_1depthConv_Conv2d +
                            conv4_1depthConv_BatchNorm2d +
                            conv4_1depthConv_ReLU)

      + conv4_1_dw_zero_all * (conv4_1pointConv_Conv2d +
                               conv4_1pointConv_BatchNorm2d +
                               conv4_1pointConv_ReLU)
      + conv4_1_pw_zero_all*(conv4_2depthConv_Conv2d +conv4_2depthConv_BatchNorm2d +conv4_2depthConv_ReLU))

print(a)

print(a/560005120)

# conv1conv=884736
# conv1bn=65536
# conv1relu=32768
# conv2_1depthConv_Conv2d=294912
# conv2_1depthConv_BatchNorm2d=65536
# conv2_1depthConv_ReLU=32768
# conv2_1pointConv_Conv2d=2162688
# conv2_1pointConv_BatchNorm2d=131072
# conv2_1pointConv_ReLU=65536
# conv2_2depthConv_Conv2d=147456
# conv2_2depthConv_BatchNorm2d=32768
# conv2_2depthConv_ReLU=16384
# conv2_2pointConv_Conv2d=2129920
# conv2_2pointConv_BatchNorm2d=65536
# conv2_2pointConv_ReLU=32768
# conv2_3depthConv_Conv2d=294912
# conv2_3depthConv_BatchNorm2d=65536
# conv2_3depthConv_ReLU=32768
# conv2_3pointConv_Conv2d=8454144
# conv2_3pointConv_BatchNorm2d=131072
# conv2_3pointConv_ReLU=65536
# conv2_4depthConv_Conv2d=589824
# conv2_4depthConv_BatchNorm2d=131072
# conv2_4depthConv_ReLU=65536
# conv2_4pointConv_Conv2d=16842752
# conv2_4pointConv_BatchNorm2d=131072
# conv2_4pointConv_ReLU=65536
# conv2_5depthConv_Conv2d=147456
# conv2_5depthConv_BatchNorm2d=32768
# conv2_5depthConv_ReLU=16384
# conv2_5pointConv_Conv2d=8421376
# conv2_5pointConv_BatchNorm2d=65536
# conv2_5pointConv_ReLU=32768
# conv3_1depthConv_Conv2d=294912
# conv3_1depthConv_BatchNorm2d=65536
# conv3_1depthConv_ReLU=32768
# conv3_1pointConv_Conv2d=16809984
# conv3_1pointConv_BatchNorm2d=65536
# conv3_1pointConv_ReLU=32768
# conv3_2depthConv_Conv2d=294912
# conv3_2depthConv_BatchNorm2d=65536
# conv3_2depthConv_ReLU=32768
# conv3_2pointConv_Conv2d=16809984
# conv3_2pointConv_BatchNorm2d=65536
# conv3_2pointConv_ReLU=32768
# conv3_3depthConv_Conv2d=294912
# conv3_3depthConv_BatchNorm2d=65536
# conv3_3depthConv_ReLU=32768
# conv3_3pointConv_Conv2d=16809984
# conv3_3pointConv_BatchNorm2d=65536
# conv3_3pointConv_ReLU=32768
# conv3_4depthConv_Conv2d=294912
# conv3_4depthConv_BatchNorm2d=65536
# conv3_4depthConv_ReLU=32768
# conv3_4pointConv_Conv2d=16809984
# conv3_4pointConv_BatchNorm2d=65536
# conv3_4pointConv_ReLU=32768
# conv3_5depthConv_Conv2d=294912
# conv3_5depthConv_BatchNorm2d=65536
# conv3_5depthConv_ReLU=32768
# conv3_5pointConv_Conv2d=16809984
# conv3_5pointConv_BatchNorm2d=65536
# conv3_5pointConv_ReLU=32768
# conv4_1depthConv_Conv2d=73728
# conv4_1depthConv_BatchNorm2d=16384
# conv4_1depthConv_ReLU=8192
# conv4_1pointConv_Conv2d=8404992
# conv4_1pointConv_BatchNorm2d=32768
# conv4_1pointConv_ReLU=16384
# conv4_2depthConv_Conv2d=36864
# conv4_2depthConv_BatchNorm2d=8192
# conv4_2depthConv_ReLU=4096
# conv4_2pointConv_Conv2d=4198400
# conv4_2pointConv_BatchNorm2d=8192
# conv4_2pointConv_ReLU=4096
#
# # mobilenetv1-Dw
#
# conv2_1_dw_zero_all= 0.69355625
#
# conv2_2_dw_zero_all= 0.783096875
#
# conv2_3_dw_zero_all= 0.7995140625
#
# conv2_4_dw_zero_all= 0.759220703125
#
# conv2_5_dw_zero_all= 0.634220703125
#
# conv3_1_dw_zero_all= 0.4933814453125
#
# conv3_2_dw_zero_all= 0.3480984375
#
# conv3_3_dw_zero_all= 0.3739482421875
#
# conv3_4_dw_zero_all= 0.3559927734375
#
# conv3_5_dw_zero_all= 0.350080078125
#
# conv4_1_dw_zero_all= 0.4123033203125
#
# conv4_2_dw_zero_all= 0.5849236328125
#
#
# a=(
#         # conv1_zero_all*(conv2_1depthConv_Conv2d+conv2_1depthConv_BatchNorm2d+conv2_1depthConv_ReLU)
#
#       +conv2_1_dw_zero_all*(conv2_1pointConv_Conv2d+conv2_1pointConv_BatchNorm2d+conv2_1pointConv_ReLU)
# #       +conv2_1_pw_zero_all*(conv2_2depthConv_Conv2d+
# # conv2_2depthConv_BatchNorm2d+
# # conv2_2depthConv_ReLU)
#
#       +conv2_2_dw_zero_all*(conv2_2pointConv_Conv2d+
# conv2_2pointConv_BatchNorm2d+
# conv2_2pointConv_ReLU)
# #       +conv2_2_pw_zero_all*(conv2_3depthConv_Conv2d+
# # conv2_3depthConv_BatchNorm2d+
# # conv2_3depthConv_ReLU)
#
#       +conv2_3_dw_zero_all*(conv2_3pointConv_Conv2d+
# conv2_3pointConv_BatchNorm2d+
# conv2_3pointConv_ReLU)
# #       +conv2_3_pw_zero_all*(conv2_4depthConv_Conv2d+
# # conv2_4depthConv_BatchNorm2d+
# # conv2_4depthConv_ReLU)
#
#       +conv2_4_dw_zero_all*(conv2_4pointConv_Conv2d+
# conv2_4pointConv_BatchNorm2d+
# conv2_4pointConv_ReLU)
# #       +conv2_4_pw_zero_all*(conv2_5depthConv_Conv2d+
# # conv2_5depthConv_BatchNorm2d+
# # conv2_5depthConv_ReLU)
#
#       +conv2_5_dw_zero_all*(conv2_5pointConv_Conv2d+
# conv2_5pointConv_BatchNorm2d+
# conv2_5pointConv_ReLU)
# #       +conv2_5_pw_zero_all*(conv3_1depthConv_Conv2d+
# # conv3_1depthConv_BatchNorm2d+
# # conv3_1depthConv_ReLU)
#
#       + conv3_1_dw_zero_all * (conv3_1pointConv_Conv2d +
#                                conv3_1pointConv_BatchNorm2d +
#                                conv3_1pointConv_ReLU)
#       # + conv3_1_pw_zero_all*(conv3_2depthConv_Conv2d +
#       #                       conv3_2depthConv_BatchNorm2d +
#       #                       conv3_2depthConv_ReLU)
#
#       + conv3_2_dw_zero_all * (conv3_2pointConv_Conv2d +
#                                conv3_2pointConv_BatchNorm2d +
#                                conv3_2pointConv_ReLU)
#       # + conv3_2_pw_zero_all*(conv3_3depthConv_Conv2d +
#       #                       conv3_3depthConv_BatchNorm2d +
#       #                       conv3_3depthConv_ReLU)
#
#       + conv3_3_dw_zero_all * (conv3_3pointConv_Conv2d +
#                                conv3_3pointConv_BatchNorm2d +
#                                conv3_3pointConv_ReLU)
#       # + conv3_3_pw_zero_all*(conv3_4depthConv_Conv2d +
#       #                       conv3_4depthConv_BatchNorm2d +
#       #                       conv3_4depthConv_ReLU)
#
#       + conv3_4_dw_zero_all * (conv3_4pointConv_Conv2d +
#                                conv3_4pointConv_BatchNorm2d +
#                                conv3_4pointConv_ReLU)
#       # + conv3_4_pw_zero_all*(conv3_5depthConv_Conv2d +
#       #                       conv3_5depthConv_BatchNorm2d +
#       #                       conv3_5depthConv_ReLU)
#
#       + conv3_5_dw_zero_all * (conv3_5pointConv_Conv2d +
#                                conv3_5pointConv_BatchNorm2d +
#                                conv3_5pointConv_ReLU)
#       # + conv3_5_pw_zero_all*(conv4_1depthConv_Conv2d +
#       #                       conv4_1depthConv_BatchNorm2d +
#       #                       conv4_1depthConv_ReLU)
#
#       + conv4_1_dw_zero_all * (conv4_1pointConv_Conv2d +
#                                conv4_1pointConv_BatchNorm2d +
#                                conv4_1pointConv_ReLU)
#       # + conv4_1_pw_zero_all*(conv4_2depthConv_Conv2d +conv4_2depthConv_BatchNorm2d +conv4_2depthConv_ReLU)
#    )
#
# print(a)
#
# print(a/141076480)









#Resnet18
# model0_Conv2d=118816768
# model0_BatchNorm2d=1605632
# model0_ReLU=802816
# model0_MaxPool2d=802816
# model1_Conv2d=115806208
# model1_BatchNorm2d=401408
# model1_ReLU=200704
# model1_2Conv2d=115806208
# model1_2BatchNorm2d=401408
# model1_2ReLU=200704
# R1=200704
# model2_Conv2d=115806208
# model2_BatchNorm2d=401408
# model2_ReLU=200704
# model2_2Conv2d=115806208
# model2_2BatchNorm2d=401408
# model2_2ReLU=200704
# R2=200704
# model3_Conv2d=57903104
# model3_BatchNorm2d=200704
# model3_ReLU=100352
# model3_2Conv2d=115705856
# model3_2BatchNorm2d=200704
# model3_2ReLU=100352
# en1_Conv2d=6522880
# en1_BatchNorm2d=200704
# en1_ReLU=100352
# R3=100352
# model4_Conv2d=115705856
# model4_BatchNorm2d=200704
# model4_ReLU=100352
# model4_2Conv2d=115705856
# model4_2BatchNorm2d=200704
# model4_2ReLU=100352
# R4=100352
# model5_Conv2d=57852928
# model5_BatchNorm2d=100352
# model5_ReLU=50176
# model5_2Conv2d=115655680
# model5_2BatchNorm2d=100352
# model5_2ReLU=50176
# en2_Conv2d=6472704
# en2_BatchNorm2d=100352
# en2_ReLU=50176
# R5=50176
# model6_Conv2d=115655680
# model6_BatchNorm2d=100352
# model6_ReLU=50176
# model6_2Conv2d=115655680
# model6_2BatchNorm2d=100352
# model6_2ReLU=50176
# R6=50176
# model7_Conv2d=57827840
# model7_BatchNorm2d=50176
# model7_ReLU=25088
# model7_2Conv2d=115630592
# model7_2BatchNorm2d=50176
# model7_2ReLU=25088
# en3_Conv2d=6447616
# en3_BatchNorm2d=50176
# en3_ReLU=25088
# R7=25088
# model8_Conv2d=115630592
# model8_BatchNorm2d=50176
# model8_ReLU=25088
# model8_2Conv2d=115630592
# model8_2BatchNorm2d=50176
# model8_2ReLU=25088
# R8=25088
# aap=0
# flatten=0
# fc=5120
#
#
#
#
# conv1_zero_all  =   0.5760453125
#
# conv2_1_zero_all=0.5462859375
# conv2_2_zero_all=0.850321875
#
# conv3_1_zero_all=0.8076296875
# conv3_2_zero_all=0.9375125
# cut_resdual1_all= 0.518125
#
#
# conv4_1_zero_all= 0.7287234375
# conv4_2_zero_all= 0.82354140625
# cut_resdual2_all= 0.38443828125
#
# conv5_1_zero_all= 0.86380078125
# conv5_2_zero_all= 0.4540578125
# cut_resdual3_all= 0.19627109375
#
#
# conv6_1_zero_all= 0.63942109375
# conv6_2_zero_all= 0.421025390625
# cut_resdual4_all= 0.304525390625
#
# conv7_1_zero_all= 0.502200390625
# conv7_2_zero_all= 0.060518359375
# cut_resdual5_all= 0.019886328125
#
#
# conv8_1_zero_all= 0.261917578125
# conv8_2_zero_all= 0.8746201171875
# cut_resdual6_all= 0.84036171875
#
# conv9_1_zero_all= 0.5364373046875
# conv9_2_zero_all= 0.9895361328125
# cut_resdual7_all= 0.8307533203125
#
# f1_1_all=0.53715625
# f2_1_all=0.42504765625
# f3_1_all=0.666312109375
# f4_1_all=0.9636369140625
#
# a=(conv1_zero_all*(model0_MaxPool2d+model1_Conv2d)
#       +conv2_1_zero_all*(model1_2Conv2d)
#       +f1_1_all*(model2_Conv2d)
#       +conv3_1_zero_all*(model2_2Conv2d)
#       +conv3_2_zero_all*(R2)
#       +cut_resdual1_all*(en1_Conv2d+model3_Conv2d)
#       +conv4_1_zero_all*(model3_2Conv2d)
#       +conv4_2_zero_all*(R3)
#       +cut_resdual2_all*(model4_Conv2d)
#       +conv5_1_zero_all*(model4_2Conv2d)
#       +conv5_2_zero_all*(R4)
#       +cut_resdual3_all*(en2_Conv2d+en2_BatchNorm2d)
#       +conv6_1_zero_all*(model5_2Conv2d)
#       +conv6_2_zero_all*(R5)
#       +cut_resdual4_all*(model6_Conv2d)
#       +conv7_1_zero_all*(model6_2Conv2d)
#       +conv7_2_zero_all*(R6)
#       +cut_resdual5_all*(en3_Conv2d+model7_Conv2d)
#       +conv8_1_zero_all*(model7_2Conv2d)
#       +conv8_2_zero_all*(R7)
#       +cut_resdual6_all*(model8_Conv2d)
#       +conv9_1_zero_all*(model8_2Conv2d)
#       +conv9_2_zero_all*(R8)
#
#
#
# )
# print(a)
#
# print(a/(1825056768))


#Cifra10Net
# conv1_pw=262144
# conv1_pw_BatchNorm2d=131072
# conv2_dw=655360
# conv2_dw_BatchNorm2d=131072
# conv2_pw=2228224
# conv2_pw_BatchNorm2d=262144
# maxpool1=131072
# conv3_dw=327680
# conv3_dw_BatchNorm2d=65536
# conv3_pw=2162688
# conv3_pw_BatchNorm2d=131072
# maxpool2=65536
# conv4_dw=163840
# conv4_dw_BatchNorm2d=32768
# conv4_pw=2129920
# conv4_pw_BatchNorm2d=65536
# maxpool3=32768
# conv5_dw=81920
# conv5_dw_BatchNorm2d=16384
# conv5_pw=2113536
# conv5_pw_BatchNorm2d=32768
# maxpool4=16384
# conv6_dw=40960
# conv6_dw_BatchNorm2d=8192
# conv6_pw=2105344
# conv6_pw_BatchNorm2d=16384
# maxpool5=8192
# relu=8192
# dropout=0
# fc=20480
# softmax=0
#
#
#
#
# conv1_pw_cut_all= 0
# conv2_dw_cut_all= 0
# conv2_pw_cut_all= 0
# conv3_dw_cut_all= 0
# conv3_pw_cut_all= 0
# conv4_dw_cut_all= 0
# conv4_pw_cut_all= 0
# conv5_dw_cut_all= 0
# conv5_pw_cut_all= 0
# conv6_dw_cut_all= 0
# conv6_pw_cut_all= 0
#
# conv2_dw_cut_all=  0.080525
# conv3_dw_cut_all= 0.0398875
# conv4_dw_cut_all= 0.027665625
# conv5_dw_cut_all= 0.048621875
# conv6_dw_cut_all=0.093410546875
#
# print(conv1_pw_cut_all*(conv2_dw+1.5*conv2_dw_BatchNorm2d)
#       +conv2_dw_cut_all*(conv2_pw)
#       +conv2_pw_cut_all*(conv3_dw+1.5*conv3_dw_BatchNorm2d+maxpool1)
#       +conv3_dw_cut_all*(conv3_pw)+
# conv3_pw_cut_all*(conv4_dw+1.5*conv4_dw_BatchNorm2d+maxpool2)
#       +conv4_dw_cut_all*(conv4_pw)+
# conv4_pw_cut_all*(conv5_dw+1.5*conv5_dw_BatchNorm2d+maxpool3)
#       +conv5_dw_cut_all*(conv5_pw)+
# conv5_pw_cut_all*(conv6_dw+1.5*conv6_dw_BatchNorm2d+maxpool4)
#       +conv6_dw_cut_all*(conv6_pw)
# +conv6_pw_cut_all*maxpool5)
#
# a =(conv1_pw_cut_all*(conv2_dw+1.5*conv2_dw_BatchNorm2d)
#       +conv2_dw_cut_all*(conv2_pw)
#       +conv2_pw_cut_all*(conv3_dw+1.5*conv3_dw_BatchNorm2d+maxpool1)
#       +conv3_dw_cut_all*(conv3_pw)+
# conv3_pw_cut_all*(conv4_dw+1.5*conv4_dw_BatchNorm2d+maxpool2)
#       +conv4_dw_cut_all*(conv4_pw)+
# conv4_pw_cut_all*(conv5_dw+1.5*conv5_dw_BatchNorm2d+maxpool3)
#       +conv5_dw_cut_all*(conv5_pw)+
# conv5_pw_cut_all*(conv6_dw+1.5*conv6_dw_BatchNorm2d+maxpool4)
#       +conv6_dw_cut_all*(conv6_pw)
# +conv6_pw_cut_all*maxpool5)
# print(a/13885440)
#
# print(conv1_pw_cut_all*(conv2_dw+1.5*conv2_dw_BatchNorm2d))

