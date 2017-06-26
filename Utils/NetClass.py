# NetClass.py
# Defines some util classes for easily construct network structure

import caffe
from caffe import layers as L
from caffe import params as P

import math
import numpy as np

class MyNetClass(object):

    def __init__(self):
        self.net = caffe.NetSpec()
        self.testnet = caffe.NetSpec()

    def ConvSameResolution_ReLU(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/Conv')
            self.ReLU(name, name + '/BN', False)
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.ReLU(name, name + '/Conv', False)

    def ConvSameResolution_sigmoid_normalize(self, nOutC, kW, kH, name, bottomName = None, norm = False, center = 0.5, range = 1.0, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2
 
        self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
        if(norm):
            self.Sigmoid(name + 'Sigmoid', name + '/Conv', True)
            mValue = center - range / 2
            self.Power(name, name + 'Sigmoid', 1.0, range, mValue)
        else: 
            self.Sigmoid(name, name + '/Conv', False)


    def ConvSameResolution_sigmoid(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/Conv')
            self.Sigmoid(name, name + '/BN', False)
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.Sigmoid(name, name + '/Conv', False)

    def ConvSameResolution_minusReLU(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/Conv')
            self.ReLU(name + '/ReLU', name + '/BN')
            self.Power(name, name + '/ReLU', 1, -1, 0)
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.ReLU(name + '/ReLU', name + '/Conv')
            self.Power(name, name + '/ReLU', 1, -1, 0)

    def ConvSameResolution_FCOnly(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name, name + '/Conv')
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name, bottomName = bottomName, initMethod = initMethod)

    def ConvSameResolution(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/Conv')
            self.ReLU(name, name + '/BN', True)
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.ReLU(name, name + '/Conv', True)

    def ConvHalfResolutionNoPooling(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = kW / 2 
        pH = kH / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, sW = 2, sH = 2, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/Conv')
            self.ReLU(name, name + '/BN')
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, sW = 2, sH = 2, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.ReLU(name, name + '/Conv')

    def Flatten(self, name, bottomName):
        self.net[name] = L.Flatten(self.net[bottomName])# InnerProduct(self.net[bottomName], num_output = nOut, weight_filler=dict(type='constant'))
        self.testnet[name] = L.Flatten(self.testnet[bottomName])#self.testnet[name] = L.InnerProduct(self.testnet[bottomName], num_output = nOut, weight_filler=dict(type='constant'))

    def ConvHalfResolution(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 1) / 2
        pH = (kH - 1) / 2

        if(withBN):
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/Conv')
            self.ReLU(name + '/ReLU', name + '/BN')
            self.MaxPooling(2, 2, 0, 0, 2, 2, name, name + '/ReLU')
        else:
            self.Conv(nOutC, kW, kH, pW = pW, pH = pH, name = name + '/Conv', bottomName = bottomName, initMethod = initMethod)
            self.ReLU(name + '/ReLU', name + '/Conv')
            self.MaxPooling(2, 2, 0, 0, 2, 2, name, name + '/ReLU')

    def BilinearUpsample(self, nOutC, name, bottomName, factor = 2):
        k = 2*factor - factor % 2
        s = factor
        p = int(math.ceil((factor - 1) / 2.0))
        self.net[name] = L.Deconvolution(self.net[bottomName], convolution_param = dict(kernel_w = k, kernel_h = k, pad_w = p, pad_h = p, stride_h = s, stride_w = s, num_output = nOutC, group = nOutC, weight_filler = dict(type='bilinear'), bias_term = False), param = dict(lr_mult = 0, decay_mult = 0)) 
        self.testnet[name] = L.Deconvolution(self.testnet[bottomName], convolution_param = dict(kernel_w = k, kernel_h = k, pad_w = p, pad_h = p, stride_h = s, stride_w = s, num_output = nOutC, group = nOutC, weight_filler = dict(type='bilinear'), bias_term = False), param = dict(lr_mult = 0, decay_mult = 0)) 

    def DeConvDoubleResolutionBilinear(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        self.ConvSameResolution(nOutC, kW, kH, name + '/Conv', bottomName, withBN, initMethod = initMethod)
        self.BilinearUpsample(nOutC, name, name + '/Conv')

    def DeConvDoubleResolution(self, nOutC, kW, kH, name, bottomName = None, withBN = False, initMethod = 'xavier'):
        pW = (kW - 2) / 2
        pH = (kH - 2) / 2

        if(withBN):
            self.DeConv(nOutC, kW, kH, pW = pW, pH = pH, sW = 2, sH = 2, name = name + '/DeConv', bottomName = bottomName, initMethod = initMethod)
            self.BatchNorm(name + '/BN', name + '/DeConv')
            self.ReLU(name + '/ReLU', name + '/BN')
        else:
            self.DeConv(nOutC, kW, kH, pW = pW, pH = pH, sW = 2, sH = 2, name = name + '/DeConv', bottomName = bottomName, initMethod = initMethod)
            self.ReLU(name + '/ReLU', name + '/DeConv')

    def FCReLU(self, nOut, name, bottomName = None, initMethod = 'xavier'):
        self.FC(nOut, name + '/FC', bottomName, initMethod)
        self.ReLU(name, name + '/FC')

    def ReLU(self, name, bottomName = None, inplace = True):
        self.net[name] = L.ReLU(self.net[bottomName], in_place = inplace)
        self.testnet[name] = L.ReLU(self.testnet[bottomName], in_place = inplace)		 

    def MinusReLU(self, name, bottomName = None, inplace = True):
        self.ReLU(name + '/ReLU', bottomName, inplace)
        self.Power(name, name + '/ReLU', 1, -1, 0) 

    def TanHNormlize(self, name, bottomName = None):
        self.TanH(name + '/TanH', bottomName)
        self.Power(name, name + '/TanH', 1, 0.5, 0.5)

    def TanH(self, name, bottomName = None):
        self.net[name] = L.TanH(self.net[bottomName], in_place = True)
        self.testnet[name] = L.TanH(self.testnet[bottomName], in_place = True)		 

    def Sigmoid(self, name, bottomName = None, inplace = True):
        self.net[name] = L.Sigmoid(self.net[bottomName], in_place = inplace)
        self.testnet[name] = L.Sigmoid(self.testnet[bottomName], in_place = inplace)

    def FC(self, nOut, name, bottomName = None, initMethod = 'xavier'):
        self.net[name] = L.InnerProduct(self.net[bottomName], num_output = nOut, weight_filler=dict(type=initMethod))
        self.testnet[name] = L.InnerProduct(self.testnet[bottomName], num_output = nOut, weight_filler=dict(type=initMethod))

    def MSELoss(self, name, bottomName0 = None, bottomName1 = None, weight = 1.0):
        self.net[name] = L.EuclideanLoss(self.net[bottomName0], self.net[bottomName1], loss_weight = weight)
        self.testnet[name] = L.EuclideanLoss(self.testnet[bottomName0], self.testnet[bottomName1], loss_weight = weight)

    def MaxPooling(self, kW, kH, pW = 0, pH = 0, sW = 1, sH = 1, name = None, bottomName = None):
        self.net[name] = L.Pooling(self.net[bottomName], kernel_w = kW, kernel_h = kH, pad_w = pW, pad_h = pH, stride_h = sH, stride_w = sW, pool = P.Pooling.MAX)
        self.testnet[name] = L.Pooling(self.testnet[bottomName], kernel_w = kW, kernel_h = kH, pad_w = pW, pad_h = pH, stride_h = sH, stride_w = sW, pool = P.Pooling.MAX)

    def Data(self, batchSize, channal, height, width, name, phase = 'all'):
        self.net[name] = L.Input(shape=[dict(dim=[batchSize, channal, height, width])])
        self.testnet[name] = L.Input(shape=[dict(dim=[1, channal, height, width])])

    def Conv(self, nOutC, kW, kH, pW = 0, pH = 0, sW = 1, sH = 1, name = None, bottomName = None, initMethod = 'xavier'):
        self.net[name] = L.Convolution(self.net[bottomName], kernel_w = kW, kernel_h = kH, pad_w = pW, pad_h = pH, stride_h = sH, stride_w = sW, num_output = nOutC, weight_filler=dict(type=initMethod))
        self.testnet[name] = L.Convolution(self.testnet[bottomName], kernel_w = kW, kernel_h = kH, pad_w = pW, pad_h = pH, stride_h = sH, stride_w = sW, num_output = nOutC, weight_filler=dict(type=initMethod))

    def DeConv(self, nOutC, kW, kH, pW = 0, pH = 0, sW = 1, sH = 1, name = None, bottomName = None, initMethod = 'xavier'):
        self.net[name] = L.Deconvolution(self.net[bottomName], convolution_param = dict(kernel_w = kW, kernel_h = kH, pad_w = pW, pad_h = pH, stride_h = sH, stride_w = sW, num_output = nOutC, weight_filler = dict(type=initMethod)))
        self.testnet[name] = L.Deconvolution(self.testnet[bottomName], convolution_param = dict(kernel_w = kW, kernel_h = kH, pad_w = pW, pad_h = pH, stride_h = sH, stride_w = sW, num_output = nOutC, weight_filler = dict(type=initMethod)))

    def BatchNorm(self, name = None, bottomName = None):
        self.net[name + '_BeforeScale'] = L.BatchNorm(self.net[bottomName], use_global_stats = False)#, in_place = True)
        self.net[name] = L.Scale(self.net[name + '_BeforeScale'], bias_term = True, bias_filler = dict(type='constant', value=0.0001), filler = dict(type='constant', value=1.00001))#, in_place = True)
       
        self.testnet[name + '_BeforeScale'] = L.BatchNorm(self.testnet[bottomName], use_global_stats = True)#, in_place = True)
        self.testnet[name] = L.Scale(self.testnet[name + '_BeforeScale'], bias_term = True)#, in_place = True)

    def Power(self, name = None, bottomName = None, power = 1, scale = 1, shift = 0):
        self.net[name] = L.Power(self.net[bottomName], power = power, scale = scale, shift = shift)#, in_place = True)
        self.testnet[name] = L.Power(self.testnet[bottomName], power = power, scale = scale, shift = shift)#, in_place = True)

    def Log(self, name = None, bottomName = None, base = -1, bias = 0):
        self.net[name] = L.Log(self.net[bottomName], base = base, shift = bias)#, in_place = True)
        self.testnet[name] = L.Log(self.testnet[bottomName], base = base, shift = bias)#, in_place = True)

    def Concat(self, name, bottomNameList = [], axis = 1):
        netList = [self.net[x] for x in bottomNameList]
        testnetList = [self.testnet[x] for x in bottomNameList]
        self.net[name] = L.Concat(*netList, concat_dim = axis)
        self.testnet[name] = L.Concat(*testnetList, concat_dim = axis)

    def EltSum(self, name, bottomNameList = []):
        netList = [self.net[x] for x in bottomNameList]
        testnetList = [self.testnet[x] for x in bottomNameList]
        self.net[name] = L.Eltwise(*netList, operation = 0)
        self.testnet[name] = L.Eltwise(*testnetList, operation = 0)

    def EltProduct(self, name, bottom0Name, bottom1Name):
        self.net[name] = L.Eltwise(self.net[bottom0Name], self.net[bottom1Name], operation = 0)
        self.testnet[name] = L.Eltwise(self.testnet[bottom0Name], self.testnet[bottom1Name], operation = 0)

    def EltDivide(self, name, bottom0Name, bottom1Name):
        self.Power(name + '/Inv', bottom1Name, -1, 1, 1e-8)
        self.EltProduct(name, bottom0Name, name + '/Inv')

    def Split2(self, name1, name2, bottomName, splitIndex):
        self.net[name1], self.net[name2] = L.Slice(self.net[bottomName], slice_point = splitIndex, ntop = 2)    
        self.testnet[name1], self.testnet[name2] = L.Slice(self.testnet[bottomName], slice_point = splitIndex, ntop = 2)
          
class BRDFNetClassLogLoss_Single_SplitChannal_New_Ratio(MyNetClass):

    def __init__(self):
        MyNetClass.__init__(self)

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	

    def createNet(self, batchSize = 64, loss_channal = 3, BN = False, normalize = False):
        
        lossweight_d = 1.0 if loss_channal == 0 else 0
        lossweight_s = 1.0 if loss_channal == 1 else 0
        lossweight_r = 1.0 if loss_channal == 2 else 0
        lossweight_t = 1.0 if loss_channal == 3 else 0
        lossweight_ratio = 1.0 if loss_channal == 4 else 0        

        width = 128
        height = 128
        channal = 1

        nBRDFChannal = 2
        nFirstFC = 2048
        nFilterFirstConv = 16

        self.Data(batchSize, channal, height, width, 'Data_Image')
        self.Data(batchSize, nBRDFChannal, 1, 1 , 'Data_BRDF')
        self.Split2('Data_Ratio', 'Data_Roughness', 'Data_BRDF', 1)

        #Conv
        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0', 'Data_Image', BN)            #128*128*16
        self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1', 'Conv0', BN)    #64*64*32
        self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2', 'Conv1', BN)    #32*32*64
        self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3', 'Conv2', BN)    #16*16*128
        self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4', 'Conv3', BN)   #8*8*256
        self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'Conv5', 'Conv4', BN)            #8*8*256

          
        self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0', 'Conv5', BN)
        self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1', 'MidConv0', BN)
        self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2', 'MidConv1', BN)

        self.FCReLU(nFirstFC, 'FCReLU_0', 'MidConv2')
        self.FCReLU(nFirstFC / 2, 'FCReLU_1', 'FCReLU_0')
        self.FCReLU(nFirstFC / 4, 'FCReLU_2', 'FCReLU_1')
        self.FC(2, 'FC', 'FCReLU_2')                        #out d/s ratio and roughness        
        self.Split2('Out_Ratio', 'Out_Roughness_Fix', 'FC', 1)
        
        self.MSELoss('RatioLoss', 'Data_Ratio', 'Out_Ratio', lossweight_ratio + lossweight_t)
        self.MSELoss('RoughnessLoss', 'Data_Roughness', 'Out_Roughness_Fix', lossweight_r + lossweight_t)

class SVBRDFNetClass_Decompose_FC_SR_Sigmoid_AN(MyNetClass):

    params = {}

    params['nFilterFirstConv'] = 16
    params['withBN'] = True

    def __init__(self, lr_a = 1.0, lr_s = 1.0, lr_r = 1.0, lr_n = 1.0):
        MyNetClass.__init__(self)
        self.lr_a = lr_a
        self.lr_s = lr_s
        self.lr_r = lr_r
        self.lr_n = lr_n

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	
    
    def SVBRDFData(self, batchSize, inputChannal = 3):
        height = 256
        width = 256
                
        self.net['Data_Image'] = L.Input(shape=[dict(dim=[batchSize, inputChannal, height, width])])
        self.net['Data_Albedo'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])
        self.net['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[batchSize, 3, 1, 1])])
        self.net['Data_Roughness'] = L.Input(shape=[dict(dim=[batchSize, 1, 1, 1])])       
        self.net['Data_Normal'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])

        self.testnet['Data_Image'] = L.Input(shape=[dict(dim=[1, inputChannal, height, width])])
        self.testnet['Data_Albedo'] = L.Input(shape=[dict(dim=[1, 3, height, width])])
        self.testnet['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[1, 3, 1, 1])])
        self.testnet['Data_Roughness'] = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        self.testnet['Data_Normal'] = L.Input(shape=[dict(dim=[1, 3, height, width])])

    def createNet(self, inputchannal = 3, batchSize = 16, BN = False, nFirstConv = 16, weightMSE = [1.0,1.0,1.0,1.0]):
        self.SVBRDFData(batchSize, inputchannal)
        nFilterFirstConv = nFirstConv

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1]
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        #Conv
        for i in range(0, 4):
            self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0_{}'.format(i), 'Data_Image', BN)            #256*256*16
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1_{}'.format(i), 'Conv0_{}'.format(i), BN)    #128*128*32
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2_{}'.format(i), 'Conv1_{}'.format(i), BN)    #64*64*64
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3_{}'.format(i), 'Conv2_{}'.format(i), BN)    #32*32*128
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4_{}'.format(i), 'Conv3_{}'.format(i), BN)   #16*16*256
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv5_{}'.format(i), 'Conv4_{}'.format(i), BN)   #8*8*256
            

        #Diffuse and Normal:

        for i in [0, 3]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256                        
            MidConvList = ['Conv5_{}'.format(i), 'MidConv2_ch{}'.format(i)]
            self.Concat('MidConv_ch{}_Merged'.format(i), MidConvList)

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 16, 3, 3, 'DeConv0_{}'.format(i), 'MidConv_ch{}_Merged'.format(i), BN)
            DeConv0List = ['Conv4_{}'.format(i), 'DeConv0_{}'.format(i)]
            self.Concat('DeConv0_ch{}_Merged'.format(i), DeConv0List)          #16*16*256                    

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv* 8, 3, 3, 'DeConv1_{}'.format(i), 'DeConv0_ch{}_Merged'.format(i), BN)
            DeConv1List = ['Conv3_{}'.format(i), 'DeConv1_{}'.format(i)]
            self.Concat('DeConv1_ch{}_Merged'.format(i), DeConv1List)          #32*32*128

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 4, 3, 3, 'DeConv2_{}'.format(i), 'DeConv1_ch{}_Merged'.format(i), BN)
            DeConv2List = ['Conv2_{}'.format(i), 'DeConv2_{}'.format(i)]
            self.Concat('DeConv2_ch{}_Merged'.format(i), DeConv2List)          #64*64*64

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 2, 3, 3, 'DeConv3_{}'.format(i), 'DeConv2_ch{}_Merged'.format(i), BN)
            DeConv3List = ['Conv1_{}'.format(i), 'DeConv3_{}'.format(i)]
            self.Concat('DeConv3_ch{}_Merged'.format(i), DeConv3List)          #128*128*32

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv, 3, 3, 'DeConv4_{}'.format(i), 'DeConv3_ch{}_Merged'.format(i), BN)
            DeConv4List = ['Conv0_{}'.format(i), 'DeConv4_{}'.format(i)]
            self.Concat('DeConv4_ch{}_Merged'.format(i), DeConv4List)          #256*256*16

        #Spec and roughness
        nFirstFC = 1024
        for i in [1, 2]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256   

            self.FCReLU(nFirstFC, 'FCReLU_0_{}'.format(i), 'MidConv2_ch{}'.format(i))
            self.FCReLU(nFirstFC / 2, 'FCReLU_1_{}'.format(i), 'FCReLU_0_{}'.format(i))
            if(i == 1):
                self.FC(3, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))
                #self.Sigmoid('Out_{}'.format(i), 'FC_{}'.format(i))
            else:
                self.FC(1, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))                                             
                #self.Sigmoid('Out_{}'.format(i), 'FC_{}'.format(i))

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_0', 'DeConv4_ch0_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Albedo', 'ConvFinal0_0', False)

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_3', 'DeConv4_ch3_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Normal', 'ConvFinal0_3', False)

        self.Power('ConvFinal_SpecAlbedo', 'FC_1', 1, 1, 0) #self.Sigmoid('ConvFinal_SpecAlbedo', 'FC_1', False)
        self.Power('ConvFinal_Roughness', 'FC_2', 1, 1, 0)
        
        self.MSELoss('MSELoss_Albedo', 'Data_Albedo', 'ConvFinal_Albedo', lossweight_d)
        self.MSELoss('MSELoss_SpecAlbedo', 'Data_SpecAlbedo', 'ConvFinal_SpecAlbedo', lossweight_s)
        self.MSELoss('MSELoss_Roughness', 'Data_Roughness', 'ConvFinal_Roughness', lossweight_r)
        self.MSELoss('MSELoss_Normal', 'Data_Normal', 'ConvFinal_Normal', lossweight_n)

class SVBRDFNetClass_Share_FC_SR_Sigmoid_AN(MyNetClass):

    params = {}

    params['nFilterFirstConv'] = 16
    params['withBN'] = True

    def __init__(self, lr_a = 1.0, lr_s = 1.0, lr_r = 1.0, lr_n = 1.0):
        MyNetClass.__init__(self)
        self.lr_a = lr_a
        self.lr_s = lr_s
        self.lr_r = lr_r
        self.lr_n = lr_n

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	
    
    def SVBRDFData(self, batchSize, inputChannal = 3):
        height = 256
        width = 256
                
        self.net['Data_Image'] = L.Input(shape=[dict(dim=[batchSize, inputChannal, height, width])])
        self.net['Data_Albedo'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])
        self.net['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[batchSize, 3, 1, 1])])
        self.net['Data_Roughness'] = L.Input(shape=[dict(dim=[batchSize, 1, 1, 1])])       
        self.net['Data_Normal'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])

        self.testnet['Data_Image'] = L.Input(shape=[dict(dim=[1, inputChannal, height, width])])
        self.testnet['Data_Albedo'] = L.Input(shape=[dict(dim=[1, 3, height, width])])
        self.testnet['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[1, 3, 1, 1])])
        self.testnet['Data_Roughness'] = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        self.testnet['Data_Normal'] = L.Input(shape=[dict(dim=[1, 3, height, width])])

    def createNet(self, inputchannal = 3, batchSize = 16, BN = False, nFirstConv = 16, weightMSE = [1.0,1.0,1.0,1.0]):
        self.SVBRDFData(batchSize, inputchannal)
        nFilterFirstConv = nFirstConv

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1] 
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        #Conv
        for i in range(0, 2):
            self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0_{}'.format(i), 'Data_Image', BN)            #256*256*16
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1_{}'.format(i), 'Conv0_{}'.format(i), BN)    #128*128*32
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2_{}'.format(i), 'Conv1_{}'.format(i), BN)    #64*64*64
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3_{}'.format(i), 'Conv2_{}'.format(i), BN)    #32*32*128
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4_{}'.format(i), 'Conv3_{}'.format(i), BN)   #16*16*256
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv5_{}'.format(i), 'Conv4_{}'.format(i), BN)   #8*8*256
            

        #Diffuse and Normal:

        for i in [0, 3]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_0', BN)
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256  
    
            MidConvList = ['Conv5_0', 'MidConv2_ch{}'.format(i)]
            self.Concat('MidConv_ch{}_Merged'.format(i), MidConvList)

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 16, 3, 3, 'DeConv0_{}'.format(i), 'MidConv_ch{}_Merged'.format(i), BN)
            DeConv0List = ['Conv4_0', 'DeConv0_{}'.format(i)]
            self.Concat('DeConv0_ch{}_Merged'.format(i), DeConv0List)          #16*16*256                    

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv* 8, 3, 3, 'DeConv1_{}'.format(i), 'DeConv0_ch{}_Merged'.format(i), BN)
            DeConv1List = ['Conv3_0', 'DeConv1_{}'.format(i)]
            self.Concat('DeConv1_ch{}_Merged'.format(i), DeConv1List)          #32*32*128

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 4, 3, 3, 'DeConv2_{}'.format(i), 'DeConv1_ch{}_Merged'.format(i), BN)
            DeConv2List = ['Conv2_0', 'DeConv2_{}'.format(i)]
            self.Concat('DeConv2_ch{}_Merged'.format(i), DeConv2List)          #64*64*64

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 2, 3, 3, 'DeConv3_{}'.format(i), 'DeConv2_ch{}_Merged'.format(i), BN)
            DeConv3List = ['Conv1_0', 'DeConv3_{}'.format(i)]
            self.Concat('DeConv3_ch{}_Merged'.format(i), DeConv3List)          #128*128*32

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv, 3, 3, 'DeConv4_{}'.format(i), 'DeConv3_ch{}_Merged'.format(i), BN)
            DeConv4List = ['Conv0_0', 'DeConv4_{}'.format(i)]
            self.Concat('DeConv4_ch{}_Merged'.format(i), DeConv4List)          #256*256*16

        #Spec and roughness
        nFirstFC = 1024
        for i in [1, 2]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_1', BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256   

            self.FCReLU(nFirstFC, 'FCReLU_0_{}'.format(i), 'MidConv2_ch{}'.format(i))
            self.FCReLU(nFirstFC / 2, 'FCReLU_1_{}'.format(i), 'FCReLU_0_{}'.format(i))
            if(i == 1):
                self.FC(3, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))
            else:
                self.FC(1, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))                                             

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_0', 'DeConv4_ch0_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Albedo', 'ConvFinal0_0', False)

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_3', 'DeConv4_ch3_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Normal', 'ConvFinal0_3', False)

        self.Power('ConvFinal_SpecAlbedo', 'FC_1', 1, 1, 0)
        self.Power('ConvFinal_Roughness', 'FC_2', 1, 1, 0)
        
        self.MSELoss('MSELoss_Albedo', 'Data_Albedo', 'ConvFinal_Albedo', lossweight_d)
        self.MSELoss('MSELoss_SpecAlbedo', 'Data_SpecAlbedo', 'ConvFinal_SpecAlbedo', lossweight_s)
        self.MSELoss('MSELoss_Roughness', 'Data_Roughness', 'ConvFinal_Roughness', lossweight_r)
        self.MSELoss('MSELoss_Normal', 'Data_Normal', 'ConvFinal_Normal', lossweight_n + lossweight_tn)

class SVBRDFNetClass_ShareAll_FC_SR_Sigmoid_AN(MyNetClass):

    params = {}

    params['nFilterFirstConv'] = 16
    params['withBN'] = True

    def __init__(self, lr_a = 1.0, lr_s = 1.0, lr_r = 1.0, lr_n = 1.0):
        MyNetClass.__init__(self)
        self.lr_a = lr_a
        self.lr_s = lr_s
        self.lr_r = lr_r
        self.lr_n = lr_n

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	
    
    def SVBRDFData(self, batchSize, inputChannal = 3):
        height = 256
        width = 256
                
        self.net['Data_Image'] = L.Input(shape=[dict(dim=[batchSize, inputChannal, height, width])])
        self.net['Data_Albedo'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])
        self.net['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[batchSize, 3, 1, 1])])
        self.net['Data_Roughness'] = L.Input(shape=[dict(dim=[batchSize, 1, 1, 1])])       
        self.net['Data_Normal'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])

        self.testnet['Data_Image'] = L.Input(shape=[dict(dim=[1, inputChannal, height, width])])
        self.testnet['Data_Albedo'] = L.Input(shape=[dict(dim=[1, 3, height, width])])
        self.testnet['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[1, 3, 1, 1])])
        self.testnet['Data_Roughness'] = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        self.testnet['Data_Normal'] = L.Input(shape=[dict(dim=[1, 3, height, width])])

    def createNet(self, inputchannal = 3, batchSize = 16, BN = False, nFirstConv = 16, weightMSE = [1.0,1.0,1.0,1.0]):
        self.SVBRDFData(batchSize, inputchannal)
        nFilterFirstConv = nFirstConv

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1] 
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        #Conv
        for i in range(0, 1):
            self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0_{}'.format(i), 'Data_Image', BN)            #256*256*16
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1_{}'.format(i), 'Conv0_{}'.format(i), BN)    #128*128*32
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2_{}'.format(i), 'Conv1_{}'.format(i), BN)    #64*64*64
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3_{}'.format(i), 'Conv2_{}'.format(i), BN)    #32*32*128
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4_{}'.format(i), 'Conv3_{}'.format(i), BN)   #16*16*256
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv5_{}'.format(i), 'Conv4_{}'.format(i), BN)   #8*8*256
            

        #Diffuse and Normal:

        for i in [0, 3]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_0', BN)
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256  
    
            MidConvList = ['Conv5_0', 'MidConv2_ch{}'.format(i)]
            self.Concat('MidConv_ch{}_Merged'.format(i), MidConvList)

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 16, 3, 3, 'DeConv0_{}'.format(i), 'MidConv_ch{}_Merged'.format(i), BN)
            DeConv0List = ['Conv4_0', 'DeConv0_{}'.format(i)]
            self.Concat('DeConv0_ch{}_Merged'.format(i), DeConv0List)          #16*16*256                    

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv* 8, 3, 3, 'DeConv1_{}'.format(i), 'DeConv0_ch{}_Merged'.format(i), BN)
            DeConv1List = ['Conv3_0', 'DeConv1_{}'.format(i)]
            self.Concat('DeConv1_ch{}_Merged'.format(i), DeConv1List)          #32*32*128

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 4, 3, 3, 'DeConv2_{}'.format(i), 'DeConv1_ch{}_Merged'.format(i), BN)
            DeConv2List = ['Conv2_0', 'DeConv2_{}'.format(i)]
            self.Concat('DeConv2_ch{}_Merged'.format(i), DeConv2List)          #64*64*64

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 2, 3, 3, 'DeConv3_{}'.format(i), 'DeConv2_ch{}_Merged'.format(i), BN)
            DeConv3List = ['Conv1_0', 'DeConv3_{}'.format(i)]
            self.Concat('DeConv3_ch{}_Merged'.format(i), DeConv3List)          #128*128*32

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv, 3, 3, 'DeConv4_{}'.format(i), 'DeConv3_ch{}_Merged'.format(i), BN)
            DeConv4List = ['Conv0_0', 'DeConv4_{}'.format(i)]
            self.Concat('DeConv4_ch{}_Merged'.format(i), DeConv4List)          #256*256*16

        #Spec and roughness
        nFirstFC = 1024
        for i in [1, 2]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_0', BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256   

            self.FCReLU(nFirstFC, 'FCReLU_0_{}'.format(i), 'MidConv2_ch{}'.format(i))
            self.FCReLU(nFirstFC / 2, 'FCReLU_1_{}'.format(i), 'FCReLU_0_{}'.format(i))
            if(i == 1):
                self.FC(3, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))
            else:
                self.FC(1, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))                                             

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_0', 'DeConv4_ch0_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Albedo', 'ConvFinal0_0', False)

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_3', 'DeConv4_ch3_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Normal', 'ConvFinal0_3', False)

        self.Power('ConvFinal_SpecAlbedo', 'FC_1', 1, 1, 0)
        self.Power('ConvFinal_Roughness', 'FC_2', 1, 1, 0)
        
        self.MSELoss('MSELoss_Albedo', 'Data_Albedo', 'ConvFinal_Albedo', lossweight_d)
        self.MSELoss('MSELoss_SpecAlbedo', 'Data_SpecAlbedo', 'ConvFinal_SpecAlbedo', lossweight_s)
        self.MSELoss('MSELoss_Roughness', 'Data_Roughness', 'ConvFinal_Roughness', lossweight_r)
        self.MSELoss('MSELoss_Normal', 'Data_Normal', 'ConvFinal_Normal', lossweight_n)

class SVBRDFNetClass_Decompose_FC_SRN_ThetaPhi_Sigmoid_A(MyNetClass):
    #Normal 2-Channal: [Theta, Phi]
    params = {}

    params['nFilterFirstConv'] = 16
    params['withBN'] = True

    def __init__(self, lr_a = 1.0, lr_s = 1.0, lr_r = 1.0, lr_n = 1.0):
        MyNetClass.__init__(self)
        self.lr_a = lr_a
        self.lr_s = lr_s
        self.lr_r = lr_r
        self.lr_n = lr_n

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	
    
    def SVBRDFData(self, batchSize, inputChannal = 3):
        height = 256
        width = 256
                
        self.net['Data_Image'] = L.Input(shape=[dict(dim=[batchSize, inputChannal, height, width])])
        self.net['Data_Albedo'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])
        self.net['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[batchSize, 3, 1, 1])])
        self.net['Data_Roughness'] = L.Input(shape=[dict(dim=[batchSize, 1, 1, 1])])       
        self.net['Data_Normal'] = L.Input(shape=[dict(dim=[batchSize, 2, height, width])])

        self.testnet['Data_Image'] = L.Input(shape=[dict(dim=[1, inputChannal, height, width])])
        self.testnet['Data_Albedo'] = L.Input(shape=[dict(dim=[1, 3, height, width])])
        self.testnet['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[1, 3, 1, 1])])
        self.testnet['Data_Roughness'] = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        self.testnet['Data_Normal'] = L.Input(shape=[dict(dim=[1, 2, height, width])])

    def createNet(self, inputchannal = 3, batchSize = 16, BN = False, nFirstConv = 16, weightMSE = [1.0,1.0,1.0,1.0]):
        self.SVBRDFData(batchSize, inputchannal)
        nFilterFirstConv = nFirstConv

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1] 
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        #Conv
        for i in range(0, 4):
            self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0_{}'.format(i), 'Data_Image', BN)            #256*256*16
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1_{}'.format(i), 'Conv0_{}'.format(i), BN)    #128*128*32
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2_{}'.format(i), 'Conv1_{}'.format(i), BN)    #64*64*64
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3_{}'.format(i), 'Conv2_{}'.format(i), BN)    #32*32*128
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4_{}'.format(i), 'Conv3_{}'.format(i), BN)   #16*16*256
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv5_{}'.format(i), 'Conv4_{}'.format(i), BN)   #8*8*256
            

        #Diffuse and Normal:

        for i in [0, 3]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256                        
            MidConvList = ['Conv5_{}'.format(i), 'MidConv2_ch{}'.format(i)]
            self.Concat('MidConv_ch{}_Merged'.format(i), MidConvList)

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 16, 3, 3, 'DeConv0_{}'.format(i), 'MidConv_ch{}_Merged'.format(i), BN)
            DeConv0List = ['Conv4_{}'.format(i), 'DeConv0_{}'.format(i)]
            self.Concat('DeConv0_ch{}_Merged'.format(i), DeConv0List)          #16*16*256                    

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv* 8, 3, 3, 'DeConv1_{}'.format(i), 'DeConv0_ch{}_Merged'.format(i), BN)
            DeConv1List = ['Conv3_{}'.format(i), 'DeConv1_{}'.format(i)]
            self.Concat('DeConv1_ch{}_Merged'.format(i), DeConv1List)          #32*32*128

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 4, 3, 3, 'DeConv2_{}'.format(i), 'DeConv1_ch{}_Merged'.format(i), BN)
            DeConv2List = ['Conv2_{}'.format(i), 'DeConv2_{}'.format(i)]
            self.Concat('DeConv2_ch{}_Merged'.format(i), DeConv2List)          #64*64*64

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 2, 3, 3, 'DeConv3_{}'.format(i), 'DeConv2_ch{}_Merged'.format(i), BN)
            DeConv3List = ['Conv1_{}'.format(i), 'DeConv3_{}'.format(i)]
            self.Concat('DeConv3_ch{}_Merged'.format(i), DeConv3List)          #128*128*32

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv, 3, 3, 'DeConv4_{}'.format(i), 'DeConv3_ch{}_Merged'.format(i), BN)
            DeConv4List = ['Conv0_{}'.format(i), 'DeConv4_{}'.format(i)]
            self.Concat('DeConv4_ch{}_Merged'.format(i), DeConv4List)          #256*256*16

        #Spec and roughness
        nFirstFC = 1024
        for i in [1, 2]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256   

            self.FCReLU(nFirstFC, 'FCReLU_0_{}'.format(i), 'MidConv2_ch{}'.format(i))
            self.FCReLU(nFirstFC / 2, 'FCReLU_1_{}'.format(i), 'FCReLU_0_{}'.format(i))
            if(i == 1):
                self.FC(3, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))
            else:
                self.FC(1, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))                                             

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_0', 'DeConv4_ch0_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Albedo', 'ConvFinal0_0', False)

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_3', 'DeConv4_ch3_Merged', BN)
        self.ConvSameResolution_FCOnly(2, 3, 3, 'ConvFinal_Normal', 'ConvFinal0_3', False)

        self.Power('ConvFinal_SpecAlbedo', 'FC_1', 1, 1, 0)
        self.Power('ConvFinal_Roughness', 'FC_2', 1, 1, 0)
        
        self.MSELoss('MSELoss_Albedo', 'Data_Albedo', 'ConvFinal_Albedo', lossweight_d)
        self.MSELoss('MSELoss_SpecAlbedo', 'Data_SpecAlbedo', 'ConvFinal_SpecAlbedo', lossweight_s)
        self.MSELoss('MSELoss_Roughness', 'Data_Roughness', 'ConvFinal_Roughness', lossweight_r)
        self.MSELoss('MSELoss_Normal', 'Data_Normal', 'ConvFinal_Normal', lossweight_n)

class SVBRDFNetClass_ShareAll_FC_SRN_ThetaPhi_Sigmoid_A(MyNetClass):
    #Normal 2-Channal: [Theta, Phi]
    params = {}

    params['nFilterFirstConv'] = 16
    params['withBN'] = True

    def __init__(self, lr_a = 1.0, lr_s = 1.0, lr_r = 1.0, lr_n = 1.0):
        MyNetClass.__init__(self)
        self.lr_a = lr_a
        self.lr_s = lr_s
        self.lr_r = lr_r
        self.lr_n = lr_n

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	
    
    def SVBRDFData(self, batchSize, inputChannal = 3):
        height = 256
        width = 256
                
        self.net['Data_Image'] = L.Input(shape=[dict(dim=[batchSize, inputChannal, height, width])])
        self.net['Data_Albedo'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])
        self.net['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[batchSize, 3, 1, 1])])
        self.net['Data_Roughness'] = L.Input(shape=[dict(dim=[batchSize, 1, 1, 1])])       
        self.net['Data_Normal'] = L.Input(shape=[dict(dim=[batchSize, 2, height, width])])

        self.testnet['Data_Image'] = L.Input(shape=[dict(dim=[1, inputChannal, height, width])])
        self.testnet['Data_Albedo'] = L.Input(shape=[dict(dim=[1, 3, height, width])])
        self.testnet['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[1, 3, 1, 1])])
        self.testnet['Data_Roughness'] = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        self.testnet['Data_Normal'] = L.Input(shape=[dict(dim=[1, 2, height, width])])

    def createNet(self, inputchannal = 3, batchSize = 16, BN = False, nFirstConv = 16, weightMSE = [1.0,1.0,1.0,1.0]):
        self.SVBRDFData(batchSize, inputchannal)
        nFilterFirstConv = nFirstConv

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1] 
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        #Conv
        for i in range(0, 1):
            self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0_{}'.format(i), 'Data_Image', BN)            #256*256*16
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1_{}'.format(i), 'Conv0_{}'.format(i), BN)    #128*128*32
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2_{}'.format(i), 'Conv1_{}'.format(i), BN)    #64*64*64
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3_{}'.format(i), 'Conv2_{}'.format(i), BN)    #32*32*128
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4_{}'.format(i), 'Conv3_{}'.format(i), BN)   #16*16*256
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv5_{}'.format(i), 'Conv4_{}'.format(i), BN)   #8*8*256
            

        #Diffuse and Normal:

        for i in [0, 3]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_0', BN)
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256  
    
            MidConvList = ['Conv5_0', 'MidConv2_ch{}'.format(i)]
            self.Concat('MidConv_ch{}_Merged'.format(i), MidConvList)

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 16, 3, 3, 'DeConv0_{}'.format(i), 'MidConv_ch{}_Merged'.format(i), BN)
            DeConv0List = ['Conv4_0', 'DeConv0_{}'.format(i)]
            self.Concat('DeConv0_ch{}_Merged'.format(i), DeConv0List)          #16*16*256                    

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv* 8, 3, 3, 'DeConv1_{}'.format(i), 'DeConv0_ch{}_Merged'.format(i), BN)
            DeConv1List = ['Conv3_0', 'DeConv1_{}'.format(i)]
            self.Concat('DeConv1_ch{}_Merged'.format(i), DeConv1List)          #32*32*128

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 4, 3, 3, 'DeConv2_{}'.format(i), 'DeConv1_ch{}_Merged'.format(i), BN)
            DeConv2List = ['Conv2_0', 'DeConv2_{}'.format(i)]
            self.Concat('DeConv2_ch{}_Merged'.format(i), DeConv2List)          #64*64*64

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 2, 3, 3, 'DeConv3_{}'.format(i), 'DeConv2_ch{}_Merged'.format(i), BN)
            DeConv3List = ['Conv1_0', 'DeConv3_{}'.format(i)]
            self.Concat('DeConv3_ch{}_Merged'.format(i), DeConv3List)          #128*128*32

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv, 3, 3, 'DeConv4_{}'.format(i), 'DeConv3_ch{}_Merged'.format(i), BN)
            DeConv4List = ['Conv0_0', 'DeConv4_{}'.format(i)]
            self.Concat('DeConv4_ch{}_Merged'.format(i), DeConv4List)          #256*256*16

        #Spec and roughness
        nFirstFC = 1024
        for i in [1, 2]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_0', BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256   

            self.FCReLU(nFirstFC, 'FCReLU_0_{}'.format(i), 'MidConv2_ch{}'.format(i))
            self.FCReLU(nFirstFC / 2, 'FCReLU_1_{}'.format(i), 'FCReLU_0_{}'.format(i))
            if(i == 1):
                self.FC(3, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))
            else:
                self.FC(1, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))                                             

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_0', 'DeConv4_ch0_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Albedo', 'ConvFinal0_0', False)

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_3', 'DeConv4_ch3_Merged', BN)
        self.ConvSameResolution_FCOnly(2, 3, 3, 'ConvFinal_Normal', 'ConvFinal0_3', False)

        self.Power('ConvFinal_SpecAlbedo', 'FC_1', 1, 1, 0)
        self.Power('ConvFinal_Roughness', 'FC_2', 1, 1, 0)
        
        self.MSELoss('MSELoss_Albedo', 'Data_Albedo', 'ConvFinal_Albedo', lossweight_d)
        self.MSELoss('MSELoss_SpecAlbedo', 'Data_SpecAlbedo', 'ConvFinal_SpecAlbedo', lossweight_s)
        self.MSELoss('MSELoss_Roughness', 'Data_Roughness', 'ConvFinal_Roughness', lossweight_r)
        self.MSELoss('MSELoss_Normal', 'Data_Normal', 'ConvFinal_Normal', lossweight_n)

class SVBRDFNetClass_Share_FC_SRN_ThetaPhi_Sigmoid_A(MyNetClass):
    #Normal 2-Channal: [Theta, Phi]
    params = {}

    params['nFilterFirstConv'] = 16
    params['withBN'] = True

    def __init__(self, lr_a = 1.0, lr_s = 1.0, lr_r = 1.0, lr_n = 1.0):
        MyNetClass.__init__(self)
        self.lr_a = lr_a
        self.lr_s = lr_s
        self.lr_r = lr_r
        self.lr_n = lr_n

    def saveNet(self, netFolderPath = ''):
        if(netFolderPath != ''):
            netFolderPath = netFolderPath + '/'
        with open(netFolderPath + 'net.prototxt', 'w') as f:
            f.write(str(self.net.to_proto()))

        with open(netFolderPath + 'net_test.prototxt', 'w') as f:
            f.write(str(self.testnet.to_proto()))	
    
    def SVBRDFData(self, batchSize, inputChannal = 3):
        height = 256
        width = 256
                
        self.net['Data_Image'] = L.Input(shape=[dict(dim=[batchSize, inputChannal, height, width])])
        self.net['Data_Albedo'] = L.Input(shape=[dict(dim=[batchSize, 3, height, width])])
        self.net['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[batchSize, 3, 1, 1])])
        self.net['Data_Roughness'] = L.Input(shape=[dict(dim=[batchSize, 1, 1, 1])])       
        self.net['Data_Normal'] = L.Input(shape=[dict(dim=[batchSize, 2, height, width])])

        self.testnet['Data_Image'] = L.Input(shape=[dict(dim=[1, inputChannal, height, width])])
        self.testnet['Data_Albedo'] = L.Input(shape=[dict(dim=[1, 3, height, width])])
        self.testnet['Data_SpecAlbedo'] = L.Input(shape=[dict(dim=[1, 3, 1, 1])])
        self.testnet['Data_Roughness'] = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        self.testnet['Data_Normal'] = L.Input(shape=[dict(dim=[1, 2, height, width])])

    def createNet(self, inputchannal = 3, batchSize = 16, BN = False, nFirstConv = 16, weightMSE = [1.0,1.0,1.0,1.0]):
        self.SVBRDFData(batchSize, inputchannal)
        nFilterFirstConv = nFirstConv

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1] 
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        #Conv
        for i in range(0, 2):
            self.ConvSameResolution(nFilterFirstConv, 3, 3, 'Conv0_{}'.format(i), 'Data_Image', BN)            #256*256*16
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 2, 3, 3, 'Conv1_{}'.format(i), 'Conv0_{}'.format(i), BN)    #128*128*32
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 4, 3, 3, 'Conv2_{}'.format(i), 'Conv1_{}'.format(i), BN)    #64*64*64
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 8, 3, 3, 'Conv3_{}'.format(i), 'Conv2_{}'.format(i), BN)    #32*32*128
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv4_{}'.format(i), 'Conv3_{}'.format(i), BN)   #16*16*256
            self.ConvHalfResolutionNoPooling(nFilterFirstConv * 16, 3, 3, 'Conv5_{}'.format(i), 'Conv4_{}'.format(i), BN)   #8*8*256
           

        #Diffuse and Normal:

        for i in [0, 3]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_0', BN)
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256  
    
            MidConvList = ['Conv5_0', 'MidConv2_ch{}'.format(i)]
            self.Concat('MidConv_ch{}_Merged'.format(i), MidConvList)

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 16, 3, 3, 'DeConv0_{}'.format(i), 'MidConv_ch{}_Merged'.format(i), BN)
            DeConv0List = ['Conv4_0', 'DeConv0_{}'.format(i)]
            self.Concat('DeConv0_ch{}_Merged'.format(i), DeConv0List)          #16*16*256                    

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv* 8, 3, 3, 'DeConv1_{}'.format(i), 'DeConv0_ch{}_Merged'.format(i), BN)
            DeConv1List = ['Conv3_0', 'DeConv1_{}'.format(i)]
            self.Concat('DeConv1_ch{}_Merged'.format(i), DeConv1List)          #32*32*128

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 4, 3, 3, 'DeConv2_{}'.format(i), 'DeConv1_ch{}_Merged'.format(i), BN)
            DeConv2List = ['Conv2_0', 'DeConv2_{}'.format(i)]
            self.Concat('DeConv2_ch{}_Merged'.format(i), DeConv2List)          #64*64*64

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv * 2, 3, 3, 'DeConv3_{}'.format(i), 'DeConv2_ch{}_Merged'.format(i), BN)
            DeConv3List = ['Conv1_0', 'DeConv3_{}'.format(i)]
            self.Concat('DeConv3_ch{}_Merged'.format(i), DeConv3List)          #128*128*32

            self.DeConvDoubleResolutionBilinear(nFilterFirstConv, 3, 3, 'DeConv4_{}'.format(i), 'DeConv3_ch{}_Merged'.format(i), BN)
            DeConv4List = ['Conv0_0', 'DeConv4_{}'.format(i)]
            self.Concat('DeConv4_ch{}_Merged'.format(i), DeConv4List)          #256*256*16

        #Spec and roughness
        nFirstFC = 1024
        for i in [1, 2]:
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv0_ch{}'.format(i), 'Conv5_1', BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv1_ch{}'.format(i), 'MidConv0_ch{}'.format(i), BN) #8*8*256
            self.ConvSameResolution(nFilterFirstConv * 16, 3, 3, 'MidConv2_ch{}'.format(i), 'MidConv1_ch{}'.format(i), BN) #8*8*256   

            self.FCReLU(nFirstFC, 'FCReLU_0_{}'.format(i), 'MidConv2_ch{}'.format(i))
            self.FCReLU(nFirstFC / 2, 'FCReLU_1_{}'.format(i), 'FCReLU_0_{}'.format(i))
            if(i == 1):
                self.FC(3, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))
            else:
                self.FC(1, 'FC_{}'.format(i), 'FCReLU_1_{}'.format(i))                                             

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_0', 'DeConv4_ch0_Merged', BN)
        self.ConvSameResolution_sigmoid(3, 3, 3, 'ConvFinal_Albedo', 'ConvFinal0_0', False)

        self.ConvSameResolution(nFilterFirstConv, 3, 3, 'ConvFinal0_3', 'DeConv4_ch3_Merged', BN)
        self.ConvSameResolution_FCOnly(2, 3, 3, 'ConvFinal_Normal', 'ConvFinal0_3', False)

        self.Power('ConvFinal_SpecAlbedo', 'FC_1', 1, 1, 0)
        self.Power('ConvFinal_Roughness', 'FC_2', 1, 1, 0)
        
        self.MSELoss('MSELoss_Albedo', 'Data_Albedo', 'ConvFinal_Albedo', lossweight_d)
        self.MSELoss('MSELoss_SpecAlbedo', 'Data_SpecAlbedo', 'ConvFinal_SpecAlbedo', lossweight_s)
        self.MSELoss('MSELoss_Roughness', 'Data_Roughness', 'ConvFinal_Roughness', lossweight_r)
        self.MSELoss('MSELoss_Normal', 'Data_Normal', 'ConvFinal_Normal', lossweight_n)


