# SVBRDFNetTraining.py
# Training script for SVBRDF-Net and SA-SVBRDF-Net

import random, os, sys, time, json, pickle, glob, math, shutil, logging
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)

sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')
from ConfigParser import ConfigParser, SafeConfigParser
from multiprocessing import Process
from multiprocessing import Queue as MultiQueue
from multiprocessing import Pipe

import caffe
import numpy as np
import cv2
import matplotlib.pyplot as plt


from NetClass import SVBRDFNetClass_Share_FC_SRN_ThetaPhi_Sigmoid_A, SVBRDFNetClass_Share_FC_SR_Sigmoid_AN, SVBRDFNetClass_Decompose_FC_SRN_ThetaPhi_Sigmoid_A, SVBRDFNetClass_Decompose_FC_SR_Sigmoid_AN, SVBRDFNetClass_ShareAll_FC_SRN_ThetaPhi_Sigmoid_A, SVBRDFNetClass_ShareAll_FC_SR_Sigmoid_AN
from utils import save_pfm, load_pfm, pfmFromBuffer, pfmToBuffer, toHDR, toLDR, renormalize, normalizeAlbedoSpec, normalBatchToThetaPhiBatch, thetaPhiBatchToNormalBatch, DataLoaderSVBRDF, RealDataLoaderSVBRDF, autoExposure, make_dir
from FastRendererCUDA import FastRenderEngine

os.chdir(working_path)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

params_global = {}

pixelCnt = 256 * 256

lightID = []

with open('folderPath_SVBRDF.txt', 'r') as f:
    params_global['geometryPath'] = r'../Render/plane.obj'
    params_global['scriptRoot'] = r'../Utils'
    params_global['outFolder'] = f.readline().strip()
    params_global['envMapFolder'] = f.readline().strip()


with open(params_global['envMapFolder'] + r'/light.txt', 'r') as f:
     lightID = map(int, f.read().strip().split('\n'))
     lightID = list(np.array(lightID) - 1)


def renderOnlineEnvlight(brdfBatch, onlineRender, params, lightIDs = [], lightXforms = [], lightNorms = []):
    imgBatch = np.zeros((brdfBatch.shape[0], 3, 256, 256))
    if(lightIDs == []):
        lightIDs = random.sample(params['lightID'], brdfBatch.shape[0])#brdfBatch.shape[0])#np.random.choice(80, brdfBatch.shape[0])
    if(lightXforms == []):
        angle_y = np.random.uniform(0.0, 360.0, brdfBatch.shape[0])
        angle_x = np.random.uniform(-45.0, 45.0, brdfBatch.shape[0])
    else:
        angle_y = lightXforms[1]
        angle_x = lightXforms[0]

    if(params['autoExposure'] == 2):
       normal_one = np.dstack((np.ones((256,256)), np.zeros((256,256)), np.zeros((256,256))))

    for i in range(0, brdfBatch.shape[0]):
        onlineRender.SetEnvLightByID(lightIDs[i] + 1) 
        onlineRender.SetLightXform(angle_x[i], angle_y[i])
                   
        onlineRender.SetAlbedoMap(brdfBatch[i,0:3,:,:].transpose((1,2,0)))
        onlineRender.SetSpecValue(brdfBatch[i,3:6,0,0])
        onlineRender.SetRoughnessValue(brdfBatch[i,6,0,0])

        if(brdfBatch.shape[1] > 7):
            onlineRender.SetNormalMap((2.0 * brdfBatch[i,7:10,:,:] - 1.0).transpose((1,2,0)))
        imgBatch[i, :, :, :] = onlineRender.Render().transpose((2,0,1))

        if(params['autoExposure'] == 1):
            imgBatch[i, 0, :, :] = imgBatch[i, 0, :, :] / lightNorms[i][0]
            imgBatch[i, 1, :, :] = imgBatch[i, 1, :, :] / lightNorms[i][1]
            imgBatch[i, 2, :, :] = imgBatch[i, 2, :, :] / lightNorms[i][2]
        elif(params['autoExposure'] == 2):
            onlineRender.SetAlbedoValue([1.0, 1.0, 1.0])
            onlineRender.SetSpecValue([0.0, 0.0, 0.0])
            onlineRender.SetNormalMap(normal_one)
            img_norm = onlineRender.Render()
            normValue = np.mean(img_norm, axis = (0, 1))
            imgBatch[i, 0, :, :] = imgBatch[i, 0, :, :] / normValue[0]
            imgBatch[i, 1, :, :] = imgBatch[i, 1, :, :] / normValue[1]
            imgBatch[i, 2, :, :] = imgBatch[i, 2, :, :] / normValue[2]
        imgBatch[i,:,:,:] = 0.5 * imgBatch[i,:,:,:]#autoExposure(imgBatch[i,:,:,:])

    return imgBatch

def DataLoadProcess_final(pipe, datafile, params, isTest = False):
    path, file = os.path.split(datafile)
    batchSize = 1 if isTest else params['batchSize']
    dataset = DataLoaderSVBRDF(path, file, 384, 384, not isTest)
    dataset.shuffle(params['randomSeed'])
    pipe.send(dataset.dataSize)
    counter = 0
    posInDataSet = 0
    epoch = 0
  
    if(params['LDR'] == 1):
       dataset.ldr = True       

    while(True):
        imgbatch, brdfbatch, name = dataset.GetBatchWithName(posInDataSet, batchSize)

        if(params['normalizeAlbedo']):
            brdfbatch = normalizeAlbedoSpec(brdfbatch)
                
        pipe.send((imgbatch, brdfbatch, name))
        counter = counter + batchSize
        posInDataSet = (posInDataSet + batchSize) % dataset.dataSize
        newepoch = counter / dataset.dataSize
        if(newepoch != epoch):
            dataset.shuffle()
        epoch = newepoch

def RealUnlabelDataLoadProcess(pipe, datafile, params):
    path, file = os.path.split(datafile)
    batchSize = params['batchSize']
    dataset = RealDataLoaderSVBRDF(path, file)
    
    dataset.shuffle(params['randomSeed'])
    pipe.send(dataset.dataSize)
    counter = 0
    posInDataSet = 0
    epoch = 0
  
    while(True):
        imgbatch = dataset.GetBatch(posInDataSet, batchSize)
        for i in range(0, batchSize):
            imgbatch[i,:,:,:] = autoExposure(imgbatch[i,:,:,:])
        pipe.send(imgbatch)
        counter = counter + batchSize
        posInDataSet = (posInDataSet + batchSize) % dataset.dataSize
        newepoch = counter / dataset.dataSize
        if(newepoch != epoch):
            dataset.shuffle()
        epoch = newepoch

def loadParams(filepath):
    config = SafeConfigParser({'albedoWeight':'1.0', 
                               'specWeight':'1.0', 
                               'roughnessWeight':'1.0', 
                               'normalWeight':'1.0',
                               'PreTrainNet':'',
                               'NetworkFile':'',
                               'LDR':'0', 
                               'loopRestartFrequency':'-1', 
                               'PreTrainSpecNet':'', 
                               'grayLight':'0', 
                               'normalizeAlbedo':'0', 
                               'PreTrainSpecNetNoLoop':'', 
                               'LogSpec':'0', 
                               'autoExposure':'0', 
                               'autoExposureLUTFile':'', 
                               'lightPoolFile':'', 
                               'NormalLoss':'L2'})
    config.read(filepath)

    params = {}
    #device
    params['randomSeed'] = config.getint('device', 'randomSeed')
    #solver
    params['SolverType'] = config.get('solver', 'SolverType')
    params['lr'] = config.getfloat('solver', 'lr')
    params['momentum'] = config.getfloat('solver', 'momentum')
    params['lrDecay'] = config.getfloat('solver', 'lrDecay')
    params['batchSize'] = config.getint('solver', 'batchSize')
    params['weightDecay'] = config.getfloat('solver', 'weightDecay')
    params['autoExposure'] = config.getint('solver', 'autoExposure')
    #stopping crteria
    params['nMaxEpoch'] = config.getint('stopping', 'nMaxEpoch')
    params['nMaxIter'] = config.getint('stopping', 'nMaxIter')
    #loop setting
    params['renderLoop'] = config.getboolean('loop', 'renderLoop')          #set to 0 to disable self-augmentation
    params['autoLoopRatio'] = config.getboolean('loop', 'autoLoopRatio')
    #the SA training would alternating between 'normalBatchLength' iteration of normal training and 'loopBatchLength' of self-augment training 
    params['normalBatchLength'] = config.getint('loop', 'normalBatchLength')
    params['loopStartEpoch'] = config.getint('loop', 'loopStartEpoch')
    params['loopStartIteration'] = config.getint('loop', 'loopStartIteration')  #add loop after this number of normal training.
    params['loopBatchLength'] = config.getint('loop', 'loopBatchLength')        #how many mini-batch iteration for ever loop optimize
    params['loopRestartFrequency'] = config.getint('loop', 'loopRestartFrequency')
    #network structure
    params['NetworkType'] = config.get('network', 'NetworkType')
    params['NetworkFile'] = config.get('network', 'NetworkFile')
    params['Channal'] = config.get('network', 'Channal')
    params['LogRoughness'] = config.getboolean('network', 'LogRoughness')
    params['LogSpec'] = config.getboolean('network', 'LogSpec')
    params['BN'] = config.getboolean('network', 'BN')
    params['DisableDecoder'] = config.getboolean('network', 'DisableDecoder')
    params['nFirstFeatureMap'] = config.getint('network', 'nFirstFeatureMap')
    params['NormalLoss'] = config.get('network', 'NormalLoss')  #L2 / ThetaPhi / DotProduct?    

    params['albedoWeight'] = config.getfloat('network', 'albedoWeight')
    params['specWeight'] = config.getfloat('network', 'specWeight')
    params['roughnessWeight'] = config.getfloat('network', 'roughnessWeight')
    params['normalWeight'] = config.getfloat('network', 'normalWeight')

    params['PreTrainNet'] = config.get('network', 'PreTrainNet')

    #dataset
    params['dataset'] = config.get('dataset', 'dataset')
    params['unlabelDataset'] = config.get('dataset', 'unlabelDataset')
    params['testDataset'] = config.get('dataset', 'testDataset')
    params['LDR'] = config.getint('dataset', 'LDR')         #0: only HDR; #1: only LDR; #2: LDR+HDR
    params['grayLight'] = config.getboolean('dataset', 'grayLight')
    params['normalizeAlbedo'] = config.getboolean('dataset', 'normalizeAlbedo')
    params['lightPoolFile'] = config.get('dataset', 'lightPoolFile')
    params['autoExposureLUTFile'] = config.get('dataset', 'autoExposureLUTFile')

    #display and testing
    params['displayStep'] = config.getint('display', 'displayStep')
    params['loopdisplayStep'] = config.getint('display', 'loopdisplayStep')
    params['checkPointStepIteration'] = config.getint('display', 'checkPointStepIteration')
    params['checkPointStepEpoch'] = config.getint('display', 'checkPointStepEpoch')
    params['visulizeStep'] = config.getint('display', 'visulizeStep')
    
    params['envMapFolder'] = params_global['envMapFolder']
    params['geometryPath'] = params_global['geometryPath']
    params['lightID'] = lightID

    params['lightIDToEnumerateID'] = {}
    for id, lid in enumerate(lightID):
        params['lightIDToEnumerateID'][lid] = id

    if(params['lightPoolFile'] != ''):
        params['lightPool'] = pickle.load(open(params_global['envMapFolder'] + r'/{}'.format(params['lightPoolFile']), 'rb'))

    if(params['autoExposureLUTFile'] != ''):
        params['autoExposureLUT'] = pickle.load(open(params_global['envMapFolder'] + r'/{}'.format(params['autoExposureLUTFile']), 'rb'))

    return params

def dumpNetwork(outfolder, solver, filename, statusDict):
    stateFiles = glob.glob(outfolder + r'/*.solverstate')
    for stateFile in stateFiles:
        os.remove(stateFile)
    solver.snapshot()
    files = glob.glob(outfolder + r'/*.caffemodel')
    files.sort(key=os.path.getmtime)
    if(os.path.exists(outfolder + r'/{}.caffemodel'.format(filename))):
        os.remove(outfolder + r'/{}.caffemodel'.format(filename))
    os.rename(files[-1], outfolder + r'/{}.caffemodel'.format(filename))

    files = glob.glob(outfolder + r'/*.solverstate')
    files.sort(key=os.path.getmtime)
    if(os.path.exists(outfolder + r'/{}.solverstate'.format(filename))):
        os.remove(outfolder + r'/{}.solverstate'.format(filename))
    os.rename(files[-1], outfolder + r'/{}.solverstate'.format(filename))    

    #current solve status: (iteration, etc)
    statusFilename = outfolder + r'/{}.caffemodel'.format(filename).replace('.caffemodel', '_status.txt')
    with open(statusFilename, 'w') as f:
        f.write('{}\n'.format(statusDict['iteration']))
        f.write('{}\n'.format(statusDict['loopiteration']))
        f.write('{}\n'.format(statusDict['total_iter']))
        f.write('{}\n'.format(statusDict['epoch']))
        f.write('{}\n'.format(statusDict['loopepoch']))
        f.write('{}\n'.format(statusDict['posInDataset']))
        f.write('{}\n'.format(statusDict['posInUnlabelDataset']))

    np.savetxt(outfolder + r'/trainloss.txt', statusDict['trainlosslist'])
    np.savetxt(outfolder + r'/traintestloss.txt', statusDict['traintestlosslist'])
    np.savetxt(outfolder + r'/testloss.txt', statusDict['testlosslist'])

    for i in range(0, len(statusDict['trainlosslist'])):
        np.savetxt(outfolder + r'/train_{}.txt'.format(i), statusDict['traintestlosslist'][i], delimiter = '\n') 
        np.savetxt(outfolder + r'/traintest_{}.txt'.format(i), statusDict['testlosslist'][i], delimiter = '\n') 

def VisualizeResult(net, params, visualfolder, iteration, names):
    #visualize current batch results
    brdf_result_a = net.blobs['ConvFinal_Albedo'].data
    brdf_result_s = net.blobs['ConvFinal_SpecAlbedo'].data
    brdf_result_r = np.exp(net.blobs['ConvFinal_Roughness'].data)
    brdf_result_n = net.blobs['ConvFinal_Normal'].data

    nCount = brdf_result_a.shape[0]
    for i in range(0, nCount):
        img_a = brdf_result_a[i, :, :, :].transpose((1, 2, 0))
        img_s = brdf_result_s[i, :].flatten() * np.ones((256,256,3))
        img_r = brdf_result_r[i, :].flatten() * np.ones((256,256))
        img_n = brdf_result_n[i, :, :, :].transpose((1, 2, 0))

        mid, lid, vid, oid = names[i]
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_a.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_a))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_s.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_s))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_r.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_r))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_n.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_n))

        img_gt = net.blobs['Data_Image'].data[i, :, :, :].transpose((1, 2, 0))
        img_a_gt = net.blobs['Data_Albedo'].data[i, :, :, :].transpose((1, 2, 0))
        img_s_gt = net.blobs['Data_SpecAlbedo'].data[i, :, 0, 0] * np.ones((256,256,3))
        img_r_gt = np.exp(net.blobs['Data_Roughness'].data[i, 0, 0, 0] * np.ones((256,256)))
        img_n_gt = net.blobs['Data_Normal'].data[i, :, :, :].transpose((1, 2, 0))

        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_img.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_gt))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_a_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_a_gt))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_s_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_s_gt))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_r_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_r_gt))
        cv2.imwrite(visualfolder + r'/trainVis_{}_{}_{}_{}_{}_n_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img_n_gt))


def copySpecWeight(netInput, netRef):
    for ch in [1,2]:
        for k in range(0, 6):
            netInput.params['Conv{}_{}/Conv'.format(k, ch)][0].data[...] = netRef.params['Conv{}_{}/Conv'.format(k, ch)][0].data[...]
            netInput.params['Conv{}_{}/Conv'.format(k, ch)][1].data[...] = netRef.params['Conv{}_{}/Conv'.format(k, ch)][1].data[...]
            
            netInput.params['Conv{}_{}/BN_BeforeScale'.format(k, ch)][0].data[...] = netRef.params['Conv{}_{}/BN_BeforeScale'.format(k, ch)][0].data[...]
            netInput.params['Conv{}_{}/BN_BeforeScale'.format(k, ch)][1].data[...] = netRef.params['Conv{}_{}/BN_BeforeScale'.format(k, ch)][1].data[...]
            netInput.params['Conv{}_{}/BN_BeforeScale'.format(k, ch)][2].data[...] = netRef.params['Conv{}_{}/BN_BeforeScale'.format(k, ch)][2].data[...]
            
            netInput.params['Conv{}_{}/BN'.format(k, ch)][0].data[...] = netRef.params['Conv{}_{}/BN'.format(k, ch)][0].data[...]
            netInput.params['Conv{}_{}/BN'.format(k, ch)][1].data[...] = netRef.params['Conv{}_{}/BN'.format(k, ch)][1].data[...]
        for k in range(0, 3):
            netInput.params['MidConv{}_ch{}/Conv'.format(k, ch)][0].data[...] = netRef.params['MidConv{}_ch{}/Conv'.format(k, ch)][0].data[...]
            netInput.params['MidConv{}_ch{}/Conv'.format(k, ch)][1].data[...] = netRef.params['MidConv{}_ch{}/Conv'.format(k, ch)][1].data[...]
            
            netInput.params['MidConv{}_ch{}/BN_BeforeScale'.format(k, ch)][0].data[...] = netRef.params['MidConv{}_ch{}/BN_BeforeScale'.format(k, ch)][0].data[...]
            netInput.params['MidConv{}_ch{}/BN_BeforeScale'.format(k, ch)][1].data[...] = netRef.params['MidConv{}_ch{}/BN_BeforeScale'.format(k, ch)][1].data[...]
            netInput.params['MidConv{}_ch{}/BN_BeforeScale'.format(k, ch)][2].data[...] = netRef.params['MidConv{}_ch{}/BN_BeforeScale'.format(k, ch)][2].data[...]

            netInput.params['MidConv{}_ch{}/BN'.format(k, ch)][0].data[...] = netRef.params['MidConv{}_ch{}/BN'.format(k, ch)][0].data[...]            
            netInput.params['MidConv{}_ch{}/BN'.format(k, ch)][1].data[...] = netRef.params['MidConv{}_ch{}/BN'.format(k, ch)][1].data[...]  
  

        netInput.params['FCReLU_0_{}/FC'.format(ch)][0].data[...] = netRef.params['FCReLU_0_{}/FC'.format(ch)][0].data[...]
        netInput.params['FCReLU_0_{}/FC'.format(ch)][1].data[...] = netRef.params['FCReLU_0_{}/FC'.format(ch)][1].data[...]   
        netInput.params['FCReLU_1_{}/FC'.format(ch)][0].data[...] = netRef.params['FCReLU_1_{}/FC'.format(ch)][0].data[...]
        netInput.params['FCReLU_1_{}/FC'.format(ch)][1].data[...] = netRef.params['FCReLU_1_{}/FC'.format(ch)][1].data[...]
    
        netInput.params['FC_{}'.format(ch)][0].data[...] = netRef.params['FC_{}'.format(ch)][0].data[...]
        netInput.params['FC_{}'.format(ch)][1].data[...] = netRef.params['FC_{}'.format(ch)][1].data[...]
    return netInput


def testFitting(testnet, pipe_test_recv, testCount, params, visualfolder = [], iteration = -1):
    pixelCnt = 256*256

    testloss = 0
    testloss_a = 0
    testloss_s = 0
    testloss_r = 0
    testloss_n = 0

    randomVis = list(np.random.choice(testCount, 20))

    for i in range(0, testCount):
        img, brdf, names = pipe_test_recv.recv()

        testnet.blobs['Data_Image'].data[...] = img
        testnet.blobs['Data_Albedo'].data[...] = brdf[0, 0:3, :, :]

        if(params['LogSpec']):
           brdf[:,3:6,:,:] = np.log(brdf[:,3:6,:,:])

        testnet.blobs['Data_SpecAlbedo'].data[...] = np.mean(brdf[:,3:6,:,:], axis = (2,3))[:,:,np.newaxis, np.newaxis]
        
        if(params['LogRoughness']):
            brdf[:,6:7,:,:] = np.log(brdf[:,6:7,:,:]) 
        testnet.blobs['Data_Roughness'].data[...] = np.mean(brdf[:, 6, :, :], axis = (1,2))[:, np.newaxis, np.newaxis, np.newaxis]

        if(params['NormalLoss'] == 'ThetaPhi'):
            testnet.blobs['Data_Normal'].data[...] = normalBatchToThetaPhiBatch(brdf[:, 7:10, :, :])
        else:
            testnet.blobs['Data_Normal'].data[...] = brdf[:, 7:10, :, :]

        testnet.forward()
        loss_a = testnet.blobs['MSELoss_Albedo'].data.flatten() / (pixelCnt)
        loss_s = testnet.blobs['MSELoss_SpecAlbedo'].data #/ #(pixelCnt)
        loss_r = testnet.blobs['MSELoss_Roughness'].data #/ #(pixelCnt)
        loss_n = 0
        loss_n = testnet.blobs['MSELoss_Normal'].data.flatten() / (pixelCnt)

        loss = loss_a + loss_s + loss_r + loss_n
        testloss += loss / testCount
        testloss_a += loss_a / testCount
        testloss_s += loss_s / testCount
        testloss_r += loss_r / testCount
        testloss_n += loss_n / testCount

        if((i in randomVis) and visualfolder != []):
            mid, lid, vid, oid = names[0]
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_img.jpg'.format(iteration, mid, lid, vid, oid), toLDR(img[0,:,:,:].transpose((1,2,0))))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_a_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(brdf[0,0:3,:,:].transpose((1,2,0))))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_s_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(brdf[0,3:6,:,:].transpose((1,2,0))))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_r_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(np.exp(brdf[0,6,:,:])))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_n_gt.jpg'.format(iteration, mid, lid, vid, oid), toLDR(brdf[0,7:10,:,:].transpose((1,2,0))))
        
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_a.jpg'.format(iteration, mid, lid, vid, oid), toLDR(testnet.blobs['ConvFinal_Albedo'].data[0,:,:,:].transpose((1,2,0))))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_s.jpg'.format(iteration, mid, lid, vid, oid), toLDR(testnet.blobs['ConvFinal_SpecAlbedo'].data[0,:].flatten() * np.ones((256,256,3))))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_r.jpg'.format(iteration, mid, lid, vid, oid), toLDR(np.exp(testnet.blobs['ConvFinal_Roughness'].data[0,:].flatten()) * np.ones((256,256))))
            cv2.imwrite(visualfolder + r'/testVis_{}_{}_{}_{}_{}_n.jpg'.format(iteration, mid, lid, vid, oid), toLDR(testnet.blobs['ConvFinal_Normal'].data[0,:,:,:].transpose((1,2,0))))


    return testloss, testloss_a, testloss_s, testloss_r, testloss_n


if __name__ == '__main__':

    configFilePath = sys.argv[1]
    outTag = sys.argv[2]
    restoreTraining = int(sys.argv[3])
    gpuid = int(sys.argv[4])
    rendergpuid = int(sys.argv[5])
    asyncQueueSize = 40


    if(restoreTraining == 1):
        outfolder = params_global['outFolder'] + r'/{}'.format(outTag)
    else:
        date = time.strftime(r"%Y%m%d_%H%M%S")
        outfolder = params_global['outFolder'] + r'/{}_{}'.format(outTag, date)

    visualfolder_train = outfolder + r'/intermediate/train'
    visualfolder_test = outfolder + r'/intermediate/test'

    make_dir(outfolder)
    make_dir(visualfolder_train)
    make_dir(visualfolder_test)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(outfolder + '/training_log_text.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    params = loadParams(configFilePath)
    params['rendergpuid'] = rendergpuid

    if(params['NormalLoss'] == 'ThetaPhi'):
        logger.info('ThetaPhi Loss.')

    #init renderer
    OnlineRender = FastRenderEngine(params['rendergpuid'])
    print('Init a render at gpu{}.'.format(params['rendergpuid']))
    OnlineRender.SetGeometry('Plane')
    OnlineRender.SetSampleCount(128, 512)
    if(params['renderLoop']):
        OnlineRender.PreLoadAllLight(r'{}/light.txt'.format(params['envMapFolder']))

    fovRadian = 60.0 / 180.0 * math.pi
    cameraDist = 1.0  / (math.tan(fovRadian / 2.0)) 
    OnlineRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 256, 256)

    random.seed(params['randomSeed'])
    np.random.seed(params['randomSeed'])
    caffe.set_random_seed(params['randomSeed'])

    #init caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)

    #init network
    outchannalDict = {'Albedo':0, 'Spec':1, 'Roughness':2, 'Normal':3, 'Full':4, 'A-S':5}
    if(restoreTraining != 1):
        if(params['NetworkFile'] == ''):
            if(params['NetworkType'] == 'HomogeneousSpec_Share'):
                if(params['NormalLoss'] == 'ThetaPhi'):
                    BRDFNet = SVBRDFNetClass_Share_FC_SRN_ThetaPhi_Sigmoid_A()
                else:
                    BRDFNet = SVBRDFNetClass_Share_FC_SR_Sigmoid_AN()
            elif(params['NetworkType'] == 'HomogeneousSpec'):
                if(params['NormalLoss'] == 'ThetaPhi'):
                    BRDFNet = SVBRDFNetClass_Decompose_FC_SRN_ThetaPhi_Sigmoid_A()
                else:
                    BRDFNet = SVBRDFNetClass_Decompose_FC_SR_Sigmoid_AN()
            elif(params['NetworkType'] == 'HomogeneousSpec_ShareAll'):
                if(params['NormalLoss'] == 'ThetaPhi'):
                    BRDFNet = SVBRDFNetClass_ShareAll_FC_SRN_ThetaPhi_Sigmoid_A()
                else:
                    BRDFNet = SVBRDFNetClass_ShareAll_FC_SR_Sigmoid_AN()
            
            nInputChannal = 6 if (params['LDR'] == 2) else 3
            BRDFNet.createNet(nInputChannal, params['batchSize'], params['BN'], params['nFirstFeatureMap'], [params['albedoWeight'], params['specWeight'], params['roughnessWeight'], params['normalWeight']])
            BRDFNet.saveNet(outfolder)
            networkPath = outfolder + '/net.prototxt'
        else:
            networkPath = params['NetworkFile']

        
        os.system(r'python.exe {}/draw_net.py {}/{} {}/net.png'.format(params_global['scriptRoot'], outfolder, r'net_test.prototxt', outfolder))
        #use Adam Solver
        with open(params_global['scriptRoot'] + r'/solver_template.prototxt', 'r') as f:
            solverDescTmp = f.read()
            solverDesc = solverDescTmp.replace('#snapshotpath#', outfolder + '/')
            solverDesc = solverDesc.replace('#netpath#', networkPath)
            solverDesc = solverDesc.replace('#base_lr#', str(params['lr']))
            solverDesc = solverDesc.replace('#momentum#', str(params['momentum']))
            solverDesc = solverDesc.replace('#gamma#', str(params['lrDecay']))
            solverDesc = solverDesc.replace('#weightDecay#', str(params['weightDecay']))
            solverDesc = solverDesc.replace('\\', '/')

        with open(outfolder + r'/solver_forward.prototxt', 'w') as f:
            f.write(solverDesc)            

    #load network
    solver = caffe.AdamSolver(outfolder + r'/solver_forward.prototxt')
    net = solver.net
    testnet = caffe.Net(outfolder + r'/net_test.prototxt', caffe.TEST)
    if(restoreTraining == 2):
       if(params['PreTrainNet'] != ''):
          net.copy_from(params['PreTrainNet'])
          logger.info('Finetune from {}'.format(params['PreTrainNet']))

    nBRDFChannal = 10

    trainingQueueLength = asyncQueueSize
    testQueueLength = 200    
    
    pipe_train_recv, pipe_train_send = Pipe(False)
    pipe_loop_recv, pipe_loop_send = Pipe(False)
    pipe_test_recv, pipe_test_send = Pipe(False)
          
    loader_train = Process(target = DataLoadProcess_final, args = (pipe_train_send, params['dataset'], params))
    loader_test = Process(target = DataLoadProcess_final, args = (pipe_test_send, params['testDataset'], params, True))
       
    loader_train.daemon = True
    loader_test.daemon = True
 
    loader_train.start()
    loader_test.start()
   
    if(params['unlabelDataset'] != ''):
        loader_loop = Process(target = RealUnlabelDataLoadProcess, args = (pipe_loop_send, params['unlabelDataset'], params))
        loader_loop.daemon = True
        loader_loop.start()
        
    logger.info('Waiting for loading some samples first...\n')
    time.sleep(15)

    loopdatasize = 0
    dataSize = pipe_train_recv.recv()
    testdatasize = pipe_test_recv.recv()
    
    if(params['unlabelDataset'] != ''):
        loopdatasize = pipe_loop_recv.recv()

    totaldatasize = dataSize + loopdatasize
    
    logger.info('datasize = {}, unlabeldatasize = {}'.format(dataSize, loopdatasize))

    if(params['autoLoopRatio'] and params['renderLoop']):
        params['loopBatchLength'] = params['normalBatchLength'] * int(np.round(loopdatasize / dataSize))
        logger.info('Normal Batch Length = {}, Loop Batch Length = {}'.format(params['normalBatchLength'], params['loopBatchLength']))
  
    params['checkPointStepEpochIteration'] = dataSize / params['batchSize'] * params['checkPointStepEpoch']
    if(params['loopStartEpoch'] != -1):
       params['loopStartIteration'] = dataSize / params['batchSize'] * params['loopStartEpoch']
       logger.info('Loop Start iteration = {}'.format(params['loopStartIteration']))

    with open(outfolder + r'/settings.txt', 'wb') as f:
        pickle.dump(params, f)

    iteration = 0
    loopiteration = 0

    total_iter = 0

    epoch = -1
    loopepoch = -1

    posInDataset = 0
    posInUnlabelDataset = 0

    avgLossEveryDisplayStep = 0
    avgLossEveryDisplayStep_a = 0
    avgLossEveryDisplayStep_s = 0
    avgLossEveryDisplayStep_r = 0
    avgLossEveryDisplayStep_n = 0

    avgLoopLossEveryDisplayStep = 0
    avgLoopLossEveryDisplayStep_a = 0
    avgLoopLossEveryDisplayStep_s = 0
    avgLoopLossEveryDisplayStep_r = 0
    avgLoopLossEveryDisplayStep_n = 0

    avgLossEveryCheckPoint = 0
    avgLossEveryCheckPoint_a = 0
    avgLossEveryCheckPoint_s = 0
    avgLossEveryCheckPoint_r = 0
    avgLossEveryCheckPoint_n = 0


    trainfigure_a = plt.figure(1)
    trainfigure_s = plt.figure(2)
    trainfigure_r = plt.figure(3)
    trainfigure_n = plt.figure(4)
    trainfigure_t = plt.figure(5)

    loopfigure_a = plt.figure(6)
    loopfigure_s = plt.figure(7)
    loopfigure_r = plt.figure(8)
    loopfigure_n = plt.figure(9)
    loopfigure_t = plt.figure(10)

    testfigure_a = plt.figure(11)
    testfigure_s = plt.figure(12)
    testfigure_r = plt.figure(13)
    testfigure_n = plt.figure(14)
    testfigure_t = plt.figure(15)

    trainlosslist = [[],[],[],[],[]]
    traintestlosslist = [[],[],[],[],[]]
    testlosslist = [[],[],[],[],[]]
    looplosslist = [[],[],[],[],[]]

    if(restoreTraining == 1):
        files = glob.glob(outfolder + r'/*.solverstate')
        files.sort(key=os.path.getmtime)
        statusFilename = files[-1].replace('.solverstate', '_status.txt')
        solver.restore(files[-1])
        with open(statusFilename, 'r') as f:
            iteration = int(f.readline())
            loopiteration = int(f.readline())
            total_iter = int(f.readline())
            epoch = int(f.readline())
            loopepoch = int(f.readline())
            posInDataset = int(f.readline())
            posInUnlabeledDataset = int(f.readline())

        nLossChannal = 5
        trainlosslist = map(np.ndarray.tolist, np.split(np.loadtxt(outfolder + r'/trainloss.txt').flatten(), nLossChannal))
        traintestlosslist = map(np.ndarray.tolist, np.split(np.loadtxt(outfolder + r'/traintestloss.txt').flatten(), nLossChannal))
        testlosslist = map(np.ndarray.tolist, np.split(np.loadtxt(outfolder + r'/testloss.txt').flatten(), nLossChannal))

    startTime = time.time()
    solveTime = 0
    loadTime = 0
    renderTime = 0

    #load pre-computed auto-exposure data    
    lightPool = []
    lightNormPool = []

    if(params['lightPoolFile'] != ''):
        for m in params['lightPool']:
            for l in range(0, params['lightPool'][m].shape[0]):
                for v in range(0, params['lightPool'][m].shape[1] - 1):
                    rotX = params['lightPool'][m][l,v,0]
                    rotY = params['lightPool'][m][l,v,1]
                    strLightMat = 'r,0,1,0,{}/r,1,0,0,{}/end'.format(rotY, rotX)
                    lightPool.append((params['lightID'][l], (rotX, rotY)))

    if(params['autoExposureLUTFile'] != '' and params['autoExposure']):
        for m in params['autoExposureLUT']:
            for l in range(0, params['autoExposureLUT'][m].shape[0]):
                for v in range(0, params['autoExposureLUT'][m].shape[1] - 1):
                    norm = params['autoExposureLUT'][m][l,v]
                    lightNormPool.append(norm)


    print('Start training...\n')
    while(True):
        #labeled training
        for i in range(0, params['normalBatchLength']):
            if(total_iter == params['nMaxIter'] or epoch == params['nMaxEpoch']):
                break
            #set data (from training set)
            load_start = time.time()
            img_data, brdf_data, names = pipe_train_recv.recv()
            load_end = time.time()

            loadTime += load_end - load_start

            net.blobs['Data_Image'].data[...] = img_data
            net.blobs['Data_Albedo'].data[...] = brdf_data[:,0:3,:,:]
            if(params['LogSpec']):
                brdf_data[:,3:6,:,:] = np.log(brdf_data[:,3:6,:,:])
            net.blobs['Data_SpecAlbedo'].data[...] = brdf_data[:,3:6,0,0][:,:,np.newaxis, np.newaxis]
           
            if(params['LogRoughness']):
                brdf_data[:,6:7,:,:] = np.log(brdf_data[:,6:7,:,:]) 
            net.blobs['Data_Roughness'].data[...] = brdf_data[:,6:7,0,0][:,:,np.newaxis, np.newaxis]
           
            if(params['NormalLoss'] == 'ThetaPhi'):
                net.blobs['Data_Normal'].data[...] = normalBatchToThetaPhiBatch(brdf_data[:,7:10,:,:])
            else:
                net.blobs['Data_Normal'].data[...] = brdf_data[:,7:10,:,:]  
            solve_time_start = time.time()
            solver.step(1)
            solve_time_end = time.time()
            solveTime += (solve_time_end - solve_time_start)

            iterLoss_a = net.blobs['MSELoss_Albedo'].data / (pixelCnt)
            iterLoss_s = net.blobs['MSELoss_SpecAlbedo'].data
            iterLoss_r = net.blobs['MSELoss_Roughness'].data
            iterLoss_n = net.blobs['MSELoss_Normal'].data / (pixelCnt)
            
            iterLoss = iterLoss_a + iterLoss_s + iterLoss_r + iterLoss_n

            avgLossEveryDisplayStep_a += iterLoss_a / params['displayStep']
            avgLossEveryDisplayStep_s += iterLoss_s / params['displayStep']
            avgLossEveryDisplayStep_r += iterLoss_r / params['displayStep']
            avgLossEveryDisplayStep_n += iterLoss_n / params['displayStep']
            avgLossEveryDisplayStep += iterLoss / params['displayStep']
            
            avgLossEveryCheckPoint_a += iterLoss_a / params['checkPointStepIteration']#params['checkPointStepEpochIteration']
            avgLossEveryCheckPoint_s += iterLoss_s / params['checkPointStepIteration']#params['checkPointStepEpochIteration']
            avgLossEveryCheckPoint_r += iterLoss_r / params['checkPointStepIteration']#params['checkPointStepEpochIteration']
            avgLossEveryCheckPoint_n += iterLoss_n / params['checkPointStepIteration']#params['checkPointStepEpochIteration']
            avgLossEveryCheckPoint += iterLoss / params['checkPointStepIteration']#params['checkPointStepEpochIteration']

            namelist = []
            namelist = ['albedo', 'spec', 'roughness', 'total']

            #display training status
            if(total_iter % params['displayStep'] == 0 and total_iter > 0):
                endTime = time.time()
                logger.info('Total Iter:{} / Normal Iter:{} / Unlabel Iter {}'.format(total_iter, iteration, loopiteration))
                logger.info('Normal Epoch: {}, Loop Epoch: {}'.format(epoch, loopepoch))
                logger.info('a-loss = {}, s-loss = {}, r-loss = {}, n-loss = {}, t-loss = {}, Time = {}, Solving Time = {}'.format(avgLossEveryDisplayStep_a, avgLossEveryDisplayStep_s, avgLossEveryDisplayStep_r, avgLossEveryDisplayStep_n, avgLossEveryDisplayStep, endTime - startTime, solveTime))
                logger.info('DataLoad Time: {}, Render Time: {}'.format(loadTime, renderTime))
                VisualizeResult(net, params, visualfolder_train, total_iter, names)
                
                startTime = time.time()	#reset timer
                solveTime = 0
                loadTime = 0
                renderTime = 0
                trainlosslist[0].append(avgLossEveryDisplayStep_a)
                trainlosslist[1].append(avgLossEveryDisplayStep_s)
                trainlosslist[2].append(avgLossEveryDisplayStep_r)
                trainlosslist[3].append(avgLossEveryDisplayStep_n)
                trainlosslist[4].append(avgLossEveryDisplayStep)

                for fid in range(1, 1+len(namelist)):
                    plt.figure(fid)
                    plt.plot(trainlosslist[fid-1], 'rs-', label = 'train')
                    plt.savefig(outfolder + r'/train_{}.png'.format(namelist[fid-1]))
                    
                    plt.figure(128 + fid)
                    plt.gcf().clear()
                    plt.plot(trainlosslist[fid-1][-10::], 'rs-', label = 'train')
                    plt.savefig(outfolder + r'/train_{}_last10.png'.format(namelist[fid-1]))

                avgLossEveryDisplayStep_a = 0
                avgLossEveryDisplayStep_s = 0
                avgLossEveryDisplayStep_r = 0
                avgLossEveryDisplayStep_n = 0
                avgLossEveryDisplayStep = 0          

            #snapshot
            if(total_iter % params['checkPointStepIteration'] == 0 and total_iter > 0):
               statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                          'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist,'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist}
               dumpNetwork(outfolder, solver, 'iter_{}'.format(total_iter), statusDict)
               logger.info('Testing..\n')
               testCount = testdatasize
               testnet.copy_from(outfolder + r'/iter_{}.caffemodel'.format(total_iter))
               testloss, testloss_a, testloss_s, testloss_r, testloss_n = testFitting(testnet, pipe_test_recv, testCount, params, visualfolder_test, total_iter)
               logger.info('Test: a-loss = {}, s-loss = {}, r-loss = {}, n-loss = {}, t-loss = {}'.format(testloss_a, testloss_s, testloss_r, testloss_n, testloss))
               traintestlosslist[0].append(avgLossEveryCheckPoint_a)
               traintestlosslist[1].append(avgLossEveryCheckPoint_s)
               traintestlosslist[2].append(avgLossEveryCheckPoint_r)
               traintestlosslist[3].append(avgLossEveryCheckPoint_n)
               traintestlosslist[4].append(avgLossEveryCheckPoint)

               testlosslist[0].append(testloss_a)
               testlosslist[1].append(testloss_s)
               testlosslist[2].append(testloss_r)
               testlosslist[3].append(testloss_n)
               testlosslist[4].append(testloss)

               for fid in range(11, 11+len(namelist)):
                   plt.figure(fid)
                   plt.plot(traintestlosslist[fid-11], 'rs-', label = 'train')
                   plt.plot(testlosslist[fid-11], 'bs-', label = 'test')
                   plt.savefig(outfolder + r'/test_{}.png'.format(namelist[fid-11]))

                   plt.figure(64 + fid)
                   plt.gcf().clear()
                   plt.plot(traintestlosslist[fid-11][-5::], 'rs-', label = 'train')
                   plt.plot(testlosslist[fid-11][-5::], 'bs-', label = 'test')                
                   plt.savefig(outfolder + r'/test_last5_{}.png'.format(namelist[fid-11]))
                                  
               avgLossEveryCheckPoint_a = 0
               avgLossEveryCheckPoint_s = 0
               avgLossEveryCheckPoint_r = 0
               avgLossEveryCheckPoint_n = 0
               avgLossEveryCheckPoint = 0

            #snapshot
            if(iteration % params['checkPointStepEpochIteration'] == 0 and total_iter > 0 and iteration > 0):
               statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                          'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist}
               dumpNetwork(outfolder, solver, 'epoch_{}'.format(epoch), statusDict)
               logger.info('Testing..\n')
               testCount = testdatasize#min(5000, testdatasize)
               testnet.copy_from(outfolder + r'/epoch_{}.caffemodel'.format(epoch))
               testloss, testloss_a, testloss_s, testloss_r, testloss_n = testFitting(testnet, pipe_test_recv, testCount, params, visualfolder_test, total_iter)
               logger.info('Test: a-loss = {}, s-loss = {}, r-loss = {}, n-loss = {}, t-loss = {}'.format(testloss_a, testloss_s, testloss_r, testloss_n, testloss))


            iteration = iteration + 1
            total_iter = total_iter + 1
            posInDataset = (posInDataset + params['batchSize']) % dataSize
            epoch = iteration * params['batchSize'] / dataSize
               
        #self-augment training
        if(params['renderLoop'] == True and iteration >= params['loopStartIteration']):
            img_w = 256
            img_h = 256
            for k in range(0, params['loopBatchLength']):
                if(total_iter == params['nMaxIter'] or epoch == params['nMaxEpoch']):
                    break
                predict_brdf = np.zeros((params['batchSize'], nBRDFChannal, img_h, img_w))
                
                #forward unlabeled batch, getting predicted BRDF
                img_data = pipe_loop_recv.recv()
                net.blobs['Data_Image'].data[...] = img_data
                net.forward()
                predict_brdf[:,0:3,:,:] = np.copy(net.blobs['ConvFinal_Albedo'].data[...])
                if(params['LogSpec']):
                   sData = np.exp(np.copy(net.blobs['ConvFinal_SpecAlbedo'].data[...]))
                else:
                   sData = np.copy(net.blobs['ConvFinal_SpecAlbedo'].data[...])
 
                predict_brdf[:,3:6,:,:] = sData[:,:,np.newaxis,np.newaxis] * np.ones((params['batchSize'], 1, 256, 256))

                if(params['LogRoughness']):
                   rData = np.exp(np.copy(net.blobs['ConvFinal_Roughness'].data[...]))
                else:
                   rData = np.copy(net.blobs['ConvFinal_Roughness'].data[...])
                predict_brdf[:,6:7,:,:] = rData[:,:,np.newaxis,np.newaxis] * np.ones((params['batchSize'], 1, 256, 256))

                if(params['NormalLoss'] == 'ThetaPhi'):
                   predict_brdf[:,7:10,:,:] = thetaPhiBatchToNormalBatch(np.copy(net.blobs['ConvFinal_Normal'].data[...]))
                else:
                   predict_brdf[:,7:10,:,:] = np.copy(net.blobs['ConvFinal_Normal'].data[...])
                                          
                predict_brdf[:,0:3, :, :] = np.minimum(1.0, np.maximum(predict_brdf[:,0:3, :, :], 0.001))
                predict_brdf[:,3:6, :, :] = np.minimum(1.0, np.maximum(predict_brdf[:,3:6, :, :], 0.001))
                predict_brdf[:,6, :, :] = np.minimum(1.0, np.maximum(predict_brdf[:,6, :], 0.001))
                predict_brdf[:,7:10, :, :] = renormalize(predict_brdf[:,7:10, :, :])
                                                  
                #random select light from light Pool
                xforms = [[],[]]
                renderIds = []
                lNorms = []
                if(params['lightPoolFile'] != ''):
                    selectedIds = np.random.choice(len(lightPool), params['batchSize'], False)
                    renderIds = [lightPool[i][0] for i in selectedIds]
                    for i in selectedIds:
                        xforms[0].append(lightPool[i][1][0])
                        xforms[1].append(lightPool[i][1][1])
                    if(params['autoExposureLUTFile'] != '' and params['autoExposure']):
                        lNorms = [lightNormPool[i] for i in selectedIds]

                #render predicted SVBRDF
                render_start = time.time()
                img_predict = renderOnlineEnvlight(predict_brdf, OnlineRender, params, renderIds, xforms, lNorms)
                render_end = time.time()
                renderTime += render_end - render_start
                net.blobs['Data_Image'].data[...] = img_predict
                net.blobs['Data_Albedo'].data[...] = predict_brdf[:,0:3,:,:]

                if(params['LogSpec']):
                    predict_brdf[:,3:6,:,:] = np.log(predict_brdf[:,3:6,:,:]) 
                net.blobs['Data_SpecAlbedo'].data[...] = predict_brdf[:,3:6,0,0][:,:,np.newaxis, np.newaxis]
        
                if(params['LogRoughness']):
                    predict_brdf[:,6:7,:,:] = np.log(predict_brdf[:,6:7,:,:]) 
                net.blobs['Data_Roughness'].data[...] = predict_brdf[:,6:7,0,0][:,:,np.newaxis, np.newaxis]

                if(params['NormalLoss'] == 'ThetaPhi'):
                    net.blobs['Data_Normal'].data[...] = normalBatchToThetaPhiBatch(predict_brdf[:,7:10,:,:])
                else:
                    net.blobs['Data_Normal'].data[...] = predict_brdf[:,7:10,:,:]
                
                #update network with rendered data            
                solve_time_start = time.time()
                solver.step(1)
                solve_time_end = time.time()
                solveTime += (solve_time_end - solve_time_start)
                iterLoss_a = net.blobs['MSELoss_Albedo'].data / (pixelCnt)
                iterLoss_s = net.blobs['MSELoss_SpecAlbedo'].data
                iterLoss_r = net.blobs['MSELoss_Roughness'].data

                iterLoss_n = net.blobs['MSELoss_Normal'].data / (pixelCnt)
                iterLoss = iterLoss_a + iterLoss_s + iterLoss_r + iterLoss_n

                avgLossEveryDisplayStep_a += iterLoss_a / params['displayStep']
                avgLossEveryDisplayStep_s += iterLoss_s / params['displayStep']
                avgLossEveryDisplayStep_r += iterLoss_r / params['displayStep']
                avgLossEveryDisplayStep_n += iterLoss_n / params['displayStep']
                avgLossEveryDisplayStep += iterLoss / params['displayStep']

                #display training status
                if(total_iter % params['displayStep'] == 0 and iteration > 0):
                    endTime = time.time()
                    logger.info('Total Iter:{} / Normal Iter:{} / Unlabel Iter {}'.format(total_iter, iteration, loopiteration))
                    logger.info('Normal Epoch: {}, Loop Epoch: {}'.format(epoch, loopepoch))
                    logger.info('a-loss = {}, s-loss = {}, r-loss = {}, n-loss = {}, t-loss = {}, Time = {}, Solving Time = {}'.format(avgLossEveryDisplayStep_a, avgLossEveryDisplayStep_s, avgLossEveryDisplayStep_r, avgLossEveryDisplayStep_n, avgLossEveryDisplayStep, endTime - startTime, solveTime))
                    logger.info('DataLoad Time: {}, Render Time: {}'.format(loadTime, renderTime))

                    startTime = time.time()	#reset timer
                    solveTime = 0
                    loadTime = 0
                    renderTime
                    trainlosslist[0].append(avgLossEveryDisplayStep_a)
                    trainlosslist[1].append(avgLossEveryDisplayStep_s)
                    trainlosslist[2].append(avgLossEveryDisplayStep_r)
                    trainlosslist[3].append(avgLossEveryDisplayStep_n)
                    trainlosslist[4].append(avgLossEveryDisplayStep)

                    for fid in range(1, 1+len(namelist)):
                        plt.figure(fid)
                        plt.plot(trainlosslist[fid-1], 'rs-', label = 'train')
                        plt.savefig(outfolder + r'/train_{}.png'.format(namelist[fid-1]))
                    
                        plt.figure(128 + fid)
                        plt.gcf().clear()
                        plt.plot(trainlosslist[fid-1][-10::], 'rs-', label = 'train')
                        plt.savefig(outfolder + r'/train_{}_last10.png'.format(namelist[fid-1]))

                    avgLossEveryDisplayStep_a = 0
                    avgLossEveryDisplayStep_s = 0
                    avgLossEveryDisplayStep_r = 0
                    avgLossEveryDisplayStep_n = 0
                    avgLossEveryDisplayStep = 0

                #snapshot
                if(total_iter % params['checkPointStepIteration'] == 0 and total_iter > 0):
                    statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                          'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist}
                    dumpNetwork(outfolder, solver, 'iter_{}'.format(total_iter), statusDict)
                    logger.info('Testing..\n')
                    testCount = testdatasize
                    testnet.copy_from(outfolder + r'/iter_{}.caffemodel'.format(total_iter))
                    testloss, testloss_a, testloss_s, testloss_r, testloss_n = testFitting(testnet, pipe_test_recv, testCount, params, visualfolder_test, total_iter)
                    logger.info('Test: a-loss = {}, s-loss = {}, r-loss = {}, n-loss = {}, t-loss = {}'.format(testloss_a, testloss_s, testloss_r, testloss_n, testloss))
                    
                    traintestlosslist[0].append(avgLossEveryCheckPoint_a)
                    traintestlosslist[1].append(avgLossEveryCheckPoint_s)
                    traintestlosslist[2].append(avgLossEveryCheckPoint_r)
                    traintestlosslist[3].append(avgLossEveryCheckPoint_n)
                    traintestlosslist[4].append(avgLossEveryCheckPoint)

                    testlosslist[0].append(testloss_a)
                    testlosslist[1].append(testloss_s)
                    testlosslist[2].append(testloss_r)
                    testlosslist[3].append(testloss_n)
                    testlosslist[4].append(testloss)

                    
                    for fid in range(11, 11+len(namelist)):
                        plt.figure(fid)
                        plt.plot(traintestlosslist[fid-11], 'rs-', label = 'train')
                        plt.plot(testlosslist[fid-11], 'bs-', label = 'test')
                        plt.savefig(outfolder + r'/test_{}.png'.format(namelist[fid-11]))
                    
                        plt.figure(64 + fid)
                        plt.gcf().clear()
                        plt.plot(traintestlosslist[fid-11][-5::], 'rs-', label = 'train')
                        plt.plot(testlosslist[fid-11][-5::], 'bs-', label = 'test')                
                        plt.savefig(outfolder + r'/test_last5_{}.png'.format(namelist[fid-11]))
                                   
                    avgLossEveryCheckPoint_a = 0
                    avgLossEveryCheckPoint_s = 0
                    avgLossEveryCheckPoint_r = 0
                    avgLossEveryCheckPoint_n = 0
                    avgLossEveryCheckPoint = 0

                loopiteration = loopiteration + 1
                total_iter = total_iter + 1
                posInUnlabelDataset = (posInUnlabelDataset + params['batchSize']) % loopdatasize
                loopepoch = loopiteration * params['batchSize'] / loopdatasize
                if(params['loopRestartFrequency'] != -1 and (loopepoch % params['loopRestartFrequency'] == 0 and loopepoch > 0)):
                   #reset solver
                   logger.info('Resetting Learning Rate...')
                   solver.snapshot()
                   files = glob.glob(outfolder + r'/*.caffemodel')
                   files.sort(key=os.path.getmtime)
                   modelfile = files[-1]
                   files_state = glob.glob(outfolder + r'/*.solverstate')
                   files_state.sort(key=os.path.getmtime)
                   statefile = files_state[-1]

                   with open(params_global['scriptRoot'] + r'/solver_template.prototxt', 'r') as f:
                        solverDescTmp = f.read()
                        solverDesc = solverDescTmp.replace('#snapshotpath#', outfolder + '/')
                        solverDesc = solverDesc.replace('#netpath#', networkPath)
                        solverDesc = solverDesc.replace('#base_lr#', '0.0001')#str(params['lr']))
                        solverDesc = solverDesc.replace('#momentum#', str(params['momentum']))
                        solverDesc = solverDesc.replace('#gamma#', str(params['lrDecay']))
                        solverDesc = solverDesc.replace('#weightDecay#', str(params['weightDecay']))
                        solverDesc = solverDesc.replace('\\', '/')

                   with open(outfolder + r'/solver_forward_restart.prototxt', 'w') as f:
                        f.write(solverDesc)           

                   if(params['SolverType'] == 'Adam'):
                      solver = caffe.AdamSolver(outfolder + r'/solver_forward_restart.prototxt')
                   elif(params['SolverType'] == 'SGD'):
                      solver = caffe.SGDSolver(outfolder + r'/solver_forward_restart.prototxt')
                   net = solver.net
                   net.copy_from(modelfile)

                   os.remove(modelfile)
                   os.remove(statefile)

        if(total_iter == params['nMaxIter'] or epoch == params['nMaxEpoch']):
           #write final net
           statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                        'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist}
           dumpNetwork(outfolder, solver, 'final', statusDict)
           break             