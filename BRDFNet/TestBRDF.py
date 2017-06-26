# TestBRDF.py
# Test Script for BRDF-Net

import random, os, time, sys
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)

sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')
import caffe
from utils import save_pfm, load_pfm, pfmFromBuffer, pfmToBuffer, DataLoaderSimple
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import json
import glob
import math
import shutil
import cv2
import jinja2
import pickle

from skimage.measure import structural_similarity as ssim
from ConfigParser import ConfigParser

from FastRendererCUDA import FastRenderEngine

os.chdir(working_path)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
params = {}

matplotlib.rcParams.update({'font.size': 14})


def draw2DHeatMap(filename, img, label_x, label_y, maxvalue = []):
    
    figure = plt.figure(1)
    figure.clear()
    if(maxvalue == []):
        maxValue = np.max(img)
    else:
        maxValue = maxvalue

    f = plt.imshow(img, cmap='hot', interpolation='nearest', vmin=0.0, vmax=maxValue)
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    plt.savefig(filename)

def draw2DCurve(filename, xlist, ylist):
    figure = plt.figure(1)
    figure.clear()
    plt.plot(xlist, ylist, 'rs-')
    plt.savefig(filename)

def toLDR(img):
    img_out = img ** (1.0 / 2.2)
    img_out = img_out * 255
    return img_out

def mean(a, axis=None):
    if a.mask is np.ma.nomask:
        return super(np.ma.MaskedArray, a).mean(axis=axis)

    sums = a.sum(axis=axis)
    counts = np.logical_not(a.mask).sum(axis=axis)
    result = sums * 1. / counts
    return result

def getLightTransList(numX, numY, seed):
    lightList = []
    np.random.seed(seed)
    for i in range(0, numX):
        for j in range(0, numY):
            angleY = np.random.uniform(0.0, 360.0)
            angleX = np.random.uniform(-30.0, 0.0)
            lightList.append('r,0,1,0,{}/r,1,0,0,{}/end'.format(angleY, angleX))
    return lightList


with open('folderPath.txt', 'r') as f:
    params['geometryPath'] = r'../Render/sphere.obj'
    params['scriptRoot'] = r'../Utils'
    params['outFolder'] = f.readline().strip()
    params['envMapFolder'] = f.readline().strip()

thetaList = np.linspace(10.0 * math.pi / 180.0, 80.0 * math.pi / 180.0, 15)
phiList = np.linspace(30.0 * math.pi / 180.0, 330.0 * math.pi / 180.0, 15)

with open(params['envMapFolder'] + r'\light.txt', 'r') as f:
     lightID = map(int, f.read().strip().split('\n'))
     lightID = list(np.array(lightID) - 1)

def centeredMean(values):
    return (sum(values) - max(values) - min(values)) / (len(values) - 2)


def testSingleBRDF(net, brdf, tid, pid, out_img = False, fixedChannal = [0]*3):
    px = np.sin(thetaList[tid]) * np.cos(phiList[pid])
    py = np.sin(thetaList[tid]) * np.sin(phiList[pid])
    pz = np.cos(thetaList[tid])
    OnlineRender.SetPointLight(0, px, py, pz, 0, 1, 1, 1)
    OnlineRender.SetAlbedoValue([brdf[0], brdf[0], brdf[0]])
    OnlineRender.SetSpecValue([brdf[1], brdf[1], brdf[1]])
    OnlineRender.SetRoughnessValue(brdf[2])

    img_gt = OnlineRender.Render()[:,:,0] 
    brdf_inNet = np.copy(brdf)
    brdf_inNet = np.exp(brdf_inNet[2])
    brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict = testImage(net, img_gt[np.newaxis, :, :], brdf_inNet.reshape((1,3,1,1)), tid, pid, out_img, fixedChannal)
    return img_gt, brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict

def testImage(net, img, brdf_inNet, tid, pid, out_img = False, fixedChannal = [0]*3):
    if(test_params['Ratio']):
        brdf_in = np.zeros((1,2,1,1))
        brdf_in[0,0,0,0] = np.log(brdf_inNet[0,0,0,0] / brdf_inNet[0,1,0,0])
        brdf_in[0,1,0,0] = brdf_inNet[0,2,0,0]
    else:
        brdf_in = brdf_inNet

    net.blobs['Data_Image'].data[...] = img
    net.blobs['Data_BRDF'].data[...] = brdf_in
    net.forward()
        

    if(net.blobs.has_key('Out_Ratio')):
        brdf_ratio_predict = np.exp(net.blobs['Out_Ratio'].data.flatten()[0])
        brdf_roughness_predict = np.exp(net.blobs['Out_Roughness_Fix'].data.flatten()[0])
        brdf_predict = np.array([brdf_ratio_predict * brdf_inNet[0,1,0,0], brdf_inNet[0,1,0,0], brdf_roughness_predict])
    else:
        brdf_predict = net.blobs['Out_LossFeature'].data.flatten()
        brdf_predict[2] = np.exp(brdf_predict[2])
        brdf_predict = np.maximum(0, brdf_predict)

    for cid, ch in enumerate(fixedChannal):
        if(ch == 1):
            brdf_predict[cid] = brdf.flatten()[cid]

    if(net.blobs.has_key('Out_Ratio')):
        loss_brdf = [net.blobs['RatioLoss'].data.flatten()[0],
                     (0.5*(brdf_predict[0] - brdf_inNet[0,0,0,0]))**2,
                     net.blobs['RoughnessLoss'].data.flatten()[0],
                     net.blobs['RatioLoss'].data.flatten()[0] + net.blobs['RoughnessLoss'].data.flatten()[0]]
    else:
        loss_brdf = [0.5 * net.blobs['DiffuseLoss'].data.flatten()[0], 
                    0.5 * net.blobs['SpecLoss'].data.flatten()[0], 
                    0.5 * net.blobs['RoughnessLoss'].data.flatten()[0], 
                    0.5 * net.blobs['MSELoss'].data.flatten()[0]]

    if(test_params['envLighting']):
        OnlineRender.SetEnvLightByID(lightID[tid]+1)
        OnlineRender.SetAlbedoValue([brdf_predict[0], brdf_predict[0], brdf_predict[0]])
        OnlineRender.SetSpecValue([brdf_predict[1], brdf_predict[1], brdf_predict[1]])
        OnlineRender.SetRoughnessValue(brdf_predict[2])   
        img_predict = OnlineRender.Render()[:,:,0]
        
        save_pfm('test_p.pfm', img_predict)
        save_pfm('test_gt.pfm', img[0,:,:])

        loss_mse = 0.5 * np.mean((img[0,:,:] - img_predict)**2)
        loss_ssim = 0.5 * (1.0 - ssim(img[0,:,:], img_predict, win_size = 5))
    else:
        px = np.sin(thetaList[tid]) * np.cos(phiList[pid])
        py = np.sin(thetaList[tid]) * np.sin(phiList[pid])
        pz = np.cos(thetaList[tid])

        OnlineRender.SetPointLight(0, px, py, pz, 0, 1, 1, 1)
        OnlineRender.SetAlbedoValue([brdf_predict[0], brdf_predict[0], brdf_predict[0]])
        OnlineRender.SetSpecValue([brdf_predict[1], brdf_predict[1], brdf_predict[1]])
        OnlineRender.SetRoughnessValue(brdf_predict[2])   
        img_predict = OnlineRender.Render()[:,:,0]

        loss_mse = np.mean((img[0,:,:] - img_predict)**2)
        loss_ssim = 1.0 - ssim(img[0,:,:], img_predict, win_size = 5)
    
    if(out_img):
        return brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict
    else:
        return brdf_predict, loss_brdf, loss_mse, loss_ssim, -1

def loadParams(filepath):
    config = ConfigParser()
    config.read(filepath)

    params = {}
    #dataset
    params['testSet'] = config.get('dataset', 'testSet')
    params['albedoRange'] = map(int, config.get('dataset','albedoRange').split(','))
    params['specRange'] = map(int, config.get('dataset','specRange').split(','))
    params['roughnessRange'] = map(int, config.get('dataset','roughnessRange').split(','))
    #sample
    params['albedoCnt'] = config.getint('sample', 'albedoCnt')
    params['specCnt'] = config.getint('sample', 'specCnt')
    params['roughnessCnt'] = config.getint('sample', 'roughnessCnt')
    params['resample'] = config.getboolean('sample', 'resample')
    #light
    params['envLighting'] = config.getboolean('light','envLighting')
    if(params['envLighting']):
        params['lRange'] = range(0, 49)
        params['vRange'] = [0]#range(0, 6)
    else:
        params['thetaRange'] = map(int, config.get('light','thetaRange').split(','))
        params['phiRange'] = map(int, config.get('light','phiRange').split(','))
    #visualList
    list_d = [] if(config.get('visualList','diffuseVisualList') == '') else map(int, config.get('visualList','diffuseVisualList').split(','))
    list_s = [] if(config.get('visualList','specVisualList') == '') else map(int, config.get('visualList','specVisualList').split(','))
    list_r = [] if(config.get('visualList','roughnessVisualList') == '') else map(int, config.get('visualList','roughnessVisualList').split(','))

    lRange0 = params['lRange'] if params['envLighting'] else params['thetaRange']
    lRange1 = params['vRange'] if params['envLighting'] else params['phiRange']
    
    params['visualTestList'] = {}
    for a in list_d:
        for s in list_s:
            for r in list_r:
                for t in lRange0:
                    for p in lRange1:
                        params['visualTestList'].append((a, s, r, t, p))
    #network
    params['outchannals'] = config.get('network', 'outchannals')
    params['Ratio'] = config.getboolean('network', 'Ratio')
    #outtag
    params['outtag'] = config.get('output', 'outtag')

    return params


if __name__ == '__main__':

    renderContext = {}
    jinjiaEnv = jinja2.Environment(loader = jinja2.FileSystemLoader('./')).get_template('template.html')

    savedNet = sys.argv[1]
    testParamPath = sys.argv[2]
    gpuid = int(sys.argv[3])

    resultfolder, modelfile = os.path.split(savedNet)
    #Load test params
    test_params = {}
    test_params = loadParams(testParamPath)
    outputFolder = resultfolder + r'/test_{}'.format(test_params['outtag'])
    if(os.path.exists(outputFolder) == False):
        os.makedirs(outputFolder)   

    OnlineRender = FastRenderEngine(0)
    OnlineRender.SetGeometry('Sphere')#params['geometryPath'], True, '')
    OnlineRender.PreLoadAllLight(r'{}/light.txt'.format(params['envMapFolder']))

    fovRadian = 60.0 / 180.0 * math.pi
    cameraDist = 1.5  / (math.tan(fovRadian / 2.0))
    OnlineRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 128, 128)
    OnlineRender.SetSampleCount(128, 512)

    renderContext['experimentTag'] = savedNet

    random.seed(23333)
    np.random.seed(23333)
    caffe.set_random_seed(23333)

    caffe.set_mode_gpu()
    caffe.set_device(gpuid)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(outputFolder + '/test_log_text.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    testnet = caffe.Net(resultfolder + r'/net_test.prototxt', caffe.TEST)
    logger.info('Loading saved model: {}'.format(savedNet))
    testnet.copy_from(savedNet)
    
    path, file = os.path.split(test_params['testSet'])

    lRange0 = test_params['lRange'] if test_params['envLighting'] else test_params['thetaRange']
    lRange1 = test_params['vRange'] if test_params['envLighting'] else test_params['phiRange']

    testSet = DataLoaderSimple(path, file, 9, 9, 14, 128, 128)
    if(test_params['resample']):
        testAlbedo = np.linspace(testSet.brdfCube[0,0,0,0], testSet.brdfCube[-1,0,0,0], test_params['albedoCnt'])
        testSpec = np.linspace(testSet.brdfCube[0,0,0,1], testSet.brdfCube[0,-1,0,1], test_params['specCnt'])
        testRoughness = np.exp(np.linspace(math.log(testSet.brdfCube[0,0,0,2]), math.log(testSet.brdfCube[0,0,-1,2]), test_params['roughnessCnt']))
    else:
        testAlbedo = testSet.brdfCube[:,0,0,0]#np.linspace(0.06, 0.95, 10)
        testSpec = testSet.brdfCube[0,:,0,1]#np.linspace(0.06, 0.95, 10)
        testRoughness = np.exp(testSet.brdfCube[0,0,:,2])#np.exp(np.linspace(math.log(0.025), math.log(0.95), 15))

        testSet.buildSubDataset(test_params['albedoRange'], test_params['specRange'], test_params['roughnessRange'], lRange0, lRange1)
        testSet.shuffle(23333)

    logger.info('Test Dataset:{}'.format(file))
    nameToIDTable = {'albedo':0, 'spec':1, 'roughness':2, 'total':3}

    traintestloss = map(np.ndarray.tolist, np.split(np.loadtxt(resultfolder + r'\traintestloss.txt').flatten(), 4))
    testlossFull = map(np.ndarray.tolist, np.split(np.loadtxt(resultfolder + r'\testlossfull.txt').flatten(), 4))

    if(test_params['outchannals'] == 'Ratio'):
        out_channals = ['albedo', 'spec', 'roughness', 'total']
    else:
        out_channals = [test_params['outchannals'].lower()] if test_params['outchannals'] != 'Full' else ['albedo', 'spec', 'roughness', 'total']


    renderContext['trainingCurveData'] = {}
    for out_channal in out_channals:
        channalID = nameToIDTable[out_channal]
        np.savetxt(outputFolder + r'/trainLoss_{}.txt'.format(out_channal), traintestloss[channalID], delimiter = '\n') 
        np.savetxt(outputFolder + r'/testLossFull_{}.txt'.format(out_channal), testlossFull[channalID], delimiter = '\n')

        txtTrain = ''
        txtTest = ''
        txtTestFull = ''
        for vid in range(0, len(traintestloss[channalID])):
            txtTrain = txtTrain + '{} '.format(traintestloss[channalID][vid])
            txtTest = txtTest + '{} '.format(testlossFull[channalID][vid])
            txtTestFull = txtTestFull + '{} '.format(testlossFull[channalID][vid])

            renderContext['trainingCurveData'][out_channal] = dict(train = txtTrain, testFull = txtTestFull)

    logger.info('Testing on test dataset...')
    errorCubeChannel = np.ma.zeros((len(testAlbedo), len(testSpec), len(testRoughness), 6))       
    errorCubeChannel.mask = True

    fixChannal = [1, 1, 1]
    for out_channal in out_channals:
        channalID = nameToIDTable[out_channal]
        if(channalID < 3):
            fixChannal[channalID] = 0

    visualResultList = []
    visualFolder = outputFolder + r'/visualCompare'
    if(os.path.exists(visualFolder) == False):
        os.makedirs(visualFolder)

    print(lRange0)
    print(lRange1)

    #Draw Ratio/Roughness Curve
    logratioAxis = np.linspace(math.log(0.1/0.95), math.log(0.95/0.1), 14)
    logroughnessAxis = np.linspace(math.log(0.03), math.log(0.87), 14)

    loss_ratio_roughness = np.zeros((14, 14))

    for aid in test_params['albedoRange']:
        for sid in test_params['specRange']:
            for rid in test_params['roughnessRange']:
                loss_v = []
                for tid in lRange0:
                    for pid in lRange1:
                        img, brdf = testSet.GetItemByID(aid, sid, rid, tid, pid, False)#test_params['envLighting'])
                        brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict = testImage(testnet, img, brdf, tid, pid, True, fixChannal)
                        l_ratio = np.log(brdf[0,0,0,0] / brdf[0,1,0,0])
                        l_roughness = brdf[0,2,0,0]
                        loss_v.append(loss_mse)
                        ratio_id = np.abs(l_ratio - logratioAxis).argmin()
                        
                loss_ratio_roughness[ratio_id, rid] = np.mean(loss_v)

    save_pfm(outputFolder + r'/test_slice_ratio_R_visual_all.pfm', loss_ratio_roughness)
    draw2DHeatMap(outputFolder + r'/test_slice_ratio_R_visual_all.png', loss_ratio_roughness, 'Roughness', 'Ratio', np.max(loss_ratio_roughness))

    for aid in test_params['albedoRange']:
        for sid in test_params['specRange']:
            for rid in test_params['roughnessRange']:
                loss_list = [[],[],[],[],[],[]]
                for tid in lRange0:
                    for pid in lRange1:
                        if(test_params['resample']):
                            brdf_gt = np.array([testAlbedo[aid], testSpec[sid], testRoughness[rid]])
                            img_gt, brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict = testSingleBRDF(testnet, brdf_gt, tid, pid, True, fixChannal)
                        else:
                            img, brdf = testSet.GetItemByID(aid, sid, rid, tid, pid, False)#test_params['envLighting'])

                            brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict = testImage(testnet, img, brdf, tid, pid, True, fixChannal)
                        for ch in range(0, 4):
                            loss_list[ch].append(loss_brdf[ch])
                        loss_list[4].append(loss_mse)
                        loss_list[5].append(loss_ssim)
                errorCubeChannel.mask[aid, sid, rid, :] = False  
                loss_mean = [np.mean(values) for values in loss_list]
                errorCubeChannel[aid, sid, rid, :] = loss_mean                                         


    #Visual Compare
    for brdfTuple in test_params['visualTestList']:
        aid, sid, rid, tid, pid = brdfTuple
        if(test_params['resample']):
            brdf = np.array([testAlbedo[aid], testSpec[sid], testRoughness[rid]])
            img, brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict = testSingleBRDF(testnet, brdf_gt, tid, pid, True, fixChannal)
        else:
            img, brdf = testSet.GetItemByID(aid, sid, rid, tid, pid)
            brdf_predict, loss_brdf, loss_mse, loss_ssim, img_predict = testImage(testnet, img, brdf, tid, pid, True, fixChannal)
            img = img[0,:,:]
            brdf = brdf.flatten()

        save_pfm(visualFolder + r'/{}_{}_{}_{}_{}_predict.pfm'.format(aid, sid, rid, tid, pid), img_predict)
        save_pfm(visualFolder + r'/{}_{}_{}_{}_{}_gt.pfm'.format(aid, sid, rid, tid, pid), img)

        cv2.imwrite(visualFolder + r'/{}_{}_{}_{}_{}_predict.png'.format(aid, sid, rid, tid, pid), toLDR(img_predict))
        cv2.imwrite(visualFolder + r'/{}_{}_{}_{}_{}_gt.png'.format(aid, sid, rid, tid, pid), toLDR(img))

        visualResultList.append(dict(predictedImgFile = r'./visualCompare/{}_{}_{}_{}_{}_predict.png'.format(aid, sid, rid, tid, pid), 
                            gtImgFile = r'./visualCompare/{}_{}_{}_{}_{}_gt.png'.format(aid, sid, rid, tid, pid),
                            predictedBRDF = '{:04f}/{:04f}/{:04f}'.format(brdf_predict[0], brdf_predict[1], brdf_predict[2]),
                            gtBRDF = '{:04f}/{:04f}/{:04f}'.format(brdf[0], brdf[1], brdf[2])))

    renderContext['width'] = 192
    if(visualResultList != []):
        numFigureInEachRow = 5
        numRows = (len(visualResultList)) / numFigureInEachRow
        renderContext['renderResult'] = map(list, np.split(np.array(visualResultList), numRows))

    avgLossChannel = mean(errorCubeChannel, axis = (0, 1, 2))

    for out_channal in out_channals:
        channal_id = nameToIDTable[out_channal]
        np.savetxt(outputFolder + r'/errorCube{}_test.txt'.format(out_channal), errorCubeChannel[:,:,:,channal_id].flatten())
        logger.info('Test: {}-loss = {}'.format(out_channal, avgLossChannel[channal_id]))

        renderContext['{}loss'.format(out_channal)] = '{:08f}'.format(avgLossChannel[channal_id])
    logger.info('Test: visual-loss = {}'.format(avgLossChannel[4]))
    logger.info('Test: SSIM-loss = {}'.format(avgLossChannel[5]))

    renderContext['visualloss'] = '{:08f}'.format(avgLossChannel[4])
    renderContext['ssimloss'] = '{:08f}'.format(avgLossChannel[5])
                   
    #2: draw slice of error cube.

    draw_s_r = True
    draw_a_s = True
    draw_a_r = True
    
    if(len(test_params['albedoRange']) == 1):
        draw_a_s = False
        draw_a_r = False

    if(len(test_params['specRange']) == 1):
        draw_s_r = False
        draw_a_s = False

    if(len(test_params['roughnessRange']) == 1):
        draw_a_r = False
        draw_s_r = False
    
    #S-R slices
    if(draw_s_r):
        slice_S_R = []
        for out_channal in out_channals:
            channal_id = nameToIDTable[out_channal]
            meanSlice_S_R = np.ma.mean(errorCubeChannel[:,:,:,channal_id], axis = 0)
            maxValue = np.max(meanSlice_S_R)
            save_pfm(outputFolder + r'/test_slice_S_R_{}_all.pfm'.format(out_channal), np.ma.filled(meanSlice_S_R, 0))
            draw2DHeatMap(outputFolder + r'/test_slice_S_R_{}_all.png'.format(out_channal), meanSlice_S_R, 'Roughness', 'Spec', maxValue)

        meanSlice_S_R = np.ma.mean(errorCubeChannel[:,:,:,4], axis = 0)
        maxValue = np.max(meanSlice_S_R)
        save_pfm(outputFolder + r'/test_slice_S_R_visual_all.pfm', np.ma.filled(meanSlice_S_R, 0))
        draw2DHeatMap(outputFolder + r'/test_slice_S_R_visual_all.png', meanSlice_S_R, 'Roughness', 'Spec', maxValue)
        meanSlice_S_R = np.ma.mean(errorCubeChannel[:,:,:,5], axis = 0)
        maxValue = np.max(meanSlice_S_R)
        save_pfm(outputFolder + r'/test_slice_S_R_SSIM_all.pfm', np.ma.filled(meanSlice_S_R, 0))
        draw2DHeatMap(outputFolder + r'/test_slice_S_R_SSIM_all.png', meanSlice_S_R, 'Roughness', 'Spec', maxValue)

        for albedoPointID in test_params['albedoRange']:
            slice = {}
            slice['value'] = '{:04f}'.format(testAlbedo[albedoPointID])
            for out_channal in out_channals:
                channal_id = nameToIDTable[out_channal]
                slice_S_R_channal = errorCubeChannel[albedoPointID,:,:,channal_id]
                save_pfm(outputFolder + r'/test_slice_S_R_{}_{}.pfm'.format(out_channal, albedoPointID), np.ma.filled(slice_S_R_channal, 0))
                draw2DHeatMap(outputFolder + r'/test_slice_S_R_{}_{}.png'.format(out_channal, albedoPointID), slice_S_R_channal, 'Roughness', 'Spec', np.max(errorCubeChannel[:,:,:,channal_id]))
                slice[out_channal] = r'./test_slice_S_R_{}_{}.png'.format(out_channal, albedoPointID)

            save_pfm(outputFolder + r'/test_slice_S_R_visual_{}.pfm'.format(albedoPointID), np.ma.filled(errorCubeChannel[albedoPointID,:,:,4], 0))
            draw2DHeatMap(outputFolder + r'/test_slice_S_R_visual_{}.png'.format(albedoPointID), errorCubeChannel[albedoPointID,:,:,4], 'Roughness', 'Spec', np.max(errorCubeChannel[:,:,:,4]))
            slice['visual'] = r'./test_slice_S_R_visual_{}.png'.format(albedoPointID)


            slice_S_R.append(slice)

        renderContext['slice_S_R'] = slice_S_R

        #curve
    if(len(test_params['albedoRange']) != 1):
        renderContext['axisAlbedo'] = {}
        for out_channal in out_channals:
            channal_id = nameToIDTable[out_channal]
            avgLossAlbedoAxis_ch = mean(errorCubeChannel[:,:,:,channal_id], axis = (1, 2)).flatten()
            draw2DCurve(outputFolder + r'/test_albedocurve_{}.png'.format(out_channal), testAlbedo, avgLossAlbedoAxis_ch)
            renderContext['axisAlbedo'][out_channal] = r'./test_albedocurve_{}.png'.format(out_channal)

        avgLossAlbedoAxis_visual = mean(errorCubeChannel[:,:,:,4], axis = (1, 2)).flatten()
        draw2DCurve(outputFolder + r'/test_albedocurve_visual.png', testAlbedo, avgLossAlbedoAxis_visual)
        renderContext['axisAlbedo']['visual'] = r'./test_albedocurve_visual.png'


    #A-R slices
    if(draw_a_r):
        slice_A_R = []

        for out_channal in out_channals:
            channal_id = nameToIDTable[out_channal]
            meanSlice_A_R = np.ma.mean(errorCubeChannel[:,:,:,channal_id], axis = 1)
            maxValue = np.max(meanSlice_A_R)
            save_pfm(outputFolder + r'/test_slice_A_R_{}_all.pfm'.format(out_channal), np.ma.filled(meanSlice_A_R, 0))
            draw2DHeatMap(outputFolder + r'/test_slice_A_R_{}_all.png'.format(out_channal), meanSlice_A_R, 'Roughness', 'Albedo', maxValue)

        meanSlice_A_R = np.ma.mean(errorCubeChannel[:,:,:,4], axis = 1)
        maxValue = np.max(meanSlice_A_R)
        save_pfm(outputFolder + r'/test_slice_A_R_visual_all.pfm', np.ma.filled(meanSlice_A_R, 0))
        draw2DHeatMap(outputFolder + r'/test_slice_A_R_visual_all.png', meanSlice_A_R, 'Roughness', 'Albedo', maxValue)
        meanSlice_A_R = np.ma.mean(errorCubeChannel[:,:,:,5], axis = 1)
        maxValue = np.max(meanSlice_A_R)
        save_pfm(outputFolder + r'/test_slice_A_R_SSIM_all.pfm', np.ma.filled(meanSlice_A_R, 0))
        draw2DHeatMap(outputFolder + r'/test_slice_A_R_SSIM_all.png', meanSlice_A_R, 'Roughness', 'Albedo', maxValue)

        for specPointID in test_params['specRange']:
            slice = {}
            slice['value'] = '{:04f}'.format(testSpec[specPointID])
            for out_channal in out_channals:
                channal_id = nameToIDTable[out_channal]
                slice_A_R_channal = errorCubeChannel[:,specPointID,:,channal_id]
                save_pfm(outputFolder + r'/test_slice_A_R_{}_{}.pfm'.format(out_channal, specPointID), np.ma.filled(slice_A_R_channal, 0))
                draw2DHeatMap(outputFolder + r'/test_slice_A_R_{}_{}.png'.format(out_channal, specPointID), slice_A_R_channal, 'Roughness', 'Albedo', np.max(errorCubeChannel[:,:,:,channal_id]))
                slice[out_channal] = r'./test_slice_A_R_{}_{}.png'.format(out_channal, specPointID)

            save_pfm(outputFolder + r'/test_slice_A_R_visual_{}.pfm'.format(specPointID), np.ma.filled(errorCubeChannel[:,specPointID,:,4], 0))
            draw2DHeatMap(outputFolder + r'/test_slice_A_R_visual_{}.png'.format(specPointID), errorCubeChannel[:,specPointID,:,4], 'Roughness', 'Albedo', np.max(errorCubeChannel[:,:,:,4]))
            slice['visual'] = r'./test_slice_A_R_visual_{}.png'.format(specPointID)
            slice_A_R.append(slice)

        renderContext['slice_A_R'] = slice_A_R

    if(len(test_params['specRange']) != 1):  
        #curve
        renderContext['axisSpec'] = {}
        for out_channal in out_channals:
            channal_id = nameToIDTable[out_channal]
            avgLossSpecAxis_ch = mean(errorCubeChannel[:,:,:,channal_id], axis = (0, 2)).flatten()
            draw2DCurve(outputFolder + r'/test_speccurve_{}.png'.format(out_channal), testSpec, avgLossSpecAxis_ch)
            renderContext['axisSpec'][out_channal] = r'./test_speccurve_{}.png'.format(out_channal)

        avgLossSpecAxis_visual = mean(errorCubeChannel[:,:,:,4], axis = (0, 2)).flatten()
        draw2DCurve(outputFolder + r'/test_speccurve_visual.png', testSpec, avgLossSpecAxis_visual)
        renderContext['axisSpec']['visual'] = r'./test_speccurve_visual.png'

    #A-S slices
    if(draw_a_s):
        slice_A_S = []

        for out_channal in out_channals:
            channal_id = nameToIDTable[out_channal]
            meanSlice_A_S = np.ma.mean(errorCubeChannel[:,:,:,channal_id], axis = 2)
            maxValue = np.max(meanSlice_A_S)
            save_pfm(outputFolder + r'/test_slice_A_S_{}_all.pfm'.format(out_channal), np.ma.filled(meanSlice_A_S, 0))
            draw2DHeatMap(outputFolder + r'/test_slice_A_S_{}_all.png'.format(out_channal), meanSlice_A_S, 'Spec', 'Albedo', maxValue)

        meanSlice_A_S = np.ma.mean(errorCubeChannel[:,:,:,4], axis = 2)
        maxValue = np.max(meanSlice_A_S)
        save_pfm(outputFolder + r'/test_slice_A_S_visual_all.pfm', np.ma.filled(meanSlice_A_S, 0))
        draw2DHeatMap(outputFolder + r'/test_slice_A_S_visual_all.png', meanSlice_A_S, 'Spec', 'Albedo', maxValue)
        meanSlice_A_S = np.ma.mean(errorCubeChannel[:,:,:,5], axis = 2)
        maxValue = np.max(meanSlice_A_S)
        save_pfm(outputFolder + r'/test_slice_A_S_SSIM_all.pfm', np.ma.filled(meanSlice_A_S, 0))
        draw2DHeatMap(outputFolder + r'/test_slice_A_S_SSIM_all.png', meanSlice_A_S, 'Spec', 'Albedo', maxValue)

        for roughnessPointID in test_params['roughnessRange']:
            slice = {}
            slice['value'] = '{:04f}'.format(testRoughness[roughnessPointID])
            for out_channal in out_channals:
                channal_id = nameToIDTable[out_channal]
                slice_A_S_channal = errorCubeChannel[:,:,roughnessPointID,channal_id]
                save_pfm(outputFolder + r'/test_slice_A_S_{}_{}.pfm'.format(out_channal, roughnessPointID), np.ma.filled(slice_A_S_channal, 0))
                draw2DHeatMap(outputFolder + r'/test_slice_A_S_{}_{}.png'.format(out_channal, roughnessPointID), slice_A_S_channal, 'Spec', 'Albedo', np.max(errorCubeChannel[:,:,:,channal_id]))
                slice[out_channal] = r'./test_slice_A_S_{}_{}.png'.format(out_channal, roughnessPointID)  
                       
            save_pfm(outputFolder + r'/test_slice_A_S_visual_{}.pfm'.format(roughnessPointID), np.ma.filled(errorCubeChannel[:,:,roughnessPointID,4], 0))
            draw2DHeatMap(outputFolder + r'/test_slice_A_S_visual_{}.png'.format(roughnessPointID), errorCubeChannel[:,:,roughnessPointID,4], 'Spec', 'Albedo', np.max(errorCubeChannel[:,:,:,4]))
            slice['visual'] = r'./test_slice_A_S_visual_{}.png'.format(roughnessPointID)                        
            slice_A_S.append(slice)

        renderContext['slice_A_S'] = slice_A_S

    if(len(test_params['roughnessRange']) != 1):
        #curve
        renderContext['axisRoughness'] = {}
        for out_channal in out_channals:
            channal_id = nameToIDTable[out_channal]
            avgLossRoughnessAxis_ch = mean(errorCubeChannel[:,:,:,channal_id], axis = (0, 1)).flatten()
            draw2DCurve(outputFolder + r'/test_roughnesscurve_{}.png'.format(out_channal), testRoughness, avgLossRoughnessAxis_ch)
            renderContext['axisRoughness'][out_channal] = r'./test_roughnesscurve_{}.png'.format(out_channal)      

        avgLossRoughnessAxis_visual = mean(errorCubeChannel[:,:,:,4], axis = (0, 1)).flatten()
        draw2DCurve(outputFolder + r'/test_roughnesscurve_visual.png', testRoughness, avgLossRoughnessAxis_visual)
        renderContext['axisRoughness']['visual'] = r'./test_roughnesscurve_visual.png'    

    folder1 = outputFolder + r'/results'
    if(os.path.exists(folder1) == False):
        os.makedirs(folder1)

    with open(folder1 + r'/loss.txt', 'w') as f:
        f.write('log ratio:{}\n'.format(avgLossChannel[0]))
        f.write('albedo:{}\n'.format(avgLossChannel[1]))
        f.write('log roughness:{}\n'.format(avgLossChannel[2]))
        f.write('total:{}\n'.format(avgLossChannel[3]))
        f.write('render mse:{}\n'.format(avgLossChannel[4]))
        f.write('render dssim:{}\n'.format(avgLossChannel[5]))


    shutil.copy(outputFolder + r'/test_slice_A_R_roughness_all.pfm', folder1 + r'/test_slice_A_R_roughness_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_R_roughness_all.png', folder1 + r'/test_slice_A_R_roughness_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_R_spec_all.pfm', folder1 + r'/test_slice_A_R_albedo_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_R_spec_all.png', folder1 + r'/test_slice_A_R_albedo_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_R_albedo_all.pfm', folder1 + r'/test_slice_A_R_ratio_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_R_albedo_all.png', folder1 + r'/test_slice_A_R_ratio_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_R_total_all.pfm', folder1 + r'/test_slice_A_R_total_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_R_total_all.png', folder1 + r'/test_slice_A_R_total_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_R_visual_all.pfm', folder1 + r'/test_slice_A_R_visual_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_R_visual_all.png', folder1 + r'/test_slice_A_R_visual_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_R_SSIM_all.pfm', folder1 + r'/test_slice_A_R_SSIM_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_R_SSIM_all.png', folder1 + r'/test_slice_A_R_SSIM_all.png')
                                 
    shutil.copy(outputFolder + r'/test_slice_S_R_roughness_all.pfm', folder1 + r'/test_slice_S_R_roughness_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_S_R_roughness_all.png', folder1 + r'/test_slice_S_R_roughness_all.png')
    shutil.copy(outputFolder + r'/test_slice_S_R_spec_all.pfm', folder1 + r'/test_slice_S_R_albedo_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_S_R_spec_all.png', folder1 + r'/test_slice_S_R_albedo_all.png')
    shutil.copy(outputFolder + r'/test_slice_S_R_albedo_all.pfm', folder1 + r'/test_slice_S_R_ratio_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_S_R_albedo_all.png', folder1 + r'/test_slice_S_R_ratio_all.png')
    shutil.copy(outputFolder + r'/test_slice_S_R_total_all.pfm', folder1 + r'/test_slice_S_R_total_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_S_R_total_all.png', folder1 + r'/test_slice_S_R_total_all.png')
    shutil.copy(outputFolder + r'/test_slice_S_R_visual_all.pfm', folder1 + r'/test_slice_S_R_visual_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_S_R_visual_all.png', folder1 + r'/test_slice_S_R_visual_all.png')
    shutil.copy(outputFolder + r'/test_slice_S_R_SSIM_all.pfm', folder1 + r'/test_slice_S_R_SSIM_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_S_R_SSIM_all.png', folder1 + r'/test_slice_S_R_SSIM_all.png')

    shutil.copy(outputFolder + r'/test_slice_A_S_roughness_all.pfm', folder1 + r'/test_slice_A_S_roughness_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_S_roughness_all.png', folder1 + r'/test_slice_A_S_roughness_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_S_spec_all.pfm', folder1 + r'/test_slice_A_S_albedo_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_S_spec_all.png', folder1 + r'/test_slice_A_S_albedo_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_S_albedo_all.pfm', folder1 + r'/test_slice_A_S_ratio_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_S_albedo_all.png', folder1 + r'/test_slice_A_S_ratio_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_S_total_all.pfm', folder1 + r'/test_slice_A_S_total_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_S_total_all.png', folder1 + r'/test_slice_A_S_total_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_S_visual_all.pfm', folder1 + r'/test_slice_A_S_visual_all.pfm')
    shutil.copy(outputFolder + r'/test_slice_A_S_visual_all.png', folder1 + r'/test_slice_A_S_visual_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_S_SSIM_all.png', folder1 + r'/test_slice_A_S_SSIM_all.png')
    shutil.copy(outputFolder + r'/test_slice_A_S_SSIM_all.pfm', folder1 + r'/test_slice_A_S_SSIM_all.pfm')
    renderedHtml = jinjiaEnv.render(renderContext)
    print('{}'.format(outputFolder + r'/result.html'))
    with open(outputFolder + r'/result.html', 'w') as f:
        f.write(renderedHtml)


