# BRDFNetTraining.py
# Training script for BRDF-Net and SA-BRDF-Net

import random, os, sys, time, json, pickle, glob, logging, math, shutil, itertools
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)
sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')

from ConfigParser import ConfigParser, SafeConfigParser
from multiprocessing import Process
from multiprocessing import Queue as MultiQueue

import caffe
import numpy as np
import cv2
import matplotlib.pyplot as plt


from NetClass import BRDFNetClassLogLoss_Single_SplitChannal_New_Ratio
from utils import save_pfm, load_pfm, pfmFromBuffer, pfmToBuffer, toHDR, toLDR, findIndex, listToStr, DataLoaderSimple

from FastRendererCUDA import FastRenderEngine

os.chdir(working_path)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

params_global = {}

alllightID = []

#loading global folder pathes
with open('folderPath.txt', 'r') as f:
    params_global['geometryPath'] = r'../Render/sphere.obj'
    params_global['scriptRoot'] = r'../Utils'
    params_global['outFolder'] = f.readline().strip()
    params_global['envMapFolder'] = f.readline().strip()
      
#load lighting ids      
with open(params_global['envMapFolder'] + r'/light.txt', 'r') as f:
     alllightID = map(int, f.read().strip().split('\n'))
     alllightID = list(np.array(alllightID) - 1)
                
def randomLight(num, thetaList, phiList):
    outLight = np.zeros((num, 3))
    thetalist = np.random.choice(thetaList, num)
    philist = np.random.choice(phiList, num)

    outLight[:, 0] = np.sin(thetalist) * np.cos(philist)
    outLight[:, 1] = np.sin(thetalist) * np.sin(philist)
    outLight[:, 2] = np.cos(thetalist)
    
    return outLight

def renderOnline_gray(brdfBatch, lightBatch = []):
    imgBatch = np.zeros((brdfBatch.shape[0], 1, 128, 128))
    if(lightBatch == []):
        lightBatch = randomLight(brdfBatch.shape[0])
    for i in range(0, brdfBatch.shape[0]):
        OnlineRender.SetAlbedoValue([brdfBatch[i, 0], brdfBatch[i, 0], brdfBatch[i, 0]])
        OnlineRender.SetSpecValue([brdfBatch[i, 1], brdfBatch[i, 1], brdfBatch[i, 1]])
        OnlineRender.SetRoughnessValue(brdfBatch[i, 2])
        OnlineRender.SetPointLight(0, lightBatch[i, 0], lightBatch[i, 1], lightBatch[i, 2], 0, 1, 1, 1)
        
        imgBatch[i, 0, :, :] = OnlineRender.Render()[:, :, 0]   
    return imgBatch

#for a given BRDF batch, render image batch by random assign a lighting to each BRDF
def renderOnlineEnvlight(brdfBatch, lightIDs = [], lightXforms = [], gray = False):
    lightTime = 0
    if(gray):
        imgBatch = np.zeros((brdfBatch.shape[0], 1, 128, 128))
    else:
        imgBatch = np.zeros((brdfBatch.shape[0], 3, 128, 128))
    if(lightXforms == []):
        angle_y = np.random.uniform(0.0, 360.0, brdfBatch.shape[0])
        angle_x = np.random.uniform(-30.0, 10.0, brdfBatch.shape[0])
    else:
        angle_x = lightXforms[0]
        angle_y = lightXforms[1]

    for i in range(0, brdfBatch.shape[0]):
        OnlineRender.SetEnvLightByID(lightIDs[i] + 1, angle_x[i], angle_y[i])
        OnlineRender.SetAlbedoValue([brdfBatch[i, 0], brdfBatch[i, 0], brdfBatch[i, 0]])
        OnlineRender.SetSpecValue([brdfBatch[i, 1], brdfBatch[i, 1], brdfBatch[i, 1]])
        OnlineRender.SetRoughnessValue(brdfBatch[i, 2])
        if(gray):
            imgBatch[i, :, :, :] = OnlineRender.Render()[:,:,0]
        else:
            imgBatch[i, :, :, :] = OnlineRender.Render().transpose((2,0,1))
    return imgBatch

#for a given BRDF batch, render image batch by random assign a lighting to each BRDF
def renderOnlineEnvlightFromPool(brdfBatch, lightPool, gray = False):
    if(gray):
        imgBatch = np.zeros((brdfBatch.shape[0], 1, 128, 128))
    else:
        imgBatch = np.zeros((brdfBatch.shape[0], 3, 128, 128))

    lightConditions = random.sample(lightPool, brdfBatch.shape[0])

    for i in range(0, brdfBatch.shape[0]):
        lightID, angle_x, angle_y = lightConditions[i]
        OnlineRender.SetEnvLightByID(params['lightID'][lightID] + 1, angle_x, angle_y)
        OnlineRender.SetAlbedoValue([brdfBatch[i, 0], brdfBatch[i, 0], brdfBatch[i, 0]])
        OnlineRender.SetSpecValue([brdfBatch[i, 1], brdfBatch[i, 1], brdfBatch[i, 1]])
        OnlineRender.SetRoughnessValue(brdfBatch[i, 2])
        if(gray):
            imgBatch[i, :, :, :] = OnlineRender.Render()[:,:,0]
        else:
            imgBatch[i, :, :, :] = OnlineRender.Render().transpose((2,0,1))
    return imgBatch


def DataLoadProcess(queue, datasetfile, params, unlabel = 0):
    fullBrdfList = []
    for a in range(0, 10):
        for s in range(0, 10):
            for r in range(0, 15):
                fullBrdfList.append([a,s,r])
    

    path, file = os.path.split(datasetfile)
    batchSize = params['batchSize']
    dataset = DataLoaderSimple(path, file, 10, 10, 15, 128, 128)
    if(params['envLighting']):
        if(unlabel):
            unlabelSample = []
            if(params['manualUnlabelSet'] != ''):                
                with open(params['manualUnlabelSet'], 'r') as f:
                    tList = f.read().strip().split('\n')
                for t in tList:
                    unlabelSample.append(map(int, t.split(',')))
            else:
               for b in fullBrdfList:
                   if b not in params['BRDFSample']:
                       unlabelSample.append(b)

            if(params['unlabellightcondition'] == 0 and params['unlabellightfile'] != ''):
               brdf_lightsample = []
               lvlist = np.loadtxt(params['unlabellightfile']).reshape((10*10*15, -1, 2))
               for brdf_sample in unlabelSample:
                   a, s, r = brdf_sample
                   id = a*150 + s*15 + r
                   for k in range(0, lvlist.shape[1]):
                       brdf_lightsample.append([a,s,r,int(lvlist[id,k,0]), int(lvlist[id,k,1])])
               dataset.buildSubDataset_2(brdf_lightsample, False)
            else:
               dataset.buildSubDataset_1(unlabelSample, params['lRange'], params['vRange'], False)
        else:
            brdf_sample = params['BRDFSample']
            dataset.buildSubDataset_1(brdf_sample, params['lRange'], params['vRange'], False)
    else:
        dataset.buildSubDataset_1(params['BRDFSample'], params['thetaRange'], params['phiRange'], unlabel)   

    albedoList = dataset.brdfCube[:,0,0,0]
    specList = dataset.brdfCube[0,:,0,1]
    roughnessList = dataset.brdfCube[0,0,:,2]
    dataset.shuffle(params['randomSeed'])
    queue.put(dataset.dataSize)
    queue.put(dataset.brdfCube.shape)
    queue.put((1, 0, 1, 0, 1, 0))

    albedoList = dataset.brdfCube[:,0,0,0]
    specList = dataset.brdfCube[0,:,0,1]
    roughnessList = dataset.brdfCube[0,0,:,2]
    queue.put((albedoList, specList, roughnessList))
    counter = 0
    posInDataSet = 0
    epoch = 0  

    while(True):
        imgbatch, brdfbatch, name = dataset.GetBatchWithName(posInDataSet, batchSize, params['color'])#dataset.GetBatch(posInDataSet, batchSize)
        if(params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor'):
            brdfbatch_ratio = np.zeros((batchSize, 2, 1, 1))
            brdfbatch_ratio[:,0,:,:] = np.log(brdfbatch[:,0,:,:] / brdfbatch[:,1,:,:])
            brdfbatch_ratio[:,1,:,:] = brdfbatch[:,2,:,:]
            queue.put((imgbatch, brdfbatch_ratio, name))
        else:
            queue.put((imgbatch, brdfbatch, name))
        counter = counter + batchSize
        posInDataSet = (posInDataSet + batchSize) % dataset.dataSize
        newepoch = counter / dataset.dataSize
        if(newepoch != epoch):
            dataset.shuffle()
        epoch = newepoch



def loadParams(filepath):
    config = SafeConfigParser({'unlabelalbedoRange':'0,1,2,3,4,5,6,7,8,9',
                               'unlabelspecRange':'0,1,2,3,4,5,6,7,8,9',
                               'unlabelroughnessRange':'0,1,2,3,4,5,6,7,8,9,10,11,12,13,14',
                               'sampleFile':'',
                               'color':'0', 
                               'PreTrainNet':'',
                               'loopRestartFrequency':'-1', 
                               'manualUnlabelSet':'', 
                               'datalightcondition':'1', 
                               'unlabellightcondition':'1', 
                               'unlabellightfile':''})
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

    #stopping crteria
    params['nMaxEpoch'] = config.getint('stopping', 'nMaxEpoch')
    params['nMaxIter'] = config.getint('stopping', 'nMaxIter')

    #loop setting
    params['renderLoop'] = config.getboolean('loop', 'renderLoop')          #set to 0 to disable self-augmentation
    params['autoLoopRatio'] = config.getboolean('loop', 'autoLoopRatio')    
    #the SA training would alternating between 'normalBatchLength' iteration of normal training and 'loopBatchLength' of self-augment training
    params['normalBatchLength'] = config.getint('loop', 'normalBatchLength')     
    params['loopStartEpoch'] = config.getint('loop', 'loopStartEpoch')      
    params['loopStartIteration'] = config.getint('loop', 'loopStartIteration') #add loop after this number of normal training.
    params['loopBatchLength'] = config.getint('loop', 'loopBatchLength') #how many mini-batch iteration for ever loop optimize
    params['loopRestartFrequency'] = config.getint('loop', 'loopRestartFrequency')
    #network structure
    params['NetworkType'] = config.get('network', 'NetworkType')
    params['Channal'] = config.get('network', 'Channal')
    params['BN'] = config.getboolean('network', 'BN')
    params['color'] = config.getboolean('network', 'color')
    params['PreTrainNet'] = config.get('network', 'PreTrainNet')
    #dataset
    params['NormalizeInput'] = config.getboolean('dataset', 'NormalizeInput')
    params['dataset'] = config.get('dataset', 'dataset')
    params['unlabelDataset'] = config.get('dataset', 'unlabelDataset')
    params['testDataset'] = config.get('dataset', 'testDataset')
    params['envLighting'] = config.getboolean('dataset', 'envLighting')
    params['lightID'] = alllightID
    params['manualUnlabelSet'] = config.get('dataset', 'manualUnlabelSet')

    params['datalightcondition'] = config.getint('dataset', 'datalightcondition')    #full:1 one:0
    params['unlabellightcondition'] = config.getint('dataset', 'unlabellightcondition')    #full:1 one:0
    params['unlabellightfile'] = config.get('dataset', 'unlabellightfile')
    
    params['albedoRange'] = map(int, config.get('dataset','albedoRange').split(','))    
    if(params['albedoRange'][0] == -1): 
        params['albedoRange'] = []
    params['specRange'] = map(int, config.get('dataset','specRange').split(','))
    if(params['specRange'][0] == -1): 
        params['specRange'] = []
    params['roughnessRange'] = map(int, config.get('dataset','roughnessRange').split(','))
    if(params['roughnessRange'][0] == -1): 
        params['albedoRange'] = []

    params['BRDFSample'] = map(list, list(itertools.product(params['albedoRange'], params['specRange'], params['roughnessRange'])))
    params['AdditionSample'] = []
    params['sampleFile'] = config.get('dataset', 'sampleFile')
    if(os.path.exists(params['sampleFile'])):
        with open(params['sampleFile'], 'r') as f:
            allData = f.read().strip().split('\n')
            for b in allData:
                params['AdditionSample'].append(map(int, b.split(',')))

    params['BRDFSample'] = params['BRDFSample'] + params['AdditionSample']

    
    if(params['envLighting']):
        params['lRange'] = range(0, 49)
        params['vRange'] = range(0, 9)
    else:
        params['thetaRange'] = map(int, config.get('dataset','thetaRange').split(','))
        params['phiRange'] = map(int, config.get('dataset','phiRange').split(','))
        
    params['testalbedoRange'] = map(int, config.get('dataset','testalbedoRange').split(','))
    params['testspecRange'] = map(int, config.get('dataset','testspecRange').split(','))
    params['testroughnessRange'] = map(int, config.get('dataset','testroughnessRange').split(','))
     
    #display and testing
    params['displayStep'] = config.getint('display', 'displayStep')
    params['loopdisplayStep'] = config.getint('display', 'loopdisplayStep')
    params['checkPointStepIteration'] = config.getint('display', 'checkPointStepIteration')
    params['checkPointStepEpoch'] = config.getint('display', 'checkPointStepEpoch')

    return params



def buildAutoTestScript(param_train, path):
    if(len(path) <= 3):
        filename = 'homogeneous_vis.ini'
    else:
        filename = path + r'/homogeneous_vis.ini'
    config = ConfigParser()

    config.add_section('dataset')
    config.set('dataset', 'testSet', params['testDataset'])
    config.set('dataset', 'albedoRange', ','.join(map(str,params['testalbedoRange'])))
    config.set('dataset', 'specRange', ','.join(map(str,params['testspecRange'])))
    config.set('dataset', 'roughnessRange', ','.join(map(str,params['testroughnessRange'])))

    config.add_section('sample')
    config.set('sample', 'albedoCnt', 10)
    config.set('sample', 'specCnt', 10)
    config.set('sample', 'roughnessCnt', 15)
    config.set('sample', 'resample', 0)

    config.add_section('light')
    config.set('light', 'envLighting', int(params['envLighting']))
    if(params['envLighting'] == False):
        config.set('light', 'thetaRange', '0,2,4,6,8,10')
        config.set('light', 'phiRange', '0,4,8,11,14')

    config.add_section('visualList')
    config.set('visualList', 'diffuseVisualList', '')
    config.set('visualList', 'specVisualList', '')
    config.set('visualList', 'roughnessVisualList', '')

    config.add_section('network')
    config.set('network', 'outchannals', params['Channal'])
    config.set('network', 'Ratio', int(params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor'))

    config.add_section('output')
    config.set('output', 'outtag', 'test')

    with open(filename, 'w') as f:
        config.write(f)

    return filename
        

def testFitting(testnet, testDataset, testCount, ratio = False):
    testloss = 0
    testloss_a = 0
    testloss_s = 0
    testloss_r = 0

    for i in range(0, testCount):
        img, brdf = testDataset.GetBatch(i, 1, params['color'])
        testnet.blobs['Data_Image'].data[...] = img
        if(ratio):
            brdf_ratio = np.zeros((1,2,1,1))
            brdf_ratio[:,0,:,:] = np.log(brdf[0,0,0,0] / brdf[0,1,0,0])
            brdf_ratio[:,1,:,:] = brdf[0,2,0,0]
            testnet.blobs['Data_BRDF'].data[...] = brdf_ratio
        else:
            testnet.blobs['Data_BRDF'].data[...] = brdf
        testnet.forward()
   
        if(ratio):
            testloss_a += testnet.blobs['RatioLoss'].data / testCount
            testloss_r += testnet.blobs['RoughnessLoss'].data / testCount
            testloss_s = 0
            testloss += (testloss_a + testloss_r) / testCount
        else:
            testloss += testnet.blobs['MSELoss'].data / testCount
            testloss_a += testnet.blobs['DiffuseLoss'].data / testCount
            testloss_s += testnet.blobs['SpecLoss'].data / testCount
            testloss_r += testnet.blobs['RoughnessLoss'].data / testCount
    
    return testloss, testloss_a, testloss_s, testloss_r

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
    np.savetxt(outfolder + r'/looploss.txt', statusDict['looplosslist'])
    np.savetxt(outfolder + r'/traintestloss.txt', statusDict['traintestlosslist'])
    np.savetxt(outfolder + r'/testloss.txt', statusDict['testlosslist'])
    np.savetxt(outfolder + r'/testlossfull.txt', statusDict['testlossFulllist'])
    np.savetxt(outfolder + r'/cubesample_{}.txt'.format(statusDict['epoch']), statusDict['cubeSample'].flatten())
    np.savetxt(outfolder + r'/brdfCubeSample_{}.txt'.format(statusDict['epoch']), statusDict['brdfCubeSample'].flatten())
    np.savetxt(outfolder + r'/brdfCubeSample.txt'.format(statusDict['epoch']), statusDict['brdfCubeSample'].flatten())


if __name__ == '__main__':
    #read params
    configFilePath = sys.argv[1]
    outTag = sys.argv[2]
    restoreTraining = int(sys.argv[3])
    gpuid = int(sys.argv[4])
    renderid = int(sys.argv[5])
    autoTest = int(sys.argv[6])

    if(restoreTraining != 1):
        date = time.strftime(r"%Y%m%d_%H%M%S")
        outfolder = params_global['outFolder'] + r'/{}_{}'.format(outTag, date)
    else:
        outfolder = params_global['outFolder'] + r'/{}'.format(outTag)

    if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)

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

    logger.info('outfolder = {}'.format(outfolder))

    print(configFilePath)

    params = loadParams(configFilePath)

    if(autoTest):
        paramDir, tmp = os.path.split(configFilePath)
        testparampath = buildAutoTestScript(params, paramDir)
        logger.info('Test Param path: {}'.format(testparampath))

    #init renderer    
    OnlineRender = FastRenderEngine(renderid)
    OnlineRender.SetGeometry('Sphere')
    fovRadian = 60.0 / 180.0 * math.pi
    cameraDist = 1.5  / (math.tan(fovRadian / 2.0))
    OnlineRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 128, 128)
    OnlineRender.SetSampleCount(128, 512)
    if(params['envLighting']):
        OnlineRender.PreLoadAllLight(r'{}/light.txt'.format(params_global['envMapFolder']))
    
    #init caffe
    random.seed(params['randomSeed'])
    np.random.seed(params['randomSeed'])
    logger.info('Setting Seed...')
    caffe.set_random_seed(params['randomSeed'])

    caffe.set_mode_gpu()
    logger.info('Setting GPU...')
    caffe.set_device(gpuid)
    logger.info('Done.')

    logger.info('Loading network and solver settings...')
    #init network
    if(restoreTraining != 1):
        BRDFNet = BRDFNetClassLogLoss_Single_SplitChannal_New_Ratio()

        if(params['Channal'] == 'Albedo'):
            BRDFNet.createNet(params['batchSize'], 0, params['BN'], params['NormalizeInput'])
        elif(params['Channal'] == 'Spec'):
            BRDFNet.createNet(params['batchSize'], 1, params['BN'], params['NormalizeInput'])
        elif(params['Channal'] == 'Roughness'):
            BRDFNet.createNet(params['batchSize'], 2, params['BN'], params['NormalizeInput'])
        elif(params['Channal'] == 'Ratio'):
            BRDFNet.createNet(params['batchSize'], 4, params['BN'], params['NormalizeInput'])
        elif(params['Channal'] == 'Full'):
            BRDFNet.createNet(params['batchSize'], 3, params['BN'], params['NormalizeInput'])

        BRDFNet.saveNet(outfolder)

        #use Adam Solver
        with open(params_global['scriptRoot'] + r'/solver_template.prototxt', 'r') as f:
            solverDescTmp = f.read()
            solverDesc = solverDescTmp.replace('#snapshotpath#', outfolder + '/')
            solverDesc = solverDesc.replace('#netpath#', outfolder + '/net.prototxt')
            solverDesc = solverDesc.replace('#base_lr#', str(params['lr']))
            solverDesc = solverDesc.replace('#momentum#', str(params['momentum']))
            solverDesc = solverDesc.replace('#gamma#', str(params['lrDecay']))
            solverDesc = solverDesc.replace('#weightDecay#', str(params['weightDecay']))
            solverDesc = solverDesc.replace('\\', '/')

        with open(outfolder + r'/solver_forward.prototxt', 'w') as f:
            f.write(solverDesc)

    #load network
    if(params['SolverType'] == 'Adam'):
        solver = caffe.AdamSolver(outfolder + r'/solver_forward.prototxt')
    elif(params['SolverType'] == 'SGD'):
        solver = caffe.SGDSolver(outfolder + r'/solver_forward.prototxt')

    net = solver.net
    testnet = caffe.Net(outfolder + r'/net_test.prototxt', caffe.TEST)

    #init dataset  
    trainingQueueLength = 200
    logger.info('Init dataset...')
    logger.info('Sync Loading queue size:{}'.format(trainingQueueLength))
    
    data_queue_train = MultiQueue(trainingQueueLength)
    data_queue_loop = MultiQueue(trainingQueueLength)

    rootPath, file = os.path.split(params['dataset'])
    #build lightPool
    lightPoolMatrix = np.zeros((10,10,15,49,9,2))
    lightPool = []

    if(os.path.exists(rootPath + r'/lightMatrix_0_0_0.txt')):
       for aid in range(0, 10):
           for sid in range(0, 10):
               for rid in range(0, 15):
                   lightMatrix = np.loadtxt(rootPath + r'/lightMatrix_{}_{}_{}.txt'.format(aid, sid, rid)).reshape((49, 9, 2))
                   lightPoolMatrix[aid,sid,rid] = lightMatrix

       for aid in range(0, 10):
           for sid in range(0, 10):
               for rid in range(0, 15):
                   for lid in range(0, 49):
                       for vid in range(0, 9):
                           angle_x = lightPoolMatrix[aid, sid, rid, lid, vid, 0]
                           angle_y = lightPoolMatrix[aid, sid, rid, lid, vid, 1]
                           lightPool.append([lid, angle_x, angle_y])


    logger.info('Labeled data: {}'.format(params['dataset']))
    logger.info('BRDF Range:')
    logger.info('...Albedo:{}'.format(','.join(map(str,params['albedoRange']))))
    logger.info('...Spec:{}'.format(','.join(map(str,params['specRange'])))) 
    logger.info('...Roughness:{}'.format(','.join(map(str,params['roughnessRange']))))
    logger.info('...Additional BRDF Points: {}'.format(params['AdditionSample']))
    loader_train = Process(target = DataLoadProcess, args = (data_queue_train, params['dataset'], params, 0))
    loader_train.daemon = True
    loader_train.start()
  
    if(params['unlabelDataset'] != '' and params['renderLoop']):
        logger.info('Unlabel data: {}'.format(params['unlabelDataset']))
        loader_loop = Process(target = DataLoadProcess, args = (data_queue_loop, params['unlabelDataset'], params, 1))
        loader_loop.daemon = True
        loader_loop.start()

        
    logger.info('Waiting for load some data first...\n')
    time.sleep(60)
    loopdatasize = 0
    datasize = data_queue_train.get()
    brdfcubeShape = data_queue_train.get()
    amean, astd, smean, sstd, rmean, rstd = data_queue_train.get()
    albedoSpace, specSpace, roughnessSpace = data_queue_train.get()
    
    print(albedoSpace)
    print(specSpace)
    print(roughnessSpace)

    if(params['unlabelDataset'] != '' and params['renderLoop']):
        loopdatasize = data_queue_loop.get()
        nouse = data_queue_loop.get()
        amean_unlabel, astd_unlabel, smean_unlabel, sstd_unlabel, rmean_unlabel, rstd_unlabel = data_queue_loop.get()
        nouse = data_queue_loop.get()


    logger.info('Test data: {}'.format(params['testDataset']))
    path, file = os.path.split(params['testDataset'])

    logger.info('Test data (Full Set): {}'.format(params['testDataset']))
    testSet_Full = DataLoaderSimple(path, file, 9, 9, 14, 128, 128)
    if(params['envLighting']):
        testSet_Full.buildSubDataset(params['testalbedoRange'], params['testspecRange'], params['testroughnessRange'], params['lRange'], [0])#params['vRange'])
    else: 
        testSet_Full.buildSubDataset(params['testalbedoRange'], params['testspecRange'], params['testroughnessRange'], params['thetaRange'], params['phiRange'])
    testSet_Full.shuffle()        
    totaldatasize = datasize + loopdatasize
    
    logger.info('datasize = {}, unlabeldatasize = {}'.format(datasize, loopdatasize))
    logger.info('totalsize:{}'.format(totaldatasize))

    if(params['autoLoopRatio'] and params['renderLoop']):
        params['loopBatchLength'] = params['normalBatchLength'] * int(np.round(loopdatasize / datasize))
        logger.info('Normal Batch Length = {}, Loop Batch Length = {}'.format(params['normalBatchLength'], params['loopBatchLength']))
  
    params['checkPointStepEpochIteration'] = datasize / params['batchSize'] * params['checkPointStepEpoch']
    if(params['loopStartEpoch'] != -1):
       params['loopStartIteration'] = datasize / params['batchSize'] * params['loopStartEpoch']
       logger.info('Loop Start iteration = {}'.format(params['loopStartIteration']))


    iteration = 0
    loopiteration = 0
    total_iter = 0

    epoch = 0
    loopepoch = 0
    total_epoch = 0

    posInDataset = 0
    posInUnlabelDataset = 0

    avgLossEveryDisplayStep = 0
    avgLossEveryDisplayStep_a = 0
    avgLossEveryDisplayStep_s = 0
    avgLossEveryDisplayStep_r = 0

    avgLoopLossEveryDisplayStep = 0

    avgLossEveryCheckPoint = 0
    avgLossEveryCheckPoint_a = 0
    avgLossEveryCheckPoint_s = 0
    avgLossEveryCheckPoint_r = 0

    startTime = time.time()

    trainfigure = plt.figure(1)
    loopfigure = plt.figure(2)

    testfigure_a = plt.figure(3)
    testfigure_s = plt.figure(4)
    testfigure_r = plt.figure(5)
    testfigure_t = plt.figure(6)


    trainlosslist = []
    traintestlosslist = [[],[],[],[]]
    testlosslist = [[],[],[],[]]
    testlossFulllist = [[],[],[],[]]
    looplosslist = []

    thetaSpace = np.linspace(10.0 * math.pi / 180.0, 80.0 * math.pi / 180.0, 15)
    phiSpace = np.linspace(30.0 * math.pi / 180.0, 330.0 * math.pi / 180.0, 15)
    
    if(params['envLighting']):
        cubeSample = np.zeros((brdfcubeShape[0], brdfcubeShape[1], brdfcubeShape[2], len(params['lRange']), len(params['vRange'])))
    else:
        cubeSample = np.zeros((brdfcubeShape[0], brdfcubeShape[1], brdfcubeShape[2], 15, 15))

    brdfCubeSample = np.zeros((brdfcubeShape[0], brdfcubeShape[1], brdfcubeShape[2]))

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
            epoch = iteration * params['batchSize'] / datasize
            if(params['renderLoop']):
                loopepoch = loopiteration * params['batchSize'] / loopdatasize

        trainlosslist = list(np.loadtxt(outfolder + r'/trainloss.txt'))
        looplosslist = list(np.loadtxt(outfolder + r'/looploss.txt'))
        traintestlosslist = map(np.ndarray.tolist, np.split(np.loadtxt(outfolder + r'/traintestloss.txt').flatten(), 4))
        testlosslist = map(np.ndarray.tolist, np.split(np.loadtxt(outfolder + r'/testloss.txt').flatten(), 4))
        testlossFulllist = map(np.ndarray.tolist, np.split(np.loadtxt(outfolder + r'/testlossfull.txt').flatten(), 4))
        brdfCubeSample = np.loadtxt(outfolder + r'/brdfCubeSample.txt').flatten().reshape(brdfcubeShape[0], brdfcubeShape[1], brdfcubeShape[2])

    elif(restoreTraining == 2): #finetune
        net.copy_from(params['PreTrainNet'])
        logger.info('Finetune from {}'.format(params['PreTrainNet']))


    logger.info('Cube shape: {}, {}, {}'.format(brdfcubeShape[0], brdfcubeShape[1], brdfcubeShape[2]))
    logger.info('Start training...\n')


    while(True):
        #labeled training
        for i in range(0, params['normalBatchLength']):
            if(total_iter == params['nMaxIter'] or epoch == params['nMaxEpoch']):
                break
            #set data (from training set)
            img_data, brdf_data, name = data_queue_train.get()
  
            net.blobs['Data_Image'].data[...] = img_data
            net.blobs['Data_BRDF'].data[...] = brdf_data        

            for brdfid in name:
                aid, sid, rid, tid, pid = brdfid
                brdfCubeSample[aid, sid, rid] += 1

            solver.step(1)
    
            if(params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor'):
                iterLoss_a = net.blobs['RatioLoss'].data
                iterLoss_s = 0
                iterLoss_r = net.blobs['RoughnessLoss'].data
                iterLoss = iterLoss_a + iterLoss_r
            else:
                iterLoss = net.blobs['MSELoss'].data
                iterLoss_a = net.blobs['DiffuseLoss'].data
                iterLoss_s = net.blobs['SpecLoss'].data
                iterLoss_r = net.blobs['RoughnessLoss'].data

            avgLossEveryDisplayStep += iterLoss / params['displayStep']
            avgLossEveryDisplayStep_a += iterLoss_a / params['displayStep']
            avgLossEveryDisplayStep_s += iterLoss_s / params['displayStep']
            avgLossEveryDisplayStep_r += iterLoss_r / params['displayStep']
                        
            avgLossEveryCheckPoint_a += iterLoss_a / params['checkPointStepIteration']
            avgLossEveryCheckPoint_s += iterLoss_s / params['checkPointStepIteration']
            avgLossEveryCheckPoint_r += iterLoss_r / params['checkPointStepIteration']
            avgLossEveryCheckPoint += iterLoss / params['checkPointStepIteration']

            #display training status
            if(total_iter % params['displayStep'] == 0 and total_iter > 0):
                endTime = time.time()
                displayLoss = 0
                if(params['Channal'] == 'Albedo' or params['Channal'] == 'Ratio'):
                    displayLoss = avgLossEveryDisplayStep_a
                elif(params['Channal'] == 'Spec'):
                    displayLoss = avgLossEveryDisplayStep_s
                elif(params['Channal'] == 'Roughness'):
                    displayLoss = avgLossEveryDisplayStep_r
                else:
                    displayLoss = avgLossEveryDisplayStep

                logger.info('Total Iter:{} / Normal Iter:{} / Unlabel Iter {}'.format(total_iter, iteration, loopiteration))
                logger.info('Normal Epoch: {}, Loop Epoch: {}'.format(epoch, loopepoch))
                logger.info('Avg Loss = {}, Time = {}'.format(displayLoss, endTime - startTime))
                logger.info('Cube Sample:')
                logger.info('a:{}'.format(np.sum(brdfCubeSample, axis = (1,2))))
                logger.info('s:{}'.format(np.sum(brdfCubeSample, axis = (0,2))))
                logger.info('r:{}'.format(np.sum(brdfCubeSample, axis = (0,1))))
                startTime = time.time()	#reset timer
                trainlosslist.append(displayLoss)
                plt.figure(1)
                plt.plot(trainlosslist, 'rs-', label = 'test')
                plt.savefig(outfolder + r'/train.png')

                plt.figure(128)
                plt.gcf().clear()
                plt.plot(trainlosslist[-10::], 'rs-', label = 'test')
                plt.savefig(outfolder + r'/train_last10.png')

                avgLossEveryDisplayStep = 0
                avgLossEveryDisplayStep_a = 0
                avgLossEveryDisplayStep_s = 0
                avgLossEveryDisplayStep_r = 0

            #snapshot
            if(total_iter % params['checkPointStepIteration'] == 0 and total_iter > 0):
               statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                            'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'looplosslist':looplosslist,
                            'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist, 'testlossFulllist': testlossFulllist, 'cubeSample': cubeSample, 'brdfCubeSample':brdfCubeSample}
               dumpNetwork(outfolder, solver, 'iter_{}'.format(total_iter), statusDict)
                #test every epoch
               testnet.copy_from(outfolder + r'/iter_{}.caffemodel'.format(total_iter))

               traintestlosslist[0].append(avgLossEveryCheckPoint_a)
               traintestlosslist[1].append(avgLossEveryCheckPoint_s)
               traintestlosslist[2].append(avgLossEveryCheckPoint_r)
               traintestlosslist[3].append(avgLossEveryCheckPoint)

               logger.info('Testing on Full dataset...')
               testloss, testloss_a, testloss_s, testloss_r = testFitting(testnet, testSet_Full, 10000, params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor')
               displayTestLoss = 0
               if(params['Channal'] == 'Albedo' or params['Channal'] == 'Ratio'):
                  displayTestLoss = testloss_a
               elif(params['Channal'] == 'Spec'):
                  displayTestLoss = testloss_s
               elif(params['Channal'] == 'Roughness'):
                  displayTestLoss = testloss_r
               else:
                  displayTestLoss = testloss

               logger.info('Full loss = {}'.format(displayTestLoss))                  
               testlossFulllist[0].append(testloss_a)
               testlossFulllist[1].append(testloss_s)
               testlossFulllist[2].append(testloss_r)
               testlossFulllist[3].append(testloss)                 

               namelist = ['albedo','spec','roughness','total']
               for fid in range(3, 7):
                   plt.figure(fid)
                   plt.plot(traintestlosslist[fid-3], 'rs-', label = 'train')
                   plt.plot(testlosslist[fid-3], 'bs-', label = 'test')
                   plt.plot(testlossFulllist[fid-3], 'gs-', label = 'test_Full')                      
                   plt.savefig(outfolder + r'/test_{}.png'.format(namelist[fid-3]))
                    
                   fig = plt.figure(255 + fid)
                   plt.gcf().clear()
                   plt.plot(traintestlosslist[fid-3][-5::], 'rs-', label = 'train')
                   plt.plot(testlosslist[fid-3][-5::], 'bs-', label = 'test')
                   plt.plot(testlossFulllist[fid-3][-5::], 'gs-', label = 'test_Full')                      
                   plt.savefig(outfolder + r'/test_last5_{}.png'.format(namelist[fid-3]))
                
               avgLossEveryCheckPoint_a = 0
               avgLossEveryCheckPoint_s = 0
               avgLossEveryCheckPoint_r = 0
               avgLossEveryCheckPoint = 0               
                        
            if(iteration % params['checkPointStepEpochIteration'] == 0 and total_iter > 0 and iteration > 0 and params['checkPointStepEpoch'] != -1):
               statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                            'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'looplosslist':looplosslist,
                            'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist, 'testlossFulllist': testlossFulllist, 'cubeSample': cubeSample, 'brdfCubeSample':brdfCubeSample}
               dumpNetwork(outfolder, solver, 'epoch_{}'.format(epoch), statusDict)


            iteration = iteration + 1
            total_iter = total_iter + 1
            posInDataset = (posInDataset + params['batchSize']) % datasize
            epoch = iteration * params['batchSize'] / datasize


        if(params['renderLoop'] == True and total_iter >= params['loopStartIteration']):
            #self-augment process
            for k in range(0, params['loopBatchLength']):
                if(total_iter == params['nMaxIter'] or epoch == params['nMaxEpoch']):
                    break
                loop_batch, brdf, name = data_queue_loop.get()

                #forward unlabeled batch, getting predicted BRDF
                net.blobs['Data_Image'].data[...] = loop_batch
                net.forward()
                
                if(params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor'):
                    bS = params['batchSize']
                    ratio_value = np.minimum(np.maximum(0.05, np.exp(net.blobs['Out_Ratio'].data[0:bS,0])), 20.0)#np.exp(brdf[:,0,0,0])
                    predict_brdf = np.zeros((bS, 3))
                    predict_brdf[:, 1] = np.sqrt(np.random.uniform(0.0025, 1.0, bS))
                    for counter in range(0, bS):
                        if(ratio_value[counter] > 1.0):
                            predict_brdf[counter, 1] /= ratio_value[counter]
                    predict_brdf[:,0] = ratio_value * predict_brdf[:, 1]
                    predict_brdf[:,2] = np.copy(net.blobs['Out_Roughness_Fix'].data[...][0:bS, 0]) #brdf[:,1,0,0]
                else:
                    predict_brdf = np.copy(net.blobs['Out_LossFeature'].data[0:bS, :])

                #fix range
                predict_brdf[:,2] = np.maximum(-10.0, np.minimum(predict_brdf[:, 2], 0.0))

                visual_brdf = np.copy(predict_brdf)
                visual_brdf[:,2] = np.exp(visual_brdf[:,2])

                #render predicted BRDF
                if(params['envLighting']):
                    renderedBatch = renderOnlineEnvlightFromPool(visual_brdf, lightPool, (params['NetworkType'] == 'Ratio'))
                else:
                    renderedBatch = renderOnline_gray(visual_brdf, lightBatch)

                index_a = findIndex(predict_brdf[:, 0], albedoSpace)
                index_s = findIndex(predict_brdf[:, 1], specSpace)
                index_r = findIndex(predict_brdf[:, 2], roughnessSpace)
                
                a_bin = np.bincount(index_a.astype(int))
                s_bin = np.bincount(index_s.astype(int))
                r_bin = np.bincount(index_r.astype(int))

                for ind in range(0, len(index_a)):
                    brdfCubeSample[index_a[ind], index_s[ind], index_r[ind]] += 1

                #update network with generated data
                net.blobs['Data_Image'].data[...] = renderedBatch
                if(params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor'):
                    inNet_brdf = np.zeros((params['batchSize'], 2, 1, 1))
                    inNet_brdf[:,0,0,0] = np.log(predict_brdf[:,0] / predict_brdf[:,1])
                    inNet_brdf[:,1,0,0] = predict_brdf[:,2]
                    net.blobs['Data_BRDF'].data[...] = inNet_brdf
                else:
                    net.blobs['Data_BRDF'].data[...] = predict_brdf[:,:,np.newaxis,np.newaxis]
                       
                solver.step(1)

                if(params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor'):
                    iterLoss_a = net.blobs['RatioLoss'].data
                    iterLoss_s = 0
                    iterLoss_r = net.blobs['RoughnessLoss'].data
                    iterLoss = iterLoss_a + iterLoss_r
                else:
                    iterLoss = net.blobs['MSELoss'].data
                    iterLoss_a = net.blobs['DiffuseLoss'].data
                    iterLoss_s = net.blobs['SpecLoss'].data
                    iterLoss_r = net.blobs['RoughnessLoss'].data

                avgLossEveryDisplayStep += iterLoss / params['displayStep']
                avgLossEveryDisplayStep_a += iterLoss_a / params['displayStep']
                avgLossEveryDisplayStep_s += iterLoss_s / params['displayStep']
                avgLossEveryDisplayStep_r += iterLoss_r / params['displayStep']
                        
                #display training status
                if(total_iter % params['displayStep'] == 0 and iteration > 0):
                    endTime = time.time()
                    displayLoss = 0
                    if(params['Channal'] == 'Albedo' or params['Channal'] == 'Ratio'):
                        displayLoss = avgLossEveryDisplayStep_a
                    elif(params['Channal'] == 'Spec'):
                        displayLoss = avgLossEveryDisplayStep_s
                    elif(params['Channal'] == 'Roughness'):
                        displayLoss = avgLossEveryDisplayStep_r
                    else:
                        displayLoss = avgLossEveryDisplayStep

                    logger.info('Total Iter:{} / Normal Iter:{} / Unlabel Iter {}'.format(total_iter, iteration, loopiteration))
                    logger.info('Normal Epoch: {}, Loop Epoch: {}'.format(epoch, loopepoch))
                    logger.info('Avg Loss = {}, Time = {}'.format(displayLoss, endTime - startTime))
                    logger.info('Cube Sample:')
                    logger.info('a:{}'.format(np.sum(brdfCubeSample, axis = (1,2))))
                    logger.info('s:{}'.format(np.sum(brdfCubeSample, axis = (0,2))))
                    logger.info('r:{}'.format(np.sum(brdfCubeSample, axis = (0,1))))
                    startTime = time.time()	#reset timer
                    trainlosslist.append(displayLoss)
                    plt.figure(1)
                    plt.plot(trainlosslist, 'rs-', label = 'test')
                    plt.savefig(outfolder + r'\train.png')
                    avgLossEveryDisplayStep = 0
                    avgLossEveryDisplayStep_a = 0
                    avgLossEveryDisplayStep_s = 0
                    avgLossEveryDisplayStep_r = 0
                    
                #snapshot
                if(total_iter % params['checkPointStepIteration'] == 0 and total_iter > 0):
                    statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                            'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'looplosslist':looplosslist,
                            'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist, 'testlossFulllist': testlossFulllist, 'cubeSample': cubeSample, 'brdfCubeSample':brdfCubeSample}
                    dumpNetwork(outfolder, solver, 'iter_{}'.format(total_iter), statusDict)
                    testnet.copy_from(outfolder + r'/iter_{}.caffemodel'.format(total_iter))

                    traintestlosslist[0].append(avgLossEveryCheckPoint_a)
                    traintestlosslist[1].append(avgLossEveryCheckPoint_s)
                    traintestlosslist[2].append(avgLossEveryCheckPoint_r)
                    traintestlosslist[3].append(avgLossEveryCheckPoint)
                   
                    logger.info('Testing on Full dataset...')
                    testloss, testloss_a, testloss_s, testloss_r = testFitting(testnet, testSet_Full, 10000, params['NetworkType'] == 'Ratio' or params['NetworkType'] == 'RatioColor')
                    displayTestLoss = 0
                    if(params['Channal'] == 'Albedo' or params['Channal'] == 'Ratio'):
                        displayTestLoss = testloss_a
                    elif(params['Channal'] == 'Spec'):
                        displayTestLoss = testloss_s
                    elif(params['Channal'] == 'Roughness'):
                        displayTestLoss = testloss_r
                    else:
                        displayTestLoss = testloss

                    logger.info('Full loss = {}'.format(displayTestLoss))                  
                    testlossFulllist[0].append(testloss_a)
                    testlossFulllist[1].append(testloss_s)
                    testlossFulllist[2].append(testloss_r)
                    testlossFulllist[3].append(testloss)
                  
                    namelist = ['albedo','spec','roughness','total']
                    for fid in range(3, 7):
                        plt.figure(fid)
                        plt.plot(traintestlosslist[fid-3], 'rs-', label = 'train')
                        plt.plot(testlosslist[fid-3], 'bs-', label = 'test')
                        plt.plot(testlossFulllist[fid-3], 'gs-', label = 'test_Full')                      
                        plt.savefig(outfolder + r'/test_{}.png'.format(namelist[fid-3]))
                    
                        fig = plt.figure(255 + fid)
                        plt.gcf().clear()
                        plt.plot(traintestlosslist[fid-3][-5::], 'rs-', label = 'train')
                        plt.plot(testlosslist[fid-3][-5::], 'bs-', label = 'test')
                        plt.plot(testlossFulllist[fid-3][-5::], 'gs-', label = 'test_Full')                      
                        plt.savefig(outfolder + r'/test_last5_{}.png'.format(namelist[fid-3]))
                
                    avgLossEveryCheckPoint_a = 0
                    avgLossEveryCheckPoint_s = 0
                    avgLossEveryCheckPoint_r = 0
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

                   if(params['SolverType'] == 'Adam'):
                      solver = caffe.AdamSolver(outfolder + r'/solver_forward.prototxt')
                   elif(params['SolverType'] == 'SGD'):
                      solver = caffe.SGDSolver(outfolder + r'/solver_forward.prototxt')
                   net = solver.net
                   net.copy_from(modelfile)

                   os.remove(modelfile)
                   os.remove(statefile)

        if(total_iter == params['nMaxIter'] or epoch == params['nMaxEpoch']):
           #write final net
           statusDict = {'iteration': iteration, 'loopiteration': loopiteration, 'total_iter': total_iter, 'epoch': epoch, 'loopepoch': loopepoch,
                         'posInDataset': posInDataset, 'posInUnlabelDataset': posInUnlabelDataset, 'trainlosslist': trainlosslist, 'looplosslist':looplosslist,
                         'traintestlosslist': traintestlosslist, 'testlosslist': testlosslist, 'testlossFulllist': testlossFulllist, 'cubeSample': cubeSample, 'brdfCubeSample':brdfCubeSample}
           dumpNetwork(outfolder, solver, 'final', statusDict)
           if(autoTest):
              logger.info('Visualizing...')
              os.system(r'python TestBRDF.py {}/final.caffemodel {} {}'.format(outfolder, testparampath, gpuid))
           break







