# RenderSVBRDFDataset.py
# Script for generate SVBRDF-Net training and testing data

import random, os, sys, time, glob, math, shutil, pickle
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)
os.chdir(working_path)
sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')
import caffe
import numpy as np
import cv2

from utils import save_pfm, load_pfm, pfmFromBuffer, pfmToBuffer, toHDR, toLDR, renormalize, normalizeAlbedoSpec, normalBatchToThetaPhiBatch, thetaPhiBatchToNormalBatch, DataLoaderSVBRDF, RealDataLoaderSVBRDF, make_dir, autoExposure
from FastRendererCUDA import FastRenderEngine
 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


params_global = {}
os.chdir(working_path)

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

lightIDToEnumerateID = {}
for id, lid in enumerate(lightID):
    lightIDToEnumerateID[lid] = id


def renderOnlineEnvlight(brdfBatch, onlineRender, lightIDs = [], lightXforms = []):
    imgBatch = np.zeros((brdfBatch.shape[0], 3, 384, 384))
    if(lightIDs == []):
        lightIDs = random.sample(lightID, brdfBatch.shape[0])
    if(lightXforms == []):
        angle_y = np.random.uniform(0.0, 360.0, brdfBatch.shape[0])
        angle_x = np.random.uniform(-45.0, 45.0, brdfBatch.shape[0])
    else:
        angle_y = lightXforms[1]
        angle_x = lightXforms[0]

    for i in range(0, brdfBatch.shape[0]):

        onlineRender.SetEnvLightByID(lightIDs[i] + 1) 
        onlineRender.SetLightXform(angle_x[i], angle_y[i])
                  

        onlineRender.SetAlbedoMap(brdfBatch[i,0:3,:,:].transpose((1,2,0)))
        onlineRender.SetSpecValue(brdfBatch[i,3:6,0,0])
        onlineRender.SetRoughnessValue(brdfBatch[i,6,0,0])
        onlineRender.SetNormalMap((2.0 * brdfBatch[i,7:10,:,:] - 1.0).transpose((1,2,0)))
#        onlineRender.SetRenderMode(1)
        img = onlineRender.Render()

        #Auto Exposure and white balance
        onlineRender.SetAlbedoValue([1.0, 1.0, 1.0])
        onlineRender.SetSpecValue([0.0, 0.0, 0.0])
        normal_one = np.dstack((np.ones((256,256)), np.zeros((256,256)), np.zeros((256,256))))
        onlineRender.SetNormalMap(normal_one)
        img_norm = onlineRender.Render()
        normValue = np.mean(img_norm, axis = (0,1))
        img = 0.5 * img / normValue
        #img = autoExposure(img)
        imgBatch[i, :, :, :] = img.transpose((2,0,1))

    return imgBatch, normValue



if __name__ == '__main__':
    data_root = sys.argv[1]
    data_tag = sys.argv[2]
    gpuid = int(sys.argv[3])
    renderType = int(sys.argv[4])
    startTag = int(sys.argv[5])
    endTag = int(sys.argv[6])
    renderTag = sys.argv[7]
    if(len(sys.argv) == 9):
        out_root = sys.argv[8]
    else:
        out_root = data_root

    labeled_file_in = data_root + r'/{}/Labeled/trainingdata.txt'.format(data_tag)
    test_file_in = data_root + r'/{}/Test/test.txt'.format(data_tag) 

    rendered_labeled_out = out_root + r'/{}/Labeled'.format(data_tag)
    rendered_test_out = out_root + r'/{}/Test'.format(data_tag) 

    specular_file = data_root + r'/{}/Labeled/specroughness.txt'.format(data_tag)
    lightpool_file = params_global['envMapFolder'] + r'/lightPool_{}.dat'.format(data_tag)

    AugmentRender = FastRenderEngine(gpuid)
    AugmentRender.SetGeometry('Plane')
    AugmentRender.PreLoadAllLight(r'{}/light.txt'.format(params_global['envMapFolder']))
    AugmentRender.SetSampleCount(128, 1024)
    fovRadian = 60.0 / 180.0 * math.pi
    cameraDist = 1.0  / (math.tan(fovRadian / 2.0))
    AugmentRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 384, 384)

    specList_final = {}
    roughnessList_final = {}
    with open(specular_file, 'r') as f:
         rawList = f.read().strip().split('\n')

    for t in rawList:
        mid = int(t.split(',')[0])
        spec = float(t.split(',')[1])
        roughness = float(t.split(',')[2])
        specList_final[mid] = spec
        roughnessList_final[mid] = roughness

    lightPool = pickle.load(open(lightpool_file, 'rb'))
    lightNormPool = {}

#    precompute auto-exposure factor
    normal_one = np.dstack((np.ones((256,256)), np.ones((256,256)), np.ones((256,256))))
    for m in specList_final.keys():
        lightNormPool[m] = np.zeros((len(lightID), 10, 3))
        for id, lid in enumerate(lightID):
            for v in range(0, lightPool[m].shape[1]):
                AugmentRender.SetEnvLightByID(lid + 1, lightPool[m][id, v, 0], lightPool[m][id, v, 1])
                AugmentRender.SetAlbedoValue([1.0, 1.0, 1.0])
                AugmentRender.SetSpecValue([0.0, 0.0, 0.0])
                AugmentRender.SetRoughnessValue(0)
                AugmentRender.SetNormalMap(normal_one)
                img_diffuse = AugmentRender.Render()
                norm = np.mean(img_diffuse, axis = (0,1))
                lightNormPool[m][id, v] = norm

    with open(params_global['envMapFolder'] + r'/lightNormPool_{}.dat'.format(data_tag), 'wb') as f:
        pickle.dump(lightNormPool, f)

    print('Factor done.\n')

    #render training
    if(renderTag == 'train' or renderTag == 'all'):
        make_dir(rendered_labeled_out)
        path, file = os.path.split(labeled_file_in)
        dataset = DataLoaderSVBRDF(path, file, 384, 384, True)
        begin = 0 if startTag == -1 else startTag
        end = dataset.dataSize if endTag == -1 else endTag

        for k in range(begin, end):
            if(k % 1000 == 0):
                print('{}/{}'.format(k, dataset.dataSize))
            name = map(int, dataset.dataList[k].split('_'))
            m,l,v,o = name           
            brdfbatch = np.ones((1, 10, 384, 384))

            albedo = load_pfm(path + r'/m_{}/gt_{}_albedo.pfm'.format(m, o))
            normal = load_pfm(path + r'/m_{}/gt_{}_normal.pfm'.format(m, o))
            specvalue = specList_final[m]
            roughnessvalue = roughnessList_final[m]

            brdfbatch[0,0:3,:,:] = albedo.transpose(2,0,1)
            brdfbatch[0,3:6,:,:] = specvalue
            brdfbatch[0,6,:,:] = roughnessvalue
            brdfbatch[0,7:10,:,:] = normal.transpose(2,0,1)

            lids = [l]
            rotX = lightPool[m][lightIDToEnumerateID[l], v, 0]
            rotY = lightPool[m][lightIDToEnumerateID[l], v, 1]
            lxforms = [[rotX],[rotY]]

            imgbatch, normValue = renderOnlineEnvlight(brdfbatch, AugmentRender, lids, lxforms)
            outfolder = rendered_labeled_out + r'/m_{}'.format(m)        
            make_dir(outfolder)
        
            #0:HDR 1:LDR 2:BOTH
           
            if(renderType == 0 or renderType == 2):
                save_pfm(outfolder + r'/{}_{}_{}_{}_image.pfm'.format(m, l, v, o), imgbatch[0,:,:,:].transpose((1,2,0)))
            if(renderType == 1 or renderType == 2):
                cv2.imwrite(outfolder + r'/{}_{}_{}_{}_image.jpg'.format(m, l, v, o), toLDR(imgbatch[0,:,:,:].transpose((1,2,0))))

    if(renderTag == 'test' or renderTag == 'all'):
    #render test
        make_dir(rendered_test_out)
        path, file = os.path.split(test_file_in)
        dataset = DataLoaderSVBRDF(path, file, 384, 384, False)
        for k in range(0, dataset.dataSize):
            if(k % 1000 == 0):
                print('{}/{}'.format(k, dataset.dataSize))
            name = map(int, dataset.dataList[k].split('_'))
            m,l,v,o = name           
            brdfbatch = np.ones((1, 10, 384, 384))

            albedo = load_pfm(path + r'/m_{}/gt_{}_albedo.pfm'.format(m, o))
            normal = load_pfm(path + r'/m_{}/gt_{}_normal.pfm'.format(m, o))
            specvalue = specList_final[m]
            roughnessvalue = roughnessList_final[m]

            brdfbatch[0,0:3,:,:] = albedo.transpose(2,0,1)
            brdfbatch[0,3:6,:,:] = specvalue
            brdfbatch[0,6,:,:] = roughnessvalue
            brdfbatch[0,7:10,:,:] = normal.transpose(2,0,1)

            lids = [l]
            rotX = lightPool[m][lightIDToEnumerateID[l], v, 0]
            rotY = lightPool[m][lightIDToEnumerateID[l], v, 1]
            lxforms = [[rotX],[rotY]]            

            imgbatch, normfactor = renderOnlineEnvlight(brdfbatch, AugmentRender, lids, lxforms)
            outfolder = rendered_test_out + r'/m_{}'.format(m)        
            make_dir(outfolder)

            #0:HDR 1:LDR 2:BOTH
            if(renderType == 0 or renderType == 2):
                save_pfm(outfolder + r'/{}_{}_{}_{}_image.pfm'.format(m, l, v, o), imgbatch[0,:,:,:].transpose((1,2,0)))
            if(renderType == 1 or renderType == 2):
                cv2.imwrite(outfolder + r'/{}_{}_{}_{}_image.jpg'.format(m, l, v, o), toLDR(imgbatch[0,:,:,:].transpose((1,2,0))))
