# TestSVBRDF.py
# Test script for SVBRDF-Net

import random, os, time, sys
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)
sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')
import caffe
from utils import save_pfm, load_pfm, pfmToBuffer, pfmFromBuffer, autoExposure
import numpy as np
import math
import logging
import matplotlib.pyplot as plt

import itertools 

import glob

from FastRendererCUDA import FastRenderEngine

import sys
import pickle, json
import shutil

from ConfigParser import ConfigParser, SafeConfigParser

from multiprocessing import Process
from multiprocessing import Queue as MultiQueue

import cv2
import jinja2
os.chdir(working_path)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
params_global = {}

lightID = []

with open('folderPath_SVBRDF.txt', 'r') as f:
    params_global['geometryPath'] = r'../Render/plane.obj'
    params_global['scriptRoot'] = r'../Utils'
    params_global['outFolder'] = f.readline().strip()
    params_global['envMapFolder'] = f.readline().strip()
      
with open(params_global['envMapFolder'] + r'/light.txt', 'r') as f:
     lightID = map(int, f.read().strip().split('\n'))
     lightID = list(np.array(lightID) - 1)

def toLDR(img):
    img_out = img ** (1.0 / 2.2)
    img_out = np.minimum(255, img_out * 255)
    return img_out.astype(np.uint8)

def toHDR(img):
    img = img / 255.0
    img_out = img ** 2.2
    return img_out.astype(np.float32)

def test(testnet, img):
    testnet.blobs['Data_Image'].data[...] = img
    testnet.forward()

    albedo_p = testnet.blobs['ConvFinal_Albedo'].data[0,:,:,:].transpose((1,2,0))
    spec_p = np.exp(testnet.blobs['ConvFinal_SpecAlbedo'].data.flatten())

    roughness_p = np.exp(testnet.blobs['ConvFinal_Roughness'].data.flatten())
    normal_p = testnet.blobs['ConvFinal_Normal'].data[0,:,:,:].transpose((1,2,0))
                        
                                                    
    return albedo_p, spec_p, roughness_p, normal_p


def renderSpecBall(renderer, spec, roughness):
    renderer.SetGeometry('Sphere')
    renderer.SetCamera(0, 0, cameraDist_1, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 256, 256)
    renderer.SetSampleCount(128, 512)
    renderer.SetPointLight(0, 1, 1, 1, 0, 0.2, 0.2, 0.2)
    renderer.SetAlbedoValue([0.0, 0.0, 0.0])
    renderer.SetSpecValue(spec)
    renderer.SetRoughnessValue(roughness)
    sphere = renderer.Render()
    #Mask
    renderer.SetPointLight(0, 1, 1, 1, 0, 0.2, 0.2, 0.2)
    renderer.SetAlbedoValue([1.0, 1.0, 1.0])
    renderer.SetSpecValue([0.0,0.0,0.0])
    renderer.SetRoughnessValue(roughness)
    renderer.SetRenderMode(1)
    mask = renderer.Render()
    renderer.SetRenderMode(0)
    sphere[mask == 0] = 1

    renderer.SetGeometry('Plane')
    renderer.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 256, 256)
    renderer.SetSampleCount(128, 512)

    return sphere

def renderRelighting(renderer, albedo, spec, roughness, normal):
    renderer.SetPointLight(0, 0.27, -0.25, 1, 0, 0.6, 0.6, 0.6)
    renderer.SetAlbedoMap(albedo)
    renderer.SetSpecValue(spec)
    renderer.SetRoughnessValue(roughness)

    normal = normal * 2.0 - 1.0
    normal[0] = normal[0] * 2.5
    len = np.linalg.norm(normal, axis = 2)
    normal = normal / np.dstack((len, len, len))
    normal = 0.5*(normal + 1.0)

    renderer.SetNormalMap(normal*2.0 - 1.0)
    img = renderer.Render()

    renderer.SetEnvLightByID(43, 30, -10.0)
    renderer.SetAlbedoMap(albedo)
    renderer.SetSpecValue(spec)
    renderer.SetRoughnessValue(roughness)
    renderer.SetNormalMap(normal*2.0 - 1.0)
    img_1 = renderer.Render()

    return 1.2 * img + 0.8 * img_1


lightIDToEnumerateID = {}
for id, lid in enumerate(lightID):
    lightIDToEnumerateID[lid] = id


if __name__ == '__main__':
    modelFile = sys.argv[1]
    testSetPath = sys.argv[2]
    gpuid = int(sys.argv[3])

    imgw = 256

    fovRadian = 60.0 / 180.0 * math.pi
    cameraDist = 1.0  / (math.tan(fovRadian / 2.0)) 
    cameraDist_1 = 1.5 / (math.tan(fovRadian / 2.0))

    RelightingRender = FastRenderEngine(gpuid)
    RelightingRender.SetGeometry('Plane')
    RelightingRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 256, 256)
    RelightingRender.SetSampleCount(128, 512)
    RelightingRender.PreLoadAllLight(r'{}/light.txt'.format(params_global['envMapFolder']))

    caffe.set_mode_gpu()
    caffe.set_device(gpuid)
    path, file = os.path.split(modelFile)
    modelFolder = path
    testnet = caffe.Net(path + r'/net_test.prototxt', caffe.TEST)
    testnet.copy_from(modelFile)

    path, file = os.path.split(testSetPath)
    with open(testSetPath, 'r') as f:
        filenames = f.read().strip().split('\n')

    np.random.shuffle(filenames)

    pixelCnt = imgw*imgw

    tag_testSet = 'test'
    jinjiaEnv = jinja2.Environment(loader = jinja2.FileSystemLoader('./')).get_template('template_visSVBRDFNet_Real.html')
    renderContext = {}
    renderContext['networkTag'] = modelFolder
    renderContext['dataList'] = []
    visualDir = modelFolder + r'/visualize_{}'.format(tag_testSet)
    if(os.path.exists(visualDir) == False):
       os.makedirs(visualDir)

    for filename in filenames:
        fullpath = path + r'/{}.jpg'.format(filename.strip())
        print('Test {}\n'.format(filename.strip()))
        img = toHDR(cv2.imread(fullpath))

        img_in = np.zeros((1,3,256,256))
        img_in[0,:,:,:] = img.transpose((2,0,1))
        albedo_p, spec_p, roughness_p, normal_p = test(testnet, img_in)
        factor = 0.5 / np.mean(np.linalg.norm(albedo_p, axis = 2))
        albedo_p = albedo_p * factor
        spec_p = spec_p * factor
        specball_fit = renderSpecBall(RelightingRender, spec_p, roughness_p)

        data = {}
        data_id = '{}'.format(filename.strip())

        cv2.imwrite(visualDir + r'/{}_img.jpg'.format(data_id), toLDR(img))
        cv2.imwrite(visualDir + r'/{}_albedo_fit.jpg'.format(data_id), toLDR(albedo_p))
        cv2.imwrite(visualDir + r'/{}_specball_fit.jpg'.format(data_id), toLDR(specball_fit))
        cv2.imwrite(visualDir + r'/{}_normal_fit.jpg'.format(data_id), toLDR(normal_p))
        cv2.imwrite(visualDir + r'/{}_relighting_fit.jpg'.format(data_id), toLDR(renderRelighting(RelightingRender, albedo_p, spec_p, roughness_p, normal_p)))

        data['ID'] = data_id
        data['img'] = visualDir + r'/{}_img.jpg'.format(data_id)
        data['albedo_fit'] = visualDir + r'/{}_albedo_fit.jpg'.format(data_id)
        data['specball_fit'] = visualDir + r'/{}_specball_fit.jpg'.format(data_id)
        data['normal_fit'] = visualDir + r'/{}_normal_fit.jpg'.format(data_id)
        data['relighting_fit'] = visualDir + r'/{}_relighting_fit.jpg'.format(data_id)

        renderContext['dataList'].append(data)
        renderedHtml = jinjiaEnv.render(renderContext)
        with open(modelFolder + r'/visResult_{}.html'.format(tag_testSet), 'w') as f1:
             f1.write(renderedHtml)