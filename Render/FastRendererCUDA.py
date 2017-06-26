# FastRendererCUDA.py
# PyCUDA based renderer class

import numpy as np
import math
import os, sys
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)
sys.path.append(root_path + r'/Utils')
import cv2

import pycuda.driver as cuda
from pycuda.compiler import SourceModule, compile

from utils import load_pfm, save_pfm, getTexCube, pfmFromBuffer, pfmToBuffer, genMipMap

import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.chdir(working_path)

def tex2DToGPU(tex):
    nChannal = 1 if (len(tex.shape) == 2) else 3

    if(nChannal == 3):
        #Add padding channal
        tex = np.dstack((tex, np.ones((tex.shape[0], tex.shape[1]))))
        tex = np.ascontiguousarray(tex).astype(np.float32)
        texGPUArray = cuda.make_multichannel_2d_array(tex, 'C')
    else:
        texGPUArray = cuda.np_to_array(tex, 'C')

    return texGPUArray

def texCubeToGPUMipmap(texCube):
    #assume width = height and is 2^x
    texMipMapList = genMipMap(texCube)
    texMipMapGPUArray = []
    for k in range(0, len(texMipMapList)):
        texMipMapGPUArray.append(texCubeToGPU(texMipMapList[k]))

    return texMipMapGPUArray



def texCubeToGPU(texCube):
    descr = cuda.ArrayDescriptor3D()
    descr.width = texCube.shape[2]
    descr.height = texCube.shape[1]
    descr.depth = 6
    descr.format = cuda.dtype_to_array_format(texCube.dtype)
    descr.num_channels = 4
    descr.flags = cuda.array3d_flags.CUBEMAP

    texCubeArray = cuda.Array(descr)
    copy = cuda.Memcpy3D()
    copy.set_src_host(texCube)
    copy.set_dst_array(texCubeArray)
    copy.width_in_bytes = copy.src_pitch = texCube.strides[1] #d*h*w*c
    copy.src_height = copy.height = texCube.shape[1]
    copy.depth = 6
    
    copy()

    return texCubeArray

class FastRenderEngine(object):

    cudadevice = None
    cudacontext = None

    nDiffCount = 128
    nSpecCount = 512

    matrixView = np.identity(4)
    matrixProj = np.identity(4)
    matrixLight = np.identity(4)

    out_buffer = None
    out_buffer_gpu = None

    geoType = 'Sphere'
    lightMode = 'Point'

    lightPos = np.zeros((8, 4))
    lightIntensity = np.zeros((8, 3))
    lightStatus = np.zeros(8)
    
    texRef_Light = None
    texRef_Light_List = []
    texRef_Albedo = None
    texRef_Spec = None
    texRef_Roughness = None
    texRef_Normal = None

    texCubeList_gpu = {}    
    nCubeRes = 512
    texAlbedo_gpu = None
    texSpec_gpu = None
    texRoughness_gpu = None
    texNormal_gpu = None

    cuda_mod = None
    renderMode = 0

    sphere_normal_map = None

    def __init__(self, gpuid):
        cuda.init()
        self.cudadevice = cuda.Device(gpuid)
        self.cudacontext = self.cudadevice.make_context()
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + r'/PixelShader.cu', 'r') as f:
            cudaCode = f.read()
        self.cuda_mod = SourceModule(cudaCode, include_dirs = [dir_path], no_extern_c = True, options = ['-O0'])

        for k in range(0, 9):
            self.texRef_Light_List.append(self.cuda_mod.get_texref('texCube{}'.format(k)))

        self.texRef_Albedo = self.cuda_mod.get_texref('albedo')
        self.texRef_Spec = self.cuda_mod.get_texref('spec')
        self.texRef_Roughness = self.cuda_mod.get_texref('roughness')
        self.texRef_Normal = self.cuda_mod.get_texref('normal')

        self.sphere_normal_map = load_pfm(dir_path + r'/sphere_normal.pfm')

        import atexit
        atexit.register(self.cudacontext.pop)


    def SetSampleCount(self, diffCount, specCount):
        self.nDiffCount = diffCount
        self.nSpecCount = specCount


    def SetLightXform(self, rotXAngle, rotYAngle):
        matRotX = np.array([[1,0,0,0],
                           [0, math.cos(rotXAngle * math.pi / 180.0), -math.sin(rotXAngle * math.pi / 180.0),0],
                           [0, math.sin(rotXAngle * math.pi / 180.0), math.cos(rotXAngle * math.pi / 180.0),0],
                           [0,0,0,1]]).transpose()
        matRotY = np.array([[math.cos(rotYAngle * math.pi / 180.0),0, math.sin(rotYAngle * math.pi / 180.0), 0],
                           [0, 1, 0, 0],
                           [-math.sin(rotYAngle * math.pi / 180.0), 0, math.cos(rotYAngle * math.pi / 180.0),0],
                           [0,0,0,1]]).transpose()

        self.matrixLight = (matRotY.dot(matRotX)).astype(np.float32)
        self.matrixLight = np.ascontiguousarray(self.matrixLight)



    def SetGeometry(self, type):
        self.geoType = type
        if(type == 'Plane'):
            normal_default = np.dstack((np.ones((256,256)), np.zeros((256,256)), np.zeros((256,256))))
        else:
            normal_default = self.sphere_normal_map
        self.SetNormalMap(normal_default)

    def SetRenderMode(self, mode):
        self.renderMode = mode

    def PreLoadAllLight(self, lightFile):
        folderPath, name = os.path.split(lightFile)
        with open(lightFile, 'r') as f:
            lightIDs = map(int, f.read().strip().split('\n'))
        for lid in lightIDs:
            crossImg = load_pfm(folderPath + r'/{:04d}.pfm'.format(lid))
            self.nCubeRes = crossImg.shape[1] / 4
            self.texCubeList_gpu[lid] = texCubeToGPUMipmap(getTexCube(crossImg))# texCubeToGPU(getTexCube(crossImg))
        
    def SetEnvLightByID(self, id, rotXAngle = 0, rotYAngle = 0):
        self.SetLightXform(rotXAngle, rotYAngle)

        for k in range(0, len(self.texCubeList_gpu[id])):
            self.texRef_Light_List[k].set_array(self.texCubeList_gpu[id][k])
            self.texRef_Light_List[k].set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
            self.texRef_Light_List[k].set_filter_mode(cuda.filter_mode.LINEAR)
            self.texRef_Light_List[k].set_address_mode(0, cuda.address_mode.WRAP)
            self.texRef_Light_List[k].set_address_mode(1, cuda.address_mode.WRAP)

        self.lightMode = 'Env'

    def SetPointLight(self, slot, x, y, z, w, r, g, b):
        self.lightPos[slot, :] = [x, y, z, w]
        self.lightIntensity[slot, :] = [b, g, r]
        self.lightStatus[slot] = 1
        self.lightMode = 'Point'
         
    def SetAlbedoMap(self, albedo):
        self.texAlbedo_gpu = tex2DToGPU(albedo.astype(np.float32))
        self.texRef_Albedo.set_array(self.texAlbedo_gpu)
        self.texRef_Albedo.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        self.texRef_Albedo.set_filter_mode(cuda.filter_mode.LINEAR)        
        self.texRef_Albedo.set_address_mode(0, cuda.address_mode.WRAP)
        self.texRef_Albedo.set_address_mode(1, cuda.address_mode.WRAP)

    def SetAlbedoValue(self, albedo):
        self.SetAlbedoMap(albedo * np.ones((256,256,3)))

    def SetSpecMap(self, spec):
        self.texSpec_gpu = tex2DToGPU(spec.astype(np.float32))
        self.texRef_Spec.set_array(self.texSpec_gpu)
        self.texRef_Spec.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        self.texRef_Spec.set_filter_mode(cuda.filter_mode.LINEAR)
        self.texRef_Spec.set_address_mode(0, cuda.address_mode.WRAP)
        self.texRef_Spec.set_address_mode(1, cuda.address_mode.WRAP)

    def SetSpecValue(self, spec):
        self.SetSpecMap(spec * np.ones((256,256,3)))

    def SetRoughnessMap(self, roughness):
        if(len(roughness.shape) == 3):
            roughness = roughness[:,:,0]
        self.texRoughness_gpu = tex2DToGPU(roughness.astype(np.float32))
        self.texRef_Roughness.set_array(self.texRoughness_gpu)
        self.texRef_Roughness.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        self.texRef_Roughness.set_filter_mode(cuda.filter_mode.LINEAR)
        self.texRef_Roughness.set_address_mode(0, cuda.address_mode.WRAP)
        self.texRef_Roughness.set_address_mode(1, cuda.address_mode.WRAP)

    def SetRoughnessValue(self, roughness):
        self.SetRoughnessMap(roughness * np.ones((256,256)))

    def SetNormalMap(self, normal):
        self.texNormal_gpu = tex2DToGPU(normal.astype(np.float32))
        self.texRef_Normal.set_array(self.texNormal_gpu)
        self.texRef_Normal.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        self.texRef_Normal.set_filter_mode(cuda.filter_mode.LINEAR)
        self.texRef_Normal.set_address_mode(0, cuda.address_mode.WRAP)
        self.texRef_Normal.set_address_mode(1, cuda.address_mode.WRAP)

    def SetCamera(self, ox, oy, oz, lx, ly, lz, ux, uy, uz, fov, clipNear, clipFar, width, height):
        upDir = np.array([ux,uy,uz])
        eyePos = np.array([ox,oy,oz])
        eyeVec = np.array([lx-ox, ly-oy, lz-oz])
        R2 = eyeVec / np.linalg.norm(eyeVec)
        R0 = np.cross(upDir, R2)
        R0 = R0 / np.linalg.norm(R0)
        R1 = np.cross(R2, R0)

        D0 = R0.dot(-eyePos)
        D1 = R1.dot(-eyePos)
        D2 = R2.dot(-eyePos)

        self.matrixView = np.array([[R0[0], R0[1], R0[2], D0],
                                    [R1[0], R1[1], R1[2], D1],
                                    [R2[0], R2[1], R2[2], D2],
                                    [0,0,0,1]]).transpose().astype(np.float32)
        self.matrixView = np.ascontiguousarray(self.matrixView)

        sinFov = math.sin(0.5*fov)
        cosFov = math.cos(0.5*fov)

        height1 = cosFov / sinFov
        width1 = height1 / (float(width) / float(height))
        fRange = clipFar / (clipFar - clipNear)

        self.matrixProj = np.array([[width1, 0, 0, 0],
                                    [0, height1, 0, 0],
                                    [0, 0, fRange, 1],
                                    [0, 0, -fRange * clipNear, 0]]).astype(np.float32)
        self.matrixProj = np.ascontiguousarray(self.matrixProj)
        
        self.matrixView_gpu = cuda.mem_alloc(self.matrixView.nbytes)
        cuda.memcpy_htod(self.matrixView_gpu, self.matrixView)
        self.matrixProj_gpu = cuda.mem_alloc(self.matrixProj.nbytes)
        cuda.memcpy_htod(self.matrixProj_gpu, self.matrixProj)

        self.out_buffer = np.zeros((height, width, 3)).astype(np.float32)
        self.out_buffer_gpu = cuda.mem_alloc(self.out_buffer.nbytes)

    def Render(self):
        renderFunc = self.cuda_mod.get_function('PS_Render_{}_{}'.format(self.geoType, self.lightMode))

        grid_x = (self.out_buffer.shape[1] - 1) / 16 + 1
        grid_y = (self.out_buffer.shape[0] - 1) / 16 + 1

        if(self.lightMode == 'Env'):
            texrefList = [self.texRef_Albedo, self.texRef_Spec, self.texRef_Normal, self.texRef_Roughness, 
                          self.texRef_Light_List[0], self.texRef_Light_List[1], self.texRef_Light_List[2],
                          self.texRef_Light_List[3], self.texRef_Light_List[4], self.texRef_Light_List[5],
                          self.texRef_Light_List[6], self.texRef_Light_List[7], self.texRef_Light_List[8]]
            matWorldToLight = np.ascontiguousarray(np.linalg.inv(self.matrixLight).astype(np.float32))
            matWorldToLight_gpu = cuda.mem_alloc(matWorldToLight.nbytes)
            cuda.memcpy_htod(matWorldToLight_gpu, matWorldToLight)
            renderFunc(self.out_buffer_gpu, np.int32(self.out_buffer.shape[1]), np.int32(self.out_buffer.shape[0]), 
                       self.matrixProj_gpu, self.matrixView_gpu, 
                       matWorldToLight_gpu, np.int32(self.nCubeRes), 
                       np.int32(self.nDiffCount), np.int32(self.nSpecCount), np.int32(self.renderMode),
                       block = (16,16,1), grid = (grid_x,grid_y,1), texrefs=texrefList)
        elif(self.lightMode == 'Point'):
            lightStatus_gpu = cuda.mem_alloc(self.lightStatus.astype(np.int32).nbytes)
            cuda.memcpy_htod(lightStatus_gpu, self.lightStatus.astype(np.int32))
            lightIntensity_gpu = cuda.mem_alloc(self.lightIntensity.astype(np.float32).nbytes)
            cuda.memcpy_htod(lightIntensity_gpu, self.lightIntensity.astype(np.float32))
            lightPos_gpu = cuda.mem_alloc(self.lightPos.astype(np.float32).nbytes)
            cuda.memcpy_htod(lightPos_gpu, self.lightPos.astype(np.float32))

            renderFunc(self.out_buffer_gpu, np.int32(self.out_buffer.shape[1]), np.int32(self.out_buffer.shape[0]), 
                       self.matrixProj_gpu, self.matrixView_gpu, 
                       lightStatus_gpu, lightIntensity_gpu, lightPos_gpu, np.int32(self.renderMode),
                       block = (16,16,1), grid = (grid_x,grid_y,1), texrefs=[self.texRef_Albedo, self.texRef_Spec, self.texRef_Normal, self.texRef_Roughness])

        cuda.memcpy_dtoh(self.out_buffer, self.out_buffer_gpu)
        return np.copy(self.out_buffer)