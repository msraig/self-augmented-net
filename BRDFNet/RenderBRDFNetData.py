import sys, os
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)
sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')
import numpy as np
import math
from utils import load_pfm, save_pfm, pfmFromBuffer, pfmToBuffer, make_dir, toLDR
from operator import itemgetter
import caffe

from FastRendererCUDA import FastRenderEngine

params = {}
os.chdir(working_path)

#dense sampled dataset, single channal
albedoCnt = 10
specCnt = 10
roughnessCnt = 15
thetaCnt = 15
phiCnt = 15

RenderOutput = r''

with open('folderPath.txt', 'r') as f:
    params['geometryPath'] = r'../Render/sphere.obj'
    params['scriptRoot'] = r'../Utils'
    params['outFolder'] = f.readline().strip()
    params['envMapFolder'] = f.readline().strip()

def idToBRDFid(id, aCnt, sCnt, rCnt):
    aCnt = 10
    sCnt = 10
    rCnt = 15

    aid = id / (sCnt * rCnt)
    sid = (id % (sCnt * rCnt)) / rCnt
    rid = id - aid * (sCnt * rCnt) - sid * rCnt

    return aid, sid, rid

def sampleCube(aid_list, sid_list, rid_list, full_list, dataset_name):
    out_list = []
    for a in aid_list:
        for s in sid_list:
            for r in rid_list:
                for tid in range(0, thetaCnt):
                    for pid in range(0, phiCnt):
                        out_list.append('{}_{}_{}_{}_{}'.format(a, s, r, tid, pid))
    unlabel_list = list(set(full_list) - set(out_list))
    with open(RenderOutput + r'/train/train_{}.txt'.format(dataset_name), 'w') as f:
        for x in out_list:
            f.write(x)
            f.write('\n')

    with open(RenderOutput + r'/train/train_unlabel_{}.txt'.format(dataset_name), 'w') as f:
        for x in unlabel_list:
            f.write(x)
            f.write('\n')

def getLightTransList(numX, numY):
    lightList = []
    angleXList = []
    angleYList = []
#    np.random.seed(23333)
    for i in range(0, numX):
        for j in range(0, numY):
            angleY = np.random.uniform(0.0, 360.0)
            angleX = np.random.uniform(-30.0, 10.0)
            lightList.append('r,0,1,0,{}/r,1,0,0,{}/end'.format(angleY, angleX))
            angleXList.append(angleX)
            angleYList.append(angleY)
    return lightList, angleXList, angleYList


if __name__ == '__main__':
    gpuid = int(sys.argv[1])
    out_root = sys.argv[2]

    make_dir(out_root + r'/train_envlight')
    make_dir(out_root + r'/test_envlight')

    trainAlbedo = np.linspace(0.05, 1.0, albedoCnt)
    trainSpec = np.linspace(0.05, 1.0, specCnt)
    trainRoughness = np.exp(np.linspace(math.log(0.02), math.log(1.0), roughnessCnt))

    testAlbedo = np.linspace(0.1, 0.95, albedoCnt - 1)
    testSpec = np.linspace(0.1, 0.95, specCnt - 1)
    testRoughness = np.exp(np.linspace(math.log(0.03), math.log(0.87), roughnessCnt - 1))

    imageCnt = albedoCnt*specCnt*roughnessCnt*thetaCnt*phiCnt

    envMapFolder = params['envMapFolder']
    with open(envMapFolder + '/light.txt', 'r') as f:
        lightID = map(int, f.read().strip().split('\n'))
        lightID = list(np.array(lightID) - 1)

    np.random.seed(gpuid)
    OnlineRender = FastRenderEngine(gpuid)
    OnlineRender.SetGeometry('Sphere')
    OnlineRender.PreLoadAllLight(r'{}/light.txt'.format(envMapFolder))

    fovRadian = 60.0 / 180.0 * math.pi
    cameraDist = 1.5  / (math.tan(fovRadian / 2.0))
    OnlineRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 128, 128)
    OnlineRender.SetSampleCount(128, 1024)
    OnlineRender.SetRenderMode(0)

    albedoCnt = len(trainAlbedo)
    specCnt = len(trainSpec)
    roughnessCnt = len(trainRoughness)
    trainCube = np.zeros((albedoCnt, specCnt, roughnessCnt, 3))
    testCube = np.zeros((albedoCnt-1, specCnt-1, roughnessCnt-1, 3))

    print('Rendering Training data...\n')
    ftrain = open(out_root + r'/train_envlight/train_full.txt', 'w')
    for aid, a in enumerate(trainAlbedo):
        for sid, s in enumerate(trainSpec):
            for rid, r in enumerate(trainRoughness):
                lightMatrix = np.zeros((len(lightID), 9, 2))
                print('...{}_{}_{}\n'.format(aid, sid, rid))
            
                trainCube[aid,sid,rid] = [a,s,r]
                brdfFolder = out_root + r'/train_envlight/{}_{}_{}'.format(aid, sid, rid)
                make_dir(brdfFolder)
                                         
                OnlineRender.SetAlbedoValue([a, a, a])
                OnlineRender.SetSpecValue([s, s, s])
                OnlineRender.SetRoughnessValue(r)

                for lid, l in enumerate(lightID):
                    OnlineRender.SetEnvLightByID(l+1)
                    lightView, lightX, lightY = getLightTransList(3, 3)
                    for vid, v in enumerate(lightView):                    
                        OnlineRender.SetLightXform(lightX[vid], lightY[vid])
                        img = OnlineRender.Render()
                        save_pfm(brdfFolder + r'/{}_{}.pfm'.format(lid, vid), img)
                        ftrain.write('{}_{}_{}_{}_{}\n'.format(aid, sid, rid, lid, vid))
                        lightMatrix[lid, vid, 0] = lightX[vid]
                        lightMatrix[lid, vid, 1] = lightY[vid]

                np.savetxt(out_root + r'/train_envlight/lightMatrix_{}_{}_{}.txt'.format(aid, sid, rid), lightMatrix.flatten())

    ftrain.close()
    np.savetxt(out_root + r'/train_envlight/brdfcube.txt', trainCube.flatten())
    print('Done.\n')

    print('Rendering Test data...\n')
    ftest = open(out_root + r'/test_envlight/test_full.txt', 'w')
    for aid, a in enumerate(testAlbedo):
        for sid, s in enumerate(testSpec):
            for rid, r in enumerate(testRoughness):
                print('...{}_{}_{}\n'.format(aid, sid, rid))
                testCube[aid, sid, rid] = [a, s, r]
                brdfFolder = out_root + r'/test_envlight/{}_{}_{}'.format(aid, sid, rid)#+ offSetAlbedo[gpuid], sid + offSetSpec[gpuid], rid)
                make_dir(brdfFolder)
            
                OnlineRender.SetAlbedoValue([a, a, a])
                OnlineRender.SetSpecValue([s, s, s])
                OnlineRender.SetRoughnessValue(r)

                for lid, l in enumerate(lightID):
                    OnlineRender.SetEnvLightByID(l+1)#(envMapFolder + r'\{:04d}.pfm'.format(l+1), '')
                    img = OnlineRender.Render()
                    save_pfm(brdfFolder + r'/{}_{}.pfm'.format(lid, 0), img)
                    ftest.write('{}_{}_{}_{}_{}\n'.format(aid, sid, rid, lid, 0))
            
    ftest.close()
    np.savetxt(out_root + r'/test_envlight/brdfcube.txt', testCube.flatten())
