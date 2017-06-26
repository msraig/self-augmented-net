import sys, os, math, random, glob, re, cStringIO
import numpy as np
import cv2

def make_dir(folder):
    if(os.path.exists(folder) == False):
        os.makedirs(folder)

def autoExposure(img):
    maxValue = np.max(img) + 1e-6
    return img / maxValue

def toHDR(img):
    img = img / 255.0
    img_out = img ** (2.2)
    return img_out.astype(np.float32)

def toLDR(img):
    img_out = img ** (1.0 / 2.2)
    img_out = np.minimum(255, img_out * 255)
    return img_out.astype(np.uint8)

#PFM load and write
def pfmFromBuffer(buffer, reverse = 1):
    sStream = cStringIO.StringIO(buffer)

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = sStream.readline().rstrip()
    color = (header == 'PF')

    width, height = map(int, sStream.readline().strip().split(' '))
    scale = float(sStream.readline().rstrip())
    endian = '<' if(scale < 0) else '>'
    scale = abs(scale)
    

    rawdata = np.fromstring(sStream.read(), endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    sStream.close()
    if(len(shape) == 3):
        return rawdata.reshape(shape).astype(np.float32)[:,:,::-1]
    else:
        return rawdata.reshape(shape).astype(np.float32)

def pfmToBuffer(img, reverse = 1):
    color = None
    sStream = cStringIO.StringIO()
    img = np.ascontiguousarray(img)
    if(img.dtype.name != 'float32'):
        img = img.astype(np.float32)


    color = True if (len(img.shape) == 3) else False

    if(reverse and color):
        img = img[:,:,::-1]

    sStream.write('PF\n' if color else 'Pf\n')
    sStream.write('%d %d\n' % (img.shape[1], img.shape[0]))
    
    endian = img.dtype.byteorder
    scale = 1.0
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    sStream.write('%f\n' % scale)
    sStream.write(img.tobytes())
    outBuffer = sStream.getvalue()
    sStream.close()
    return outBuffer 
 
def save_pfm(filepath, img, reverse = 1):
    color = None
    file = open(filepath, 'wb')
    if(img.dtype.name != 'float32'):
        img = img.astype(np.float32)

    color = True if (len(img.shape) == 3) else False

    if(reverse and color):
        img = img[:,:,::-1]

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (img.shape[1], img.shape[0]))
    
    endian = img.dtype.byteorder
    scale = 1.0
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)
    img.tofile(file)
    file.close()

def load_pfm(filepath, reverse = 1):
    file = open(filepath, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    color = (header == 'PF')

    width, height = map(int, file.readline().strip().split(' '))
    scale = float(file.readline().rstrip())
    endian = '<' if(scale < 0) else '>'
    scale = abs(scale)

    rawdata = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()

    if(color):  
        return rawdata.reshape(shape).astype(np.float32)[:,:,::-1]
    else:
        return rawdata.reshape(shape).astype(np.float32)


def load_and_clip(filepath, left, top, width, height, reverse = 1):
    name,ext = os.path.splitext(filepath)
    if(ext == '.pfm'):
        img = load_pfm(filepath)
    else:
        img = toHDR(cv2.imread(filepath))

    if(len(img.shape) == 3):
        return img[top:top+height,left:left+width,:]   
    else:
        return img[top:top+height,left:left+width] 





#utilties
def renormalize(normalMap):
    nOut = np.zeros(normalMap.shape)
    for i in range(0, normalMap.shape[0]):
        normal_1 = (2.0 * normalMap[i,:,:,:] - 1).transpose(1,2,0)
        length = np.linalg.norm(normal_1, axis = 2)
        normal_1 = normal_1 / np.dstack((length, length, length))
        nOut[i,:,:,:] = (0.5 * (normal_1 + 1)).transpose(2,0,1)
    return nOut

def normalizeAlbedoSpec(brdfbatch):
    for i in range(0, brdfbatch.shape[0]):
        factor = 0.5 / np.mean(np.linalg.norm(brdfbatch[i,0:3,:,:], axis = 0))
        brdfbatch[i,0:6,:,:] *= factor

    return brdfbatch

#normal map : [0,1] range!
def normalBatchToThetaPhiBatch(data):
    outBatch = np.zeros((data.shape[0], 2, data.shape[2], data.shape[3]))
    data_1 = data * 2 - 1.0
    outBatch[:,0,:,:] = np.arccos(data_1[:,0,:,:])
    outBatch[:,1,:,:] = np.arctan2(data_1[:,1,:,:], data_1[:,2,:,:])
    return outBatch


def thetaPhiBatchToNormalBatch(data):
    outBatch = np.zeros((data.shape[0], 3, data.shape[2], data.shape[3]))
    outBatch[:,0,:,:] = np.cos(data[:,0,:,:])
    outBatch[:,1,:,:] = np.sin(data[:,0,:,:]) * np.sin(data[:,1,:,:])
    outBatch[:,2,:,:] = np.sin(data[:,0,:,:]) * np.cos(data[:,1,:,:])

    outBatch = 0.5*(outBatch + 1.0)

    return outBatch
    #n*3*256*256

def findIndex(query, pList):
    out = np.zeros((len(query)))
    for id, p in enumerate(query):
        out[id] = np.argmin(np.abs(p - pList))

    return out

def listToStr(numlist):
    strList = ['{},'.format(x) for x in numlist]
    strList[-1] = strList[-1][:-1]
    return 

def dssim(img1, img2):
    img1_g = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2GRAY)
    return 0.5 * (1.0 - ssim(img1_g, img2_g))


def meanDownsample(img):
    out = 0.25*(img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2])
    return out

def genMipMap(texCube):
    #assume width = height and is 2^x
    nLevel = int(min(10, math.log(texCube.shape[1], 2))) + 1
    texMipMapList = []
    texMipMapList.append(texCube)
    for k in range(1, nLevel):
        prevCube = texMipMapList[k-1]
        if(len(prevCube.shape) == 3):
            newCube = np.ones((6, prevCube.shape[1] / 2, prevCube.shape[2] / 2))
        else:
            newCube = np.ones((6, prevCube.shape[1] / 2, prevCube.shape[2] / 2, 4))

        for f in range(0, 6):
            newCube[f] = meanDownsample(prevCube[f])#cv2.pyrDown(prevCube[f])

        texMipMapList.append(newCube.astype(np.float32))
    return texMipMapList

def getTexCube(crossImg):
    #TOBGRA since cuda only accept float4 textures
    if(len(crossImg.shape) == 2):
        crossImg = np.dstack((crossImg, crossImg, crossImg))
    faceRes = crossImg.shape[1] / 4
    width = height = faceRes
    if(len(crossImg.shape) == 3):
        texCube = np.ones((6, faceRes, faceRes, 4))
        texCube[0, :, :, 0:3] = crossImg[faceRes:faceRes+height, 2*faceRes:2*faceRes+width,:]
        texCube[1, :, :, 0:3] = crossImg[faceRes:faceRes+height, 0:width,:]
        texCube[3, :, :, 0:3] = crossImg[0:height, faceRes:faceRes+width,:]
        texCube[2, :, :, 0:3] = crossImg[2*faceRes:2*faceRes+height, faceRes:faceRes+width,:]
        texCube[4, :, :, 0:3] = crossImg[faceRes:faceRes+height, faceRes:faceRes+width,:]
        texCube[5, :, :, 0:3] = crossImg[faceRes:faceRes+height, 3*faceRes:3*faceRes+width,:]
    else:
        texCube = np.ones((6, faceRes, faceRes))
        texCube[0, :, :] = crossImg[faceRes:faceRes+height, 2*faceRes:2*faceRes+width]
        texCube[1, :, :] = crossImg[faceRes:faceRes+height, 0:width]
        texCube[3, :, :] = crossImg[0:height, faceRes:faceRes+width]
        texCube[2, :, :] = crossImg[2*faceRes:2*faceRes+height, faceRes:faceRes+width]
        texCube[4, :, :] = crossImg[faceRes:faceRes+height, faceRes:faceRes+width]
        texCube[5, :, :] = crossImg[faceRes:faceRes+height, 3*faceRes:3*faceRes+width]

    for i in range(0, 6):
        texCube[i, :, :] = texCube[i, ::-1, :]


    return np.ascontiguousarray(texCube, dtype=np.float32)

    


#DataLoader class
def checkVaild(root, mid, lid, vid, oid):
    imgpath = root + r'/m_{}/{}_{}_{}_{}_image.pfm'.format(mid, mid, lid, vid, oid)
    apath = root + r'/m_{}/gt_{}_albedo.pfm'.format(mid, oid)
    spath = root + r'/m_{}/gt_{}_specalbedo.pfm'.format(mid, oid)
    rpath = root + r'/m_{}/gt_{}_roughness.pfm'.format(mid, oid)

    if(os.path.exists(imgpath) and os.path.exists(apath) and os.path.exists(spath) and os.path.exists(rpath)):# and os.path.exists(npath)):
        return True
    else:
        return False

class RealDataLoaderSVBRDF(object):
    dataSize = 0
    rootPath = '' 
    
    dataList = []
    cursorPos = 0
    
    width = 256
    height = 256

    def __init__(self, rootPath, imgListFile):
        with open(rootPath + r'/{}'.format(imgListFile), 'r') as f:
            self.dataList = f.read().strip().split('\n')

        self.rootPath = rootPath        
        self.dataSize = len(self.dataList)
        
        self.cursorPos = 0
        self.width = 256
        self.height = 256
 
    def shuffle(self, seed = []):
        if(seed == []):
            np.random.shuffle(self.dataList)
        else:
            np.random.seed(seed)
            np.random.shuffle(self.dataList)

    def GetImg(self, idx):
        path = r'{}/{}'.format(self.rootPath, self.dataList[idx]).strip()  #for the FUCKING CRLF difference between WINDOWS and Linux
        img = toHDR(cv2.imread(path)).transpose(2,0,1)# / 255.0
        return img[np.newaxis, :, :, :]

    def GetImgWithName(self, idx):
        img = self.GetImg(idx)
        name = self.dataList[idx]
        return img, name

    def GetBatchWithName(self, start, n):
        dataBatch = np.zeros((n, 3, self.height, self.width))
        nameList = []

        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :],  name = self.GetImgWithName(idx)
            nameList.append(name)

        return dataBatch, nameList

    def GetBatch(self, start, n):
        dataBatch = np.zeros((n, 3, self.height, self.width))

        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :] = self.GetImg(idx)

        return dataBatch

    def GetNextBatch(self, n):
        dataBatch = self.GetBatch(self.cursorPos, n, unlabel)
        self.cursorPos = (self.cursorPos + n) % self.dataSize

        return dataBatch    

class DataLoaderSVBRDF(object):
    dataSize = 0
    rootPath = '' 
    
    dataList = []
    cursorPos = 0
    
    width = 256
    height = 256
    
    nBRDFChannal = 10
    
    rawwidth = 0
    rawheight = 0

    randomClip = False

    #hack
    clipPos = []
    
    ldr = False
    smallInput = False

    def __init__(self, rootPath, imgListFile, rawWidth = 256, rawHeight = 256, randomClip = False):
        self.mList = []
        self.lList = []
        self.vList = []
        self.oList = []
        self.clipPosList = []
        self.specRoughnessList = {}

        with open(rootPath + r'/{}'.format(imgListFile), 'r') as f:
            self.dataList = f.read().strip().split('\n')

        with open(rootPath + r'/{}'.format(imgListFile), 'r') as f:
            self.fullDataList = f.read().strip().split('\n')

        with open(rootPath + r'/{}'.format('specroughness.txt'), 'r') as f:
            rawList = f.read().strip().split('\n')
            for t in rawList:
                mid = int(t.split(',')[0])
                spec = float(t.split(',')[1])
                roughness = float(t.split(',')[2])
                self.specRoughnessList[mid] = (spec, roughness)
        if(os.path.exists(rootPath + r'/{}'.format('translatepos.txt'))):
            self.clipPosList = pickle.load(open(rootPath + r'/{}'.format('translatepos.dat'), 'rb'))

        self.rootPath = rootPath        

        self.rawwidth = rawWidth
        self.rawheight = rawHeight   
        self.randomClip = randomClip     

        self.buildMLVOList()
        self.dataSize = len(self.dataList)
        self.fulldataSize = len(self.dataList)
        
        self.cursorPos = 0
        self.width = 256
        self.height = 256
            
        self.nBRDFChannal = 10
 
    def shuffle(self, seed = []):
        if(seed == []):
            np.random.shuffle(self.dataList)
        else:
            np.random.seed(seed)
            np.random.shuffle(self.dataList)
      

    def checkList(self):
        newList = []
        for item in self.dataList:
            m, l, v, o = map(int, item.split('_'))
            if(checkVaild(self.rootPath, m, l, v, o)):
                newList.append(item)
        
        self.dataList = newList

    def buildMLVOList(self):
        mList = set()
        lList = set()
        vList = set()
        oList = set()
    
        for data in self.dataList:
            m, l, v, o = map(int, data.split('_'))
            mList.add(m)
            lList.add(l)
            vList.add(v)
            oList.add(o)

        self.mList = sorted(list(mList))
        self.lList = sorted(list(lList))
        self.vList = sorted(list(vList))   
        self.oList = sorted(list(oList))  

    def buildSubDataset(self, mid_list, lid_list, vid_list, oid_list):
        dataList = []
        for m in mid_list:
            for l in lid_list:
                for v in vid_list:
                    dataList.append('{}_{}_{}_{}'.format(m, l ,v, o))

        self.mList = sorted(mid_list)
        self.lList = sorted(lid_list)
        self.vList = sorted(vid_list)
        self.oList = sorted(oid_list)

        self.dataList = dataList
        self.dataSize = len(self.dataList)    
  
    def GetItem(self, idx):
        mid, lid, vid, oid = map(int, self.dataList[idx].split('_'))
        img, brdf = self.GetItemByID(mid, lid, vid, oid)
        return img, brdf

    def GetImgOnly(self, idx):
        mid, lid, vid, oid = map(int, self.dataList[idx].split('_'))
        img = self.GetImgOnlyByID(mid, lid, vid, oid)

        return img            

    def GetAlbedoAndNormal(self, idx):
        mid, lid, vid, oid = map(int, self.dataList[idx].split('_'))
        brdf = self.GetAlbedoAndNormalOnlyByID(mid, lid, vid, oid)
        return brdf
  

    def GetAlbedoAndNormalWithName(self, idx,):
        brdf = self.GetAlbedoAndNormal(idx)
        name = map(int, self.dataList[idx].split('_')) 
        return brdf, name

    def GetItemWithName(self, idx):
        img, brdf = self.GetItem(idx)
        name = map(int, self.dataList[idx].split('_'))

        return img, brdf, name

    def GetBatchWithName(self, start, n):
        dataBatch = np.zeros((n, 3, self.height, self.width))
        brdfBatch = np.zeros((n, self.nBRDFChannal, self.height, self.width))
        nameList = []

        tmpSize = self.dataSize

        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :], brdfBatch[i, :, :, :], name = self.GetItemWithName(idx)
            nameList.append(name)

        return dataBatch, brdfBatch, nameList

    def GetAlbedoAndNormalBatchWithName(self, start, n):
        brdfBatch = np.zeros((n, self.nBRDFChannal, self.height, self.width))
        nameList = []

        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            brdfBatch[i, :, :, :], name = self.GetAlbedoAndNormalWithName(idx)
            nameList.append(name)

        return brdfBatch, nameList

    def GetImgOnlyBatch(self, start, n):
        dataBatch = np.zeros((n, 3, self.height, self.width))

        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :] = self.GetImgOnly(idx)

        return dataBatch  

    def GetBatch(self, start, n):
        dataBatch = np.zeros((n, 3, self.height, self.width))
        brdfBatch = np.zeros((n, self.nBRDFChannal, self.height, self.width))

        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :], brdfBatch[i, :, :, :] = self.GetItem(idx)

        return dataBatch, brdfBatch

    def GetImgOnlyNextBatch(self, n):
        dataBatch = self.GetImgOnlyBatch(self.cursorPos, n)
        self.cursorPos = (self.cursorPos + n) % self.dataSize

        return dataBatch

    def GetNextBatch(self, n):
        dataBatch, brdfBatch = self.GetBatch(self.cursorPos, n)
        self.cursorPos = (self.cursorPos + n) % self.dataSize

        return dataBatch, brdfBatch

    def GetAlbedoAndNormalOnlyByID(self, mid, lid, vid, oid):       ##random give spec and roughness
        brdf = np.zeros((1, self.nBRDFChannal, self.height, self.width))
        specValue = np.random.uniform(0.005, 0.15)
        roughnessValue = np.random.uniform(0.005, 0.15)
        if(self.randomClip):
            if(self.clipPosList != []):
                clip_left, clip_top = clipPos['{}_{}_{}_{}'.format(mid,lid,vid,oid)]
            else:
                clip_left = np.random.randint(0, self.rawwidth - 1 - self.width)
                clip_top = np.random.randint(0, self.rawheight - 1 - self.height)
        else:
            clip_left = self.rawwidth / 2 - self.width / 2
            clip_top = self.rawheight / 2 - self.height / 2

        self.clipPos = [clip_left, clip_top]
        brdf[0,0:3,:,:] = load_and_clip(self.rootPath + r'/m_{}/gt_{}_albedo.pfm'.format(mid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        brdf[0,3:6,:,:] = specValue
        brdf[0,6,:,:] = roughnessValue
        brdf[0,7:10,:,:] = load_and_clip(self.rootPath + r'/m_{}/gt_{}_normal.pfm'.format(mid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        brdf[0,7,:,:][np.isnan(brdf[0,7,:,:])] = 1.0
        brdf[0,8,:,:][np.isnan(brdf[0,8,:,:])] = 0.0
        brdf[0,9,:,:][np.isnan(brdf[0,9,:,:])] = 0.0
                
        return brdf           

    def GetImgOnlyByID(self, mid, lid, vid, oid):
        if(self.randomClip):
            if(self.clipPosList != []):
                clip_left, clip_top = clipPos['{}_{}_{}_{}'.format(mid,lid,vid,oid)]
            else:
                clip_left = np.random.randint(0, self.rawwidth - 1 - self.width)
                clip_top = np.random.randint(0, self.rawheight - 1 - self.height)
        else:
            clip_left = self.rawwidth / 2 - self.width / 2
            clip_top = self.rawheight / 2 - self.height / 2
        self.clipPos = [clip_left, clip_top]
        if(self.ldr):
            img = load_and_clip(self.rootPath + r'/m_{}/{}_{}_{}_{}_image.jpg'.format(mid, mid, lid, vid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        else:
            img = load_and_clip(self.rootPath + r'/m_{}/{}_{}_{}_{}_image.pfm'.format(mid, mid, lid, vid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        return img[np.newaxis, :, :, :]

    def GetItemByID(self, mid, lid, vid, oid):
        brdf = np.zeros((1, self.nBRDFChannal, self.height, self.width))

        if(self.randomClip):
            if(self.clipPosList != []):
                clip_left, clip_top = clipPos['{}_{}_{}_{}'.format(mid,lid,vid,oid)]
            else:
                clip_left = np.random.randint(0, self.rawwidth - 1 - self.width)
                clip_top = np.random.randint(0, self.rawheight - 1 - self.height)            
        else:
            clip_left = self.rawwidth / 2 - self.width / 2
            clip_top = self.rawheight / 2 - self.height / 2
        self.clipPos = [clip_left, clip_top]

        if(os.path.exists(self.rootPath + r'/m_{}/{}_{}_{}_{}_image.pfm'.format(mid, mid, lid, vid, oid)) == False or 
           os.path.exists(self.rootPath + r'/m_{}/{}_{}_{}_{}_image.jpg'.format(mid, mid, lid, vid, oid)) == False):
            img = -100*np.ones((3,self.height,self.width))
        else:
            if(self.ldr):
                img = load_and_clip(self.rootPath + r'/m_{}/{}_{}_{}_{}_image.jpg'.format(mid, mid, lid, vid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
            else:
                img = load_and_clip(self.rootPath + r'/m_{}/{}_{}_{}_{}_image.pfm'.format(mid, mid, lid, vid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        brdf[0,0:3,:,:] = load_and_clip(self.rootPath + r'/m_{}/gt_{}_albedo.pfm'.format(mid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        brdf[0,3:6,:,:] = self.specRoughnessList[mid][0]
        brdf[0,6,:,:] = self.specRoughnessList[mid][1]
        brdf[0,7:10,:,:] = load_and_clip(self.rootPath + r'/m_{}/gt_{}_normal.pfm'.format(mid, oid), clip_left, clip_top, self.width, self.height).transpose((2,0,1))
        brdf[0,7,:,:][np.isnan(brdf[0,7,:,:])] = 1.0
        brdf[0,8,:,:][np.isnan(brdf[0,8,:,:])] = 0.0
        brdf[0,9,:,:][np.isnan(brdf[0,9,:,:])] = 0.0
                
        return img[np.newaxis, :, :, :], brdf      
  
class DataLoaderSimple(object):
    fulldataSize = 0
    dataSize = 0

    brdfCube = []
    fulldataList = []
    dataList = []
    rootPath = ''

    cursorPos = 0
    width = 0
    height = 0
    
    aCnt = 0
    sCnt = 0
    rCnt = 0

    tCnt = 15
    pCnt = 15

    def __init__(self, rootPath, imgListFile, aCnt, sCnt, rCnt, width, height):
        #load brdf cube
        if(os.path.exists(rootPath + r'/cubeResolution.txt')):
            with open(rootPath + r'/cubeResolution.txt', 'r') as f:
                aCnt, sCnt, rCnt = map(int, f.read().strip().split(','))

        print(aCnt, sCnt, rCnt)
        self.brdfCube = np.loadtxt(rootPath + r'/brdfcube.txt').reshape((aCnt, sCnt, rCnt, 3))
        self.brdfCube[:,:,:,2] = np.log(self.brdfCube[:,:,:,2])
        self.brdfCube_Masked = np.ma.array(self.brdfCube)
        self.brdfCube_Masked.mask = False

        with open(rootPath + r'/{}'.format(imgListFile), 'r') as f:
            self.dataList = f.read().strip().split('\n')

        with open(rootPath + r'/{}'.format(imgListFile), 'r') as f:
            self.fulldataList = f.read().strip().split('\n')


        self.dataSize = len(self.dataList)
        self.fulldataSize = len(self.dataList)
        self.rootPath = rootPath
        self.cursorPos = 0
        self.width = 128
        self.height = 128
        self.aCnt = aCnt
        self.sCnt = sCnt
        self.rCnt = rCnt
        self.lightCount = 15*15


    def buildSubDataset_2(self, brdf_light_list, inverse = 0):
        dataList = []
        self.lightCount = -1
        self.brdfCube_Masked.mask = True
        
        for brdf_light in brdf_light_list:
            a, s, r, l, v = brdf_light
            self.brdfCube_Masked.mask[a, s, r, :] = False
            dataList.append('{}_{}_{}_{}_{}'.format(a, s, r, l, v))         

        if(inverse):
            self.dataList = list(set(self.fulldataList) - set(dataList))
            self.dataSize = len(self.dataList)
        else:
            self.dataList = dataList
            self.dataSize = len(self.dataList)

    def buildSubDataset_1(self, brdf_list, tid_list, pid_list, inverse = 0):
        dataList = []

        self.lightCount = len(tid_list) * len(pid_list)
        self.brdfCube_Masked.mask = True
        
        for brdf in brdf_list:
            a, s, r = brdf
            self.brdfCube_Masked.mask[a, s, r, :] = False
            for t in tid_list:
                for p in pid_list:
                    dataList.append('{}_{}_{}_{}_{}'.format(a, s, r, t, p))         

        if(inverse):
            self.dataList = list(set(self.fulldataList) - set(dataList))
            self.dataSize = len(self.dataList)
        else:
            self.dataList = dataList
            self.dataSize = len(self.dataList)
        

    def buildSubDataset(self, aid_list, sid_list, rid_list, tid_list, pid_list, inverse = 0):
        dataList = []
        a_min, a_max = min(aid_list), max(aid_list)
        s_min, s_max = min(sid_list), max(sid_list)
        r_min, r_max = min(rid_list), max(rid_list)
        if(a_max >= self.brdfCube.shape[0]):
            print('a error')
            aid_list = range(a_min, self.brdfCube.shape[0])
        if(s_max >= self.brdfCube.shape[1]):
            print('s error')
            sid_list = range(s_min, self.brdfCube.shape[1])
        if(r_max >= self.brdfCube.shape[2]):
            rint('r error')
            rid_list = range(r_min, self.brdfCube.shape[2])

        self.lightCount = len(tid_list) * len(pid_list)
        self.brdfCube_Masked.mask = True
        for a in aid_list:
            for s in sid_list:
                for r in rid_list:
                    self.brdfCube_Masked.mask[a, s, r, :] = False
                    for t in tid_list:
                        for p in pid_list:
                            dataList.append('{}_{}_{}_{}_{}'.format(a, s, r, t, p))

        if(inverse):
            self.dataList = list(set(self.fulldataList) - set(dataList))
            self.dataSize = len(self.dataList)
        else:
            self.dataList = dataList
            self.dataSize = len(self.dataList)


    def normalizeDataSet(self):
        self.amean, self.astd = np.ma.mean(self.brdfCube_Masked[:,:,:,0]), np.ma.std(self.brdfCube_Masked[:,:,:,0])
        self.smean, self.sstd = np.ma.mean(self.brdfCube_Masked[:,:,:,0]), np.ma.std(self.brdfCube_Masked[:,:,:,0])
        self.rmean, self.rstd = np.ma.mean(self.brdfCube_Masked[:,:,:,0]), np.ma.std(self.brdfCube_Masked[:,:,:,0])

        self.astd = 1 if self.astd == 0 else self.astd
        self.sstd = 1 if self.sstd == 0 else self.sstd
        self.rstd = 1 if self.rstd == 0 else self.rstd
        
        self.brdfCube_Masked[:,:,:,0] = (self.brdfCube_Masked[:,:,:,0] - self.amean) / self.astd   
        self.brdfCube_Masked[:,:,:,1] = (self.brdfCube_Masked[:,:,:,1] - self.smean) / self.sstd 
        self.brdfCube_Masked[:,:,:,2] = (self.brdfCube_Masked[:,:,:,2] - self.rmean) / self.rstd         

        self.brdfCube[:,:,:,0] = (self.brdfCube[:,:,:,0] - self.amean) / self.astd   
        self.brdfCube[:,:,:,1] = (self.brdfCube[:,:,:,1] - self.smean) / self.sstd 
        self.brdfCube[:,:,:,2] = (self.brdfCube[:,:,:,2] - self.rmean) / self.rstd         

    def shuffle(self, seed = []):
        if(seed == []):
            np.random.shuffle(self.dataList)
        else:
            np.random.seed(seed)
            np.random.shuffle(self.dataList)


    def GetItem(self, idx, color = False):
        aid, sid, rid, tid, pid = map(int, self.dataList[idx].split('_'))
        return self.GetItemByID(aid, sid, rid, tid, pid, color)#img, brdf

    def GetItemByID(self, aid, sid, rid, tid, pid, color = False):
        if(color):
            img = load_pfm(self.rootPath + r'/{}_{}_{}/{}_{}.pfm'.format(aid, sid, rid, tid, pid)).transpose((2,0,1))
        else:
            img = load_pfm(self.rootPath + r'/{}_{}_{}/{}_{}.pfm'.format(aid, sid, rid, tid, pid))
            if(len(img.shape) == 3):
                img = img[:,:,0]
            img = img[np.newaxis,:,:]
        brdf = self.brdfCube[aid, sid, rid].reshape((1,3,1,1))#np.array(map(float, self.gtlist[idx].strip().split(','))).reshape((1, 3, 1, 1))
        return img, brdf         

    def GetItemWithName(self, idx, color = False):
        img, brdf = self.GetItem(idx, color)
        name = map(int, self.dataList[idx].split('_'))

        return img, brdf, name

    def GetBatchWithName(self, start, n, color = False):
        if(color):
            dataBatch = np.zeros((n, 3, self.height, self.width)) 
        else:
            dataBatch = np.zeros((n, 1, self.height, self.width))
        brdfBatch = np.zeros((n, 3, 1, 1))
        nameList = []
        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :], brdfBatch[i, :, :, :], name = self.GetItemWithName(idx, color)
            nameList.append(name)

        return dataBatch, brdfBatch, nameList

    def GetBatch(self, start, n, color = False):
        if(color):
            dataBatch = np.zeros((n, 3, self.height, self.width)) 
        else:
            dataBatch = np.zeros((n, 1, self.height, self.width))
        brdfBatch = np.zeros((n, 3, 1, 1))
        tmpSize = self.dataSize
        for i in range(0, n):
            idx = (start + i) % tmpSize
            dataBatch[i, :, :, :], brdfBatch[i, :, :, :] = self.GetItem(idx, color)

        return dataBatch, brdfBatch

    def GetNextBatch(self, n, color = False):
        dataBatch, brdfBatch = self.GetBatch(self.cursorPos, n, color)
        self.cursorPos = (self.cursorPos + n) % self.dataSize

        return dataBatch, brdfBatch