import cv2, numpy
from js import Uint8Array, window
from pyscript import document
import base64

alertElement = document.querySelector("#alert")
messageElement = document.querySelector("#message")
resultElemet = document.querySelector("#result")
fileSelector = document.querySelector("#fileselector")
thumbnailImageA = document.querySelector('#img_a')
thumbnailImageB = document.querySelector('#img_b')
thumbnailImageC = document.querySelector('#img_c')
submitButton = document.querySelector("#submit")
anker = document.querySelector("#anker")

preprocessed = False

imgA = None
imgB = None
imgC = None

async def readImageAsNdarray(image):
  arrayBuffer = Uint8Array.new(await image.arrayBuffer())
  return numpy.asarray(bytearray(arrayBuffer), dtype=numpy.uint8)

async def readFileAsCV2Image(file):
  return cv2.imdecode(await readImageAsNdarray(file), 1)

def cv2ImageToBase64Text(img):
  _, encoded = cv2.imencode(".jpg", img)
  return base64.b64encode(encoded).decode("ascii")

def clipBrightArea(img, threshold):
    height, width = numpy.shape(img)
    niti = (img > threshold) * 1

    top = 0

    while(niti[top].sum() < width/2 and top < height):
        top += 1

    bottom = top

    while(niti[bottom].sum() > 1 and bottom < height):
        bottom += 1
    
    return top, bottom

def clipImage(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  top, bottom = clipBrightArea(gray, 128)
  left, right = clipBrightArea(gray.T, 128)
  return img[top:bottom, left:right]

def stitchingImg3(imgA, imgB, imgC):  
  diff = abs(imgA - imgB)
  diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  
  bottom = len(diff) - 1
  
  while(diff[bottom].sum() == 0 and bottom > 0):
      bottom -= 1
  
  top = bottom
  
  while(diff[top].sum() != 0 and top > 0):
      top -= 1
  
  clipedImgA = imgA[top:bottom]
  clipedImgB = imgB[top:bottom]
  clipedImgC = imgC[top:bottom]

  stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
  status, stitchedImg = stitcher.stitch([clipedImgA, clipedImgB, clipedImgC])
  completeImg = numpy.vstack([imgA[:top], stitchedImg, imgA[bottom+1:-1]])
  
  return completeImg

def stitchingImg2(imgA, imgB):  
  diff = abs(imgA - imgB)
  diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  
  bottom = len(diff) - 1
  
  while(diff[bottom].sum() == 0 and bottom > 0):
      bottom -= 1
  
  top = bottom
  
  while(diff[top].sum() != 0 and top > 0):
      top -= 1
  
  clipedImgA = imgA[top:bottom]
  clipedImgB = imgB[top:bottom]

  stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
  status, stitchedImg = stitcher.stitch([clipedImgA, clipedImgB])
  completeImg = numpy.vstack([imgA[:top], stitchedImg, imgA[bottom+1:-1]])
  
  return completeImg

async def showThumbnail(e):
  global imgA, imgB, imgC, preprocessed
  alertElement.textContent = ""
  fileList = fileSelector.files

  thumbnailImageA.src = ''
  thumbnailImageB.src = ''
  thumbnailImageC.src = ''

  if not (len(fileList) == 2 or len(fileList) == 3):
    alertElement.textContent = "画像を2枚または3枚選択してください"
    return

  imgA = await readFileAsCV2Image(fileList.item(0))
  imgB = await readFileAsCV2Image(fileList.item(1))
  if len(fileList) == 3:
    imgC = await readFileAsCV2Image(fileList.item(2))
  
  imgA = clipImage(imgA)
  imgB = clipImage(imgB)
  if len(fileList) == 3:
    imgC = clipImage(imgC)

  preprocessed = True

  thumbnailImageA.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(imgA)}"
  thumbnailImageB.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(imgB)}"
  if len(fileList) == 3:
    thumbnailImageC.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(imgC)}"

async def main(e):
  global imgA, imgB, imgC, preprocessed

  fileList = fileSelector.files

  if not preprocessed:
    messageElement.textContent = "Wating Preprocess ..."
    showThumbnail()
  
  messageElement.textContent = "Processing ..."
  completeImg = None
  if len(fileList) == 2:
    completeImg = stitchingImg2(imgA, imgB)
  if len(fileList) == 3:
    completeImg = stitchingImg3(imgA, imgB, imgC)
  resultElemet.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(completeImg)}"
  imgA = None
  imgB = None
  imgC = None
  messageElement.textContent = ""
  preprocessed = False

  anker.click()

fileSelector.onchange = showThumbnail
submitButton.onclick = main