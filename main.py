import cv2, numpy
from js import Uint8Array
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

def clipBrightArea(img, threshold, topLevel = 0):
  height, width = numpy.shape(img)
  niti = (img > threshold) * 1

  top = topLevel
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

def concatPlayReport(imgs: list[numpy.ndarray]):
  diff = abs(imgs[0] - imgs[1])
  h, _, _ = numpy.shape(diff)
  top, bottom = clipBrightArea(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 128, int(h*0.3))
  skillReports = [img[top:bottom] for img in imgs]

  stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
  _, stitchedImg = stitcher.stitch(skillReports)

  _, original_w, _ = numpy.shape(imgs[0])
  _, stitched_w, _ = numpy.shape(stitchedImg)
  width = min(original_w, stitched_w)

  top = imgs[0][numpy.ix_([i for i in range(top)], [i for i in range(width)])]
  skillReport = stitchedImg[:, [i for i in range(width)]]

  completeImg = numpy.vstack([top, skillReport])
  return completeImg

async def showThumbnail(e):
  global imgA, imgB, imgC
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

  thumbnailImageA.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(imgA)}"
  thumbnailImageB.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(imgB)}"
  if len(fileList) == 3:
    thumbnailImageC.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(imgC)}"

async def main(e):
  global imgA, imgB, imgC

  fileList = fileSelector.files
  messageElement.textContent = "Processing ..."
  imgs = [imgA, imgB]
  if len(fileList) == 3:
    imgs.append(imgC)
  #if imgB == None:
  #  await showThumbnail(None)
  try:
    completeImg = concatPlayReport(imgs)
    resultElemet.src = f"data:image/jpeg;base64,{cv2ImageToBase64Text(completeImg)}"
  except:
    alertElement.textContent = "すみません。結合できませんでした。"
  imgA = None
  imgB = None
  imgC = None
  messageElement.textContent = ""

  anker.click()

fileSelector.onchange = showThumbnail
submitButton.onclick = main