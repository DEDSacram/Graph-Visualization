from easyocr import Reader
import easyocr
import cv2
import time
import re

basedir = "data/images/"
prefix = "../"
imagepath = prefix+basedir + "pepeppo.png"
image = cv2.imread(imagepath)
image2 = cv2.imread('test2.png')
# OCR the input image using EasyOCR

reader = Reader(['en'],gpu=True)
t0 = time.time()
results = reader.recognize(image)
print(reader.recognize(image))

print(re.findall(r'\d+', reader.recognize(image2)[0][1]))
t1 = time.time()
x = 200
x2 = 250
y = 200
y2 = 220
h= 100
w=50
crop_img = image[y:y2, x:x2]
print(re.findall(r'\d+', reader.recognize(image2)[0][1]))
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
total = t1-t0
print(total)
