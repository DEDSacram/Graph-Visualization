from multiprocessing.connection import wait
import this
from turtle import *
import random
import math
from xml.dom.minicompat import NodeList
import cv2
from cv2 import line
from cv2 import blur
import numpy as np
import os
import re
import time

import easyocr

class Node:

        # declare and initialize an instance variable
    def __init__(self):
        self.connectedto = []
        self.connectedtovalues = []
        self.connectedtonodes = []
    position= []
    size = []
    outercircle = []
    # connectedto = []
    value = ""

    def add(self,content):
        self.connectedto.append(content)
    def addnode(self,node):
        self.connectedtonodes.append(node)
    def addvalues(self,value):
        self.connectedtovalues.append(value)

class Edge:
    position= []
    size = []
    value = ""


r = 30
basedir = "data/images/"
images = ["Cool.png","pog.png","pepeppo.png","HAHAUDIETHANOSSNAP.png","sus.png","garbage.jpeg"] # to be changed
imagepath = basedir + "pepeppo.png"
prefix = "../"
def detectlines(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 60  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    img_erosion = cv2.erode(line_image, (kernel_size, kernel_size), iterations=1)
    edges = cv2.Canny(img_erosion, low_threshold, high_threshold)
    mask_image = np.copy(img) * 0
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_image, contours, -1, (0,255,0), 3)
    cv2.imshow("HoughCirlces",	mask_image)
    cv2.waitKey()
    return contours

def imageintoGraph(path):
    # os.system(f"cd .. &&  python detect.py --weights runs/train/graphs/weights/last.pt --img 640 --conf 0.6 --source "+path+" --hide-labels")
    # time.sleep(5)
    cords = [] # all detections

    nodelist = []
    edgelist = []

    reader = easyocr.Reader(['en'],gpu=True)

    linestopbot=[]

    lineedges = detectlines(prefix+path) # line Contours
    planets	= cv2.imread(prefix+path)
    original = planets.copy()
    gray_img=cv2.cvtColor(planets,	cv2.COLOR_BGR2GRAY)
    img	= cv2.medianBlur(gray_img,	1)
    circles	= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,param1=100,param2=30,minRadius=20,maxRadius=120)
    circles	= np.uint16(np.around(circles))

    with open("alldetections.txt") as f:
        lines = f.readlines()
        for x in lines:
            x = x.rstrip() 
            cords.append([int(float(i)) for i in x.split(',')])
  
    for cord in cords:
        node = Node()
        edge = Edge()
        width = (cord[2]-cord[0])
        height = (cord[1]-cord[3])
        centerx = cord[0] + int(width/2)
        centery = cord[1] - int(height/2)
        isin = False
        for i in circles[0]:
            if((math.pow((cord[0] - i[0]), 2) + math.pow((cord[1] - i[1]), 2)) < math.pow(i[2],2)):
                        isin = True
                        node.outercircle = [i[0],i[1],i[2]]
            cv2.circle(planets,(i[0],i[1]),i[2],(0,128,64),6)

      
     
        if isin:
            crop_img = original[cord[1]:cord[3], cord[0]:cord[2]]
            results = reader.recognize(crop_img)
          
            node.position = [cord[0],cord[1],cord[2],cord[3],centerx,centery]
            node.size = [width,height]
            node.value= results[0][1]
            nodelist.append(node)
            cv2.circle(planets,(centerx,centery),20,(0,128,0),3)
        else:
            crop_img = original[cord[1]+1:cord[3]+1, cord[0]+1:cord[2]+1]
            results = reader.recognize(crop_img)
            edge = Edge()
            edge.position = [cord[0],cord[1],cord[2],cord[3],centerx,centery]
            edge.size = [width,height]
            # edge.value = re.findall(r'\d+', results[0][1])
            edge.value = results[0][1]
            edgelist.append(edge)
            cv2.circle(planets,(centerx,centery),20,(0,0,255),3)
    


    cv2.drawContours(planets, lineedges,-1, (0,255,0), 3)
    for crt in lineedges:
        leftmost = tuple(crt[crt[:,:,0].argmin()][0])
        rightmost = tuple(crt[crt[:,:,0].argmax()][0])
        # topmost = tuple(crt[crt[:,:,1].argmin()][0])
        # bottommost = tuple(crt[crt[:,:,1].argmax()][0])
        # cv2.circle(planets,(topmost[0],topmost[1]),10,(0,128,64),6)
        # cv2.circle(planets,(bottommost[0],bottommost[1]),10,(0,256,64),6)
        center = [int((rightmost[0]+leftmost[0])/2),int((rightmost[1]+leftmost[1])/2)]
        cv2.circle(planets,(rightmost[0],rightmost[1]),4,(64,64,64),6)
        cv2.circle(planets,(leftmost[0],leftmost[1]),4,(0,0,256),6)
        cv2.circle(planets,(int((rightmost[0]+leftmost[0])/2),int((rightmost[1]+leftmost[1])/2)),4,(128,64,0),6)
        linestopbot.append([leftmost[0],leftmost[1],rightmost[0],rightmost[1]])

    for line in linestopbot:
        
        for node in nodelist:
            x1 = node.outercircle[0] - node.outercircle[2]-20   
            x2 = node.outercircle[0] + node.outercircle[2]+20
            y1 = node.outercircle[1] + node.outercircle[2]+20
            y2 = node.outercircle[1] - node.outercircle[2]-20
            if(((line[0]>x1 and line[0]<x2) and (line[1]>y2 and line[1]<y1)) or ((line[2]>x1 and line[2]<x2) and (line[3]>y2 and line[3]<y1))):
                cv2.rectangle(planets,(x1,y1),(x2,y2),(255,0,0),5)
                node.add(line)
    

    for node in nodelist:
        for node2 in nodelist:
            for connection in node2.connectedto:
                if connection in node.connectedto and node2 != node:
                    # node.addnode(node2)
                    node.addnode(node2)

    for node in nodelist:
        # print(node.value)
        # print(node.connectedto)
     
        for line in linestopbot:
            for edgy in edgelist:
          
                x1 = edgy.position[0]-50
                x2 = edgy.position[2]+50
                y1 = edgy.position[1]+50
                y2 = edgy.position[3]-50
                xlinecenter= int((line[0]+line[2])/2)
                ylinecenter=  int((line[1]+line[3])/2)
                if(((xlinecenter>x1 and xlinecenter<x2) and (ylinecenter>y2 and ylinecenter<y1)) and line in node.connectedto):
                    # print(edgy.value)
                    node.addvalues(edgy.value)
                    cv2.rectangle(planets,(x1,y1),(x2,y2),(255,128,64),3)
    
    graph = [ [ 123 for y in range( len(nodelist)) ]
             for x in range( len(nodelist)) ]
    # print(len(graph))
 

    #Temp fix
    nodevaluearr = []
    for node in nodelist:
        nodevaluearr.append(node.value)

  
  
    for x in range(len(nodelist)):
        # print(len(nodelist[x].connectedtonodes[x])-1)
        print(nodelist[x].value)
        print("Connected",nodelist[x].connectedtovalues)
        # print("NodeValue",nodelist[x].value)
        # print("Connected",nodelist[x].connectedtonodes)
        for y in range(len(nodelist[x].connectedtonodes)):
            # print(nodelist[x].connectedtonodes[y].value)
            # print(nodelist[x].connectedtovalues)
            # print(y)
            for z in range(len(nodevaluearr)):
                if(nodelist[x].connectedtonodes[y].value == nodevaluearr[z]):
                    # print(z)
                    if(len(nodelist[x].connectedtovalues) < y):
                        print("failed")
                    else:
                        nodelist[x].connectedtovalues[y]
                        graph[x][z] = nodelist[x].connectedtovalues[y]
                    # print(x)
                   
                  
             
          
    print(nodevaluearr)
  
   
    # graph[current][x] = xcon
    for r in graph:
        # print(r)
        for c in r:
            pass
    
    
  
    with open('paths.txt', 'w') as f:
        for node in graph:
            f.write(','.join(str(x) for x in node))
            f.write('\n')
    cv2.imshow("HoughCirlces",	planets)
    cv2.waitKey()
    # cv2.destroyAllWindows()

def textintoGraph(file):
    graph = []
    with open(file) as f:
        lines = f.readlines()
        for x in lines:
            x = x.rstrip() 
            graph.append([int(i) for i in x.split(',')])

        return graph

def graphintoText(graph):
    with open('dist.txt', 'w') as f:
        for node in graph:
            f.write(','.join(str(x) for x in node))
            f.write('\n')
graph = textintoGraph('paths.txt')
nodes = len(graph)

def floydWarshall(graph):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
    for k in range(nodes):
 

        for i in range(nodes):
 
        
            for j in range(nodes):
 
        
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j]
                                 )
    graphintoText(dist)
    return dist
WIDTH, HEIGHT = 1980, 1080
screen = Screen()
screen.setup(WIDTH + 4, HEIGHT + 8)  # fudge factors due to window borders & title bar
screen.setworldcoordinates(0, 0, WIDTH, HEIGHT)
screen.colormode(255)
speed(2)
points = []
def randomCords():
    x=random.randint(40, WIDTH-40)
    y=random.randint(40, HEIGHT-40)
    return [x,y]
def verify(cords,points):
    if not points:
        return True
    for i in range(len(points)):
        #Check collisions Spaces between
        if cords[0] >= points[i][0]-200 and cords[0] >= points[i][0]+200 and cords[1] >= points[i][1]-200 and cords[1] >= points[i][1]+200:
            return False
    return True
def GenerateCordsforNodes(cords,points):
    i = 0
    while i <=nodes:
        cords = randomCords()
        if verify(cords,points):
            points.append(randomCords())
        else:
            i-=1
        i+=1
    return points
def drawnodes(nodes,points):
    for i in range(nodes):
        penup()
        goto(points[i][0]-5,points[i][1]+r-16)
        pendown()
        write(i, font=("Arial", 16, "normal"))
        penup()
        goto(points[i][0],points[i][1])
        pendown()
        circle(r)
# PROBLEMS WHEN TOO CLOSE TO X,Y axis intersections of the circle. BECAUSE OF REVERSING SIDES LINE GOES THROUGH THE NODE 
def drawconnectionlines(points,graph,offset,add,distances):
    nodeint = 0
    i = 0
    for node in graph:
        if add:
            color((random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)))
        for dist in node:
            if add:
                if dist == distances[nodeint][i]:
                    i+=1
                    continue
            if dist != 0 and dist != 123:
                angle = math.atan2(points[i][1]+r - points[nodeint][1]+r, points[i][0] - points[nodeint][0])*(180/math.pi)
                xoncircle = r*math.sin(angle+offset)
                yoncircle = r*math.cos(angle+offset)
                xpoint1= 0
                xpoint2= 0
                ypoint1= 0
                ypoint2= 0
                if points[nodeint][0] > points[i][0]:
                    xpoint2 = points[nodeint][0]+xoncircle
                    xpoint1 = points[i][0]-xoncircle
                else:
                    xpoint2 = points[nodeint][0]-xoncircle
                    xpoint1 = points[i][0]+xoncircle
                if points[nodeint][1] > points[i][1]:
                    ypoint2=points[nodeint][1]+r-yoncircle
                    ypoint1=points[i][1]+r+yoncircle
                else: 
                    ypoint2=points[nodeint][1]+r+yoncircle
                    ypoint1=points[i][1]+r-yoncircle
                point2 = (xpoint2, ypoint2)
                point1 = (xpoint1, ypoint1)
                penup()
                goto(int((point2[0]+point1[0])/2),int((point2[1]+point1[1])/2)+offset)
                write(dist, font=("Arial", 8, "normal"))
                goto(point2)
                pendown()
                goto(point1[0],point1[1]+4)
                begin_fill()
                goto(point1)
                circle(4)
                end_fill()
            i+=1
        i=0
        nodeint+=1

imageintoGraph(imagepath)

floydWarshall(graph)
distances = textintoGraph('paths.txt')
points = GenerateCordsforNodes(randomCords(),points)
drawnodes(nodes,points)
drawconnectionlines(points,graph,0,False,distances)
# drawconnectionlines(points,distances,15,True,graph)
done()