from multiprocessing.connection import wait
from turtle import *
import random
import math
import cv2
from cv2 import line
from cv2 import blur
import numpy as np
import os
import re
import time

import easyocr


##NEEDED CLASSES
class Node:
       
    def __init__(self):
        self.connectedto = []
        self.connectedtonodes = []
        self.belongsto=[]
    position= []
    size = []
    outercircle = []
    value = ""
    def addvaluetoline(self,line):
        self.belongsto.append(line)
    def add(self,content):
        self.connectedto.append(content)
    def addnode(self,node):
        self.connectedtonodes.append(node)

class Edge:
    position= []
    size = []
    value = ""
##NEEDED CLASSES


##circle diameter,base directory, available images in that directory, prefix from this directory to yolov5
r = 30
basedir = "data/images/"
images = [] # to be changed
imagepath = basedir + "graf.png"
prefix = "../"
##circle diameter,base directory, available images in that directory, prefix from this directory to yolov5

## Function to detect lines between nodes, so we know which edge value belongs to what
def detectlines(image,edge,circles):

    ## Reading from given imagepath, base configuration for canny and hough lines
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    img_not = cv2.bitwise_not(blackAndWhiteImage)
    cv2.imshow("HoughCirlces",	img_not)
    cv2.waitKey()
    
    for i in circles[0]:
         cv2.circle(img_not,(i[0],i[1]),i[2]+10,(0,0,0),-1)
    for x in edge:
        cv2.rectangle(img_not,(x.position[0],x.position[1]),(x.position[2],x.position[3]+5),(0,0,0),-1)
        print(x.position)


 


 
    dilation = cv2.dilate(img_not,kernel,iterations = 1)
    cv2.imshow("HoughCirlces",	dilation)
    cv2.waitKey()
 
    # edges = cv2.Canny(dilation, low_threshold, high_threshold)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ##Prepairing lines on black background for detection of contours and returning them, drawing contours on another black background (not needed)
    return contours






def imageintoGraph(path):
    ## RUNNING AI, time.sleep introduced for CUDA out of memory error
    os.system(f"cd .. &&  python detect.py --weights Graph-Visualization/weights/last.pt --img 640 --conf 0.6 --source "+path+" --hide-labels")
    time.sleep(2)
    ## RUNNING AI, time.sleep introduced for CUDA out of memory error

    ## Initializing arrays,OCR and prepairing image for detection of circles (nodes)
    cords = [] # all detections

    nodelist = [] # Containing node classes
    edgelist = [] # containing edge classes

    reader = easyocr.Reader(['en'],gpu=True)

    linestopbot=[] #

    # lineedges = detectlines(prefix+path) # line Contours
    planets	= cv2.imread(prefix+path)
    original = planets.copy()
    gray_img=cv2.cvtColor(planets,	cv2.COLOR_BGR2GRAY)
    img	= cv2.medianBlur(gray_img,	1)
    circles	= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,param1=100,param2=30,minRadius=20,maxRadius=120)
    circles	= np.uint16(np.around(circles))
    ## Initializing arrays,OCR and prepairing image for detection of circles (nodes)


    ## READING FROM AI OUTPUT
    with open("alldetections.txt") as f:
        lines = f.readlines()
        for x in lines:
            x = x.rstrip() 
            cords.append([int(float(i)) for i in x.split(',')])
    ## READING FROM AI OUTPUT


    ## Going through all detected values by AI,If value is within the detected circle it is a node if not its an edge then reading values by of detected rectangle by OCR
    for cord in cords:
        node = Node()
        edge = Edge()
        width = abs(cord[2]-cord[0])
        height = abs(cord[1]-cord[3])
        centerx = cord[0] + int(width/2)
        centery = cord[1] - int(height/2)
        isin = False
        for i in circles[0]:
            if((math.pow((cord[0] - i[0]), 2) + math.pow((cord[1] - i[1]), 2)) < math.pow(i[2],2)):
                        isin = True
                        node.outercircle = [i[0],i[1],i[2]]
          

      
     
        if isin:
            crop_img = original[cord[1]:cord[3], cord[0]:cord[2]]
            results = reader.recognize(crop_img)
          
            node.position = [cord[0],cord[1],cord[2],cord[3],centerx,centery]
            node.size = [width,height]
            node.value= results[0][1]
            nodelist.append(node)
          
        else:
            crop_img = original[cord[1]+1:cord[3]+1, cord[0]+1:cord[2]+1]
            results = reader.recognize(crop_img)
            edge = Edge()
            edge.position = [cord[0],cord[1],cord[2],cord[3],centerx,centery]
            edge.size = [width,height]
            # edge.value = re.findall(r'\d+', results[0][1])
            edge.value = results[0][1]
            edgelist.append(edge)
      
    ## Going through all detected values by AI,If value is within the detected circle it is a node if not its an edge then reading values by of detected rectangle by OCR



    lineedges = detectlines(prefix+path,edgelist,circles) # line Contours
    # cv2.drawContours(planets, lineedges,-1, (0,255,0), 3) # Not sure why I have this here




    ## going through contours getting the closest 2 points of the line to node
    for crt in lineedges:
        leftmost = tuple(crt[crt[:,:,0].argmin()][0])
        rightmost = tuple(crt[crt[:,:,0].argmax()][0])
        # topmost = tuple(crt[crt[:,:,1].argmin()][0])
        # bottommost = tuple(crt[crt[:,:,1].argmax()][0])
        # cv2.circle(planets,(topmost[0],topmost[1]),10,(0,128,64),6)
        # cv2.circle(planets,(bottommost[0],bottommost[1]),10,(0,256,64),6)
        cv2.circle(planets,(rightmost[0],rightmost[1]),4,(64,64,64),6)
        cv2.circle(planets,(leftmost[0],leftmost[1]),4,(0,0,256),6)
        cv2.circle(planets,(int((rightmost[0]+leftmost[0])/2),int((rightmost[1]+leftmost[1])/2)),4,(128,64,0),6)
        linestopbot.append([leftmost[0],leftmost[1],rightmost[0],rightmost[1]])
    ## going through contours getting the closest 2 points of the line to node

    ## closest point of line to node is or isnt within the given box
    for line in linestopbot:
        
        for node in nodelist:
            x1 = node.outercircle[0] - node.outercircle[2]-30   
            x2 = node.outercircle[0] + node.outercircle[2]+30
            y1 = node.outercircle[1] + node.outercircle[2]+30
            y2 = node.outercircle[1] - node.outercircle[2]-30
            if(((line[0]>x1 and line[0]<x2) and (line[1]>y2 and line[1]<y1)) or ((line[2]>x1 and line[2]<x2) and (line[3]>y2 and line[3]<y1))):
                cv2.rectangle(planets,(x1,y1),(x2,y2),(255,0,0),5)
                node.add(line)

    ## closest point of line to node is or isnt within the given box

    ##adding connected node class to node class
    for node in nodelist:
        for node2 in nodelist:
            for connection in node2.connectedto:
                if connection in node.connectedto and node2 != node:
                    node.addnode(node2)
    ##adding connected node class to node class

    ## if edge value is within the given box created at middle of the line add value
    for node in nodelist:
        for line in linestopbot:
            for edgy in edgelist:
                x1 = edgy.position[0]-70
                x2 = edgy.position[2]+70
                y1 = edgy.position[1]+70
                y2 = edgy.position[3]-70
                xlinecenter= int((line[0]+line[2])/2)
                ylinecenter=  int((line[1]+line[3])/2)
                if(((xlinecenter>x1 and xlinecenter<x2) and (ylinecenter>y2 and ylinecenter<y1)) and line in node.connectedto):
                    node.addvaluetoline([node.value,line,edgy.value])
                    cv2.rectangle(planets,(x1,y1),(x2,y2),(255,128,64),3)
     ## if edge value is within the given box created at middle of the line add value

 

    
    ## CREATING INFINITY GRAPH
    graph = [ [ 123 for y in range( len(nodelist)) ]
             for x in range( len(nodelist)) ]
    ## CREATING INFINITY GRAPH
 

    ## CHECKING ORDER OF NODES IN GRAPH
    nodevaluearr = []
    for node in nodelist:
        nodevaluearr.append(node.value)
    ## CHECKING ORDER OF NODES IN GRAPH


    ## WHICH NODE IS CONNECTED TO WHICH INCLUDING VALUE OF EDGE BETWEEN THEM
    for x in range(len(nodelist)):
     
        print("New",nodelist[x].belongsto)
        for u in nodelist[x].belongsto:
                 for z in range(len(nodevaluearr)):
                          for i in nodelist[z].belongsto:
                              if(u[1] == i[1] and u[0]!=i[0]):
                                  print(u[0],i[0],i[2])
                                  print(x,z)
                                  graph[x][z] = i[2]
                                  
        print("Next")
    ## WHICH NODE IS CONNECTED TO WHICH INCLUDING VALUE OF EDGE BETWEEN THEM
    print(nodevaluearr)   
    
    ## WRITING ALL VALUES TO TXT
    with open('paths.txt', 'w') as f:
        for node in graph:
            f.write(','.join(str(x) for x in node))
            f.write('\n')
    ## WRITING ALL VALUES TO TXT

    ## SHOWING ALL DETECTUONS
    cv2.imshow("HoughCirlces",	planets)
    cv2.waitKey()
    ## SHOWING ALL DETECTUONS

#function to read txt and return graph with numeric values
def textintoGraph(file):
    graph = []
    with open(file) as f:
        lines = f.readlines()
        for x in lines:
            x = x.rstrip() 
            graph.append([int(i) for i in x.split(',')])

        return graph

#function to write graph to txt file
def graphintoText(graph):
    with open('dist.txt', 'w') as f:
        for node in graph:
            f.write(','.join(str(x) for x in node))
            f.write('\n')
            
graph = textintoGraph('paths.txt')
nodes = len(graph)

# function for floydWarshall
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


## TURTLE DRAWING GRAPHS BASE VALUES + points
WIDTH, HEIGHT = 1980, 1080
screen = Screen()
screen.setup(WIDTH + 4, HEIGHT + 8)  # fudge factors due to window borders & title bar
screen.setworldcoordinates(0, 0, WIDTH, HEIGHT)
screen.colormode(255)
speed(2)
points = []
## TURTLE DRAWING GRAPHS BASE VALUES + points

#function to generate random chords within window so it doesnt touch
def randomCords():
    x=random.randint(r, WIDTH-r)
    y=random.randint(r, HEIGHT-r)
    return [x,y]

#function to verify if chords are not generated at the same place
def verify(cords,points):
    if not points:
        return True
    for i in range(len(points)):
        #Check collisions Spaces between
        if cords[0] >= points[i][0]-200 and cords[0] >= points[i][0]+200 and cords[1] >= points[i][1]-200 and cords[1] >= points[i][1]+200:
            return False
    return True
# function that includes randomCords() and verify() this function generates new random chords if verification doesnt go through
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
# function to graw nodes from points to canvas
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
# Draw lines between drawn nodes on canvas
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