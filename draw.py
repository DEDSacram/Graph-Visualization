from turtle import *
import random
import math
import cv2
import numpy as np
import os

r = 30
basedir = "data/images/"
images = ["Cool.png","pog.png","pepeppo.png","HAHAUDIETHANOSSNAP.png"] # to be changed
imagepath = basedir + images[0]
def detectlines(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    threshold = 2
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

   
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    # points = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
       

    return lines

def imageintoGraph(path):
    # os.system(f"cd .. &&  python detect.py --weights runs/train/graphs/weights/last.pt --img 640 --conf 0.6 --source "+path+" --hide-labels")

    prefix = "../"

    cords = [] # all detections
    centercords = [] # all detections

    nodes = []
    edges = []
    lineedges = detectlines(prefix+path) # lines estimate

    planets	= cv2.imread(prefix+path)
    gray_img=cv2.cvtColor(planets,	cv2.COLOR_BGR2GRAY)
    img	= cv2.medianBlur(gray_img,	1)
    print(lineedges)
    
    circles	= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,param1=100,param2=30,minRadius=20,maxRadius=120)
    circles	= np.uint16(np.around(circles))

  
    with open("alldetections.txt") as f:
        lines = f.readlines()
        for x in lines:
            x = x.rstrip() 
            cords.append([int(float(i)) for i in x.split(',')])

    for cord in cords:
        width = (cord[2]-cord[0])
        height = (cord[1]-cord[3])
        centerx = cord[0] + int(width/2)
        centery = cord[1] - int(height/2)
     
        centercords.append([centerx,centery])
    # (x - center_x)² + (y - center_y)² < radius²
    # for	i in circles[0,:]:         
    #                 for cord in centercords:
    #                     if((math.pow((cord[0] - i[0]), 2) + math.pow((cord[1] - i[1]), 2)) < math.pow(i[2],2)):
    #                         # print(True)
    #                         nodes.append([cord[0],cord[1]])
    #                     else:
    #                         # print(False)
    #                         if [cord[0],cord[1]] not in edges:
    #                             edges.append([cord[0],cord[1]])
    #                 #	draw	the	outer	circle
    #                 cv2.circle(planets,(i[0],i[1]),i[2],(0,128,255),6)
    #                 # #	draw	the	center	of	the	circle
    #                 # cv2.circle(planets,(i[0],i[1]),2,(0,0,255),3)



    # Check if in circle then append
    for cord in centercords:
        isin = False
        for i in circles[0]:
            if((math.pow((cord[0] - i[0]), 2) + math.pow((cord[1] - i[1]), 2)) < math.pow(i[2],2)):
                        isin = True
            cv2.circle(planets,(i[0],i[1]),i[2],(0,128,64),6)
        if isin:
            nodes.append([cord[0],cord[1]])
        else:
            edges.append([cord[0],cord[1]])

    # Draw
    for a in nodes:
         cv2.circle(planets,(a[0],a[1]),20,(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),3)
    for b in edges:
         cv2.circle(planets,(b[0],b[1]),20,(255,0,128),3)
    for line in lineedges:
        for x1,y1,x2,y2 in line:
            # points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv2.line(planets,(x1,y1),(x2,y2),(255,0,0),5)
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
graph = textintoGraph('graph.txt')
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
speed(10)
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
distances = textintoGraph('dist.txt')
points = GenerateCordsforNodes(randomCords(),points)
drawnodes(nodes,points)
drawconnectionlines(points,graph,0,False,distances)
drawconnectionlines(points,distances,15,True,graph)
done()