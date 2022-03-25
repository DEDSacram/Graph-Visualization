from turtle import *
import random
import math





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
        goto(points[i][0]-2,points[i][1]+8)
        pendown()
        write(i)
        penup()
        goto(points[i][0],points[i][1])
        pendown()
        circle(15)
def drawconnectionlines(points,graph,offset,add,distances):
    
    r = 15
    nodeint = 0
    i = 0
    for node in graph:
        if add:
            color((random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)))
        for dist in node:
            
            if add:
                print(dist,distances[nodeint][i])
                if dist == distances[nodeint][i]:
                    i+=1
                    continue
            if dist != 0 and dist != 123:
                #from mid point
                # point1 = (points[i][0], points[i][1]+15)
                # point2 = (points[nodeint][0], points[nodeint][1]+15)
                angle = math.atan2(points[i][1]+r - points[nodeint][1]+r, points[i][0] - points[nodeint][0])*(180/math.pi)
                xoncircle = r*math.sin(angle+offset)
                yoncircle = r*math.cos(angle+offset)
                point1 = (points[i][0]+xoncircle, points[i][1]+r+yoncircle)
                point2 = (points[nodeint][0]+xoncircle, points[nodeint][1]+r+yoncircle)
                penup()
                #write number in middle
                goto(int((point2[0]+point1[0])/2),int((point2[1]+point1[1])/2))
                write(dist)
                #continue
                goto(point2)
                pendown()
                goto(point1)
            i+=1
        i=0
        nodeint+=1
        


floydWarshall(graph)
distances = textintoGraph('dist.txt')

points = GenerateCordsforNodes(randomCords(),points)
drawnodes(nodes,points)
drawconnectionlines(points,graph,0,False,distances)
drawconnectionlines(points,distances,15,True,graph)
done()

