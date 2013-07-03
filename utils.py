#============================================================================
#                                  UTILS
#============================================================================
# Script name: utils.py
# Created on: 10/10/2012
# Author: Paula D. P. Costa
# Purpose: Collection of small pieces of codes that are useful
#          to someone that likes to play with image processing.
#
# Updates:
#  26/10/2012 - Added 'crop' function
#  11/01/2013 - Added 'addcoltofile' function
#  18/04/2013 - Added 'visualcheckImageDB' function
#  18/04/2013 - Added 'dumpMatrix2File' and 'loadMatrixFromFile'
#  07/06/2013 - Added 'applyKernelToPoints'
#
# Notice:
# Copyright (C) 2013  Paula D. Paro Costa
#=============================================================================

import numpy as np
import Image,ImageDraw
import cv2




#================================================
#               imdisplay
#
# Show OpenCV image and waits for ESC key. (CV2)
#================================================
def imdisplay(cv2_im):
    cv2.namedWindow('show')
    cv2.imshow('show',cv2_im)
    while True:
        ch=0xFF & cv2.waitKey()
        if ch==27:
            break
        cv2.imshow('show',cv2_im)
    cv2.destroyWindow('show')
    return


#================================================
#           drawPointsOnImage
#
# Draw points on a PIL image.
# im   -->     PIL image
# x,y  -->     arrays of point coordinates
# radius -->   radius of points
# zoom -->     zoom of displayed image
# color -->    color of points
#================================================

def drawPointsOnImage(im,x,y,radius=5,zoom=1,convert=True,color=(255,255,255)):
    if convert==True:
        im=im.convert('RGB')
    size=im.size
    size=int(size[0]*zoom),int(size[1]*zoom)
    #print size
    x=np.asarray(x)
    y=np.asarray(y)
    draw=ImageDraw.Draw(im)
    for i in range(x.shape[0]):
        draw.ellipse((x[i]-radius, y[i]-radius, x[i]+radius, y[i]+radius), fill=color)
    display=im.resize(size)
    return display


#==================================================
#               centroid
#
# Calculates the centroid of a shape
# sv --> shape vector defined by 'k' points
#        with coordinates x and y.
#        sv=(x0,y0,x1,y2,...,xk,yk)
#==================================================
def centroid(sv):
    sv_x=sv[0::2]
    sv_y=sv[1::2]
    xc=sum(sv_x)/float(len(sv_x))
    yc=sum(sv_y)/float(len(sv_y))
    return xc,yc

#==================================================
#               dist
#
# Calculates the euclidean distance between two points
# given their coordinates (x,y) and (u,v)
#==================================================
def dist(x,y,u,v):
    dist=np.sqrt(pow((x-u),2)+pow((y-v),2))
    return dist

#============================================================
#               nearest_point
#
# Given a set of points defined by the sequence of
# coordinates in the vectors 'x' e 'y', the function
# returns two vectors that determines the nearest point and
# the calculated distance between the corresponding point
# and the remaining points.
# EXAMPLE:
#      Consider the set of points:
#          P1=(10,20)
#          P2=(11,21)
#          P3=(100,200)
#      The input to the function will be:
#          x=([10,11,100])
#          y=([20,21,200])
#      P2 is the nearest point to P1 and vice-versa.
#      P2 is the nearest point of P3.
#      So, the function will return:
#          indices: [1,0,1]
#          distances:[1.41,1.41,199.9]
#
#============================================================
def nearest_point(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    number_of_points=x.shape[0]
    d=np.array(np.zeros(number_of_points))

    ut1,ut2=np.triu_indices(number_of_points,1) #without the main diagonal
    distances=dist(x[ut1[:]],y[ut1[:]],x[ut2[:]],y[ut2[:]])

    d_matrix=np.array(np.zeros((number_of_points,number_of_points)))
    d_matrix[ut1[:],ut2[:]]=distances[:]
    d_matrix[ut2[:],ut1[:]]=distances[:]
    d_matrix[range(number_of_points),range(number_of_points)]=np.max(distances)

    min_indices=np.array(np.zeros(number_of_points))
    min_dist=np.array(np.zeros(number_of_points))
    for i in range(number_of_points):
        min_indices[i]=int(np.argmin(d_matrix[i,:]))
        min_dist[i]=d_matrix[i,np.uint8(min_indices[i])]
        print min_indices[i]

    return min_indices,min_dist

#============================================================
#               reticulate
#
# Creates an 1 channel image/array of dimensions h x w pixels
# with a reticulate that is spaced s pixels, with lines l
# pixels large. (Background is white, net is black)
#============================================================

def reticulate(h=302,w=527,s=15,l=2):
    ret=np.array(np.zeros((h,w)))
    ret=ret+255
    for i in range(l):
        ret[:,i::s]=0
        ret[i::s]=0
    return ret

#===================================================================
#               crop
#
# im -> image (numpy array)
# ox -> column to start crop (column included in cropped image)
# oy -> row to start crop (row included in cropped image)
# width -> of final image
# height -> of final image
#===================================================================

def crop(im,ox,oy,width,height):
    cropped_image=im[oy:(oy+height),ox:(ox+width)]
    return cropped_image

#========================================================================
#               addcoltofile
#
# filename -> the array will be added as a column to this file
# a -> array (will be transformed on a 1d array)
# sep -> separator string
#
#========================================================================

def addcoltofile(filename,a,sep):
    a=np.ravel(np.asarray(a))
    s=sep
    try:
        f=open(filename,'r+')
    except IOError:
        s=""
        try:
            f=open(filename,'w+r+')
        except IOError:
            print "IOError."
            #return



    line=f.readline()

    if line=="":
        # File is empty
        for i in range(len(a)):
            f.write(str(a[i])+'\n')
    else:
        EOF=False
        pointer_to_write=0
        pointer_to_read=f.tell()
        new_line=line.rstrip('\n')+sep+str(a[0])+'\n'
        #print 'new_line= '+new_line
        invasion=len(new_line)-len(line)
        #print 'size of invasion='+str(invasion)

        #print 'pointer_to_write='+str(pointer_to_write)
        #print 'pointer_to_read='+str(pointer_to_read)
        buf=""
        for i in range(1,len(a)+1):
            #print EOF
            if EOF==False:
                aux=f.read(invasion)
                buf=buf+aux
                #print "Invasion read: "+str(aux)
                
            aux=""                        
            while (aux.find('\n')==-1) and (EOF==False):
                aux=f.read(1)
                buf=buf+aux
                #print 'updated buffer= \n'+buf
                if aux=="":
                    # Reached EOF
                    EOF=True
                    #print 'EOF'
                    break
                
            pointer_to_read=f.tell()
            
            f.seek(pointer_to_write)
            f.write(new_line)
            pointer_to_write=f.tell()
            f.seek(pointer_to_read)
            #print 'pointer_to_read='+str(pointer_to_read)
            #print 'pointer_to_write='+str(pointer_to_write)

            if i<(len(a)):
                x=buf.find('\n')
                line=buf[0:x+1]
                #print 'line= '+line
                new_line=line.rstrip('\n')+sep+str(a[i])+'\n'
                #print 'new_line= '+new_line
                invasion=len(new_line)
                #print 'size of invasion='+str(invasion)
                buf=buf[x+1::]
                #print 'buffer without line= \n'+buf
            else:
                break
                

            
        f.seek(pointer_to_write)          

        if f.readline()!="":
            print "Attention!The provided array has less elements than\n"
            print "the number of lines in the file."


        f.close()
        return


#========================================================================
#               visualCheckImageDB
#
# imagedb -> CSV filename
# imagedbtype -> 0 for a complete database (filenames+labels+shape)
#                1 for the simple database (filenames+shape)
# zoom -> to scale image on screen
#
#
#========================================================================
def visualcheckImageDB(imagedb,imagedbtype=0,zoom=0.5):
    import procdb
    if imagedbtype==0:
        images,shape,labels=procdb.processImageDB(imagedb)
    else:
        images,shape=procdb.processImageDB2(imagedb)

    shape=np.asarray(shape)
    print shape
    
    for i in range(len(images)):
        im=Image.open(images[i])
        im=drawPointsOnImage(im,shape[i,:,0],shape[i,:,1])
        im=im.resize((int(im.size[0]*zoom+0.5),int(im.size[1]*zoom+0.5)))
        print images[i]
        im.show()
        raw_input('Press ENTER to proceed to next image...')
    return


#========================================================================
#               dumpMatrix2File
#
# matrix -> numpy 1D or 2D arrays
# filename -> name of the file to be created
#
#
#========================================================================
def dumpMatrix2File(matrix,filename):
    datafile=open(filename,'w')
    dim=len(matrix.shape)
    if dim==1:
        datafile.write(','.join(map(str,matrix)))
    elif dim==2:
        for i in range(matrix.shape[0]):
            datafile.write(','.join(map(str,matrix[i])))
            datafile.write('\n')
    else:
        print "The matrix is not an 1D or 2D array. The matrix was not saved."

    datafile.close()
    return

#========================================================================
#               loadMatrixFromFile
#
# filename -> name of the file to be created
# rows -> number of rows of the matrix
# cols -> number of cols of the matrix
#
#========================================================================
def loadMatrixFromFile(filename, rows,cols):
    datafile=open(filename,'r')
    matrix=np.asarray(np.zeros((rows,cols)))
    for i in range(rows):
        aux=datafile.readline()
        aux=aux.split(',')
        matrix[i]=np.asarray(aux,dtype=np.float64)
        #print 'Reading line '+str(i)+' of file '+filename
    return matrix


#========================================================================
#               applyKernelToPoints
#========================================================================

def applyKernelToPoints(image,pts,kernel,border_type='BLACK'):
    """
    Applies the kernel (multiply and sum the neighborhood)
    at the specified points of an image.
    Returns an array of results for each selected point.

    The algorithm adds a frame to the original image to calculate
    the result of applying the kernel to the pixels that are at
    the borders of the original image.
    
    Key arguments:
    image   -- numpy array representing an image
    pts     -- array of points [[x1,y1],[x2,y2],...]
    kernel  -- numpy array with the weighting elements of the sum
    border_type -- BLACK (default) (added frame filled with pixels=0)
                   WHITE (added frame filled with pixels=255)
                   ANTIALIAS (infinite texture of replicated copies
                   of the original image) 
    """
    
    
    pts=np.asarray(pts)
    image=np.asarray(image)
    image.shape
    if len(image.shape)>2:
        grayscale=False
        shaperesult=(len(pts),image.shape[2])
    elif len(image.shape)==1:
        image=image.reshape(1,image.shape[0])
        shaperesult=len(pts)
        grayscale=True

    else:
        grayscale=True

    # Kernel dimensions - they are integers
    krows=kernel.shape[0]   
    kcols=kernel.shape[1]

    if krows%2==0:
        # Is even
        ldrows=(krows/2)-1
        udrows=krows/2
        
    else:
        # Is odd
        ldrows=krows/2
        udrows=krows/2

    if kcols%2==0:
        # Is even
        ldcols=(kcols/2)-1
        udcols=kcols/2
    else:
        # Is odd
        ldcols=kcols/2
        udcols=kcols/2

    #------------------------------------
    # ADD FRAME TO THE ORIGINAL IMAGE
    #------------------------------------

    dummyM=image.shape[0]+krows-1
    dummyN=image.shape[1]+kcols-1
    
    if grayscale==True:
        dummyimage=np.asarray(np.zeros((dummyM,dummyN)))
        
    else:
        dummyimage=np.asarray(np.zeros((dummyM,dummyN,image.shape[2])))

    if border_type=="WHITE":
        dummyimage=dummyimage+255

    elif border_type=="ANTIALIAS":
        # Fills top border
        dummyimage[0:ldrows,ldcols:ldcols+image.shape[1]]=image[image.shape[0]-ldrows:image.shape[0],:]

        # Fills bottom border
        dummyimage[(ldrows+image.shape[0]):,ldcols:(ldcols+image.shape[1])]=image[0:udrows,:]
        
        # Fills left border
        dummyimage[ldrows:ldrows+image.shape[0],0:ldcols]=image[:,image.shape[1]-ldcols:]

        # Fills right border
        dummyimage[ldrows:ldrows+image.shape[0],(ldcols+image.shape[1]):]=image[:,0:udcols]
        
        # Fills top, left corner
        dummyimage[0:ldrows,0:ldcols]=image[image.shape[0]-ldrows,image.shape[1]-ldcols]

        # Fills bottom, left corner
        dummyimage[(ldrows+image.shape[0]):,0:ldcols]=image[0:udrows,(image.shape[1]-ldcols):]
        
        # Fills top, right corner
        dummyimage[0:ldrows,(ldcols+image.shape[1]):]=image[(image.shape[0]-ldrows):,0:udcols]
        
        # Fills bottom, right corner
        dummyimage[(ldrows+image.shape[0]):,(ldcols+image.shape[1]):]=image[0:udrows,0:udcols]
                
    dummyimage[ldrows:ldrows+image.shape[0],ldcols:ldcols+image.shape[1]]=image    
    
    result=np.asarray(np.zeros(shaperesult))
    
    pts[:,0]=pts[:,0]+ldrows
    pts[:,1]=pts[:,1]+ldcols
        
    for k in range(len(pts)):
        total=0
        
        for i in range(-ldrows,udrows+1):
            for j in range(-ldcols,udcols+1):
                total=total+dummyimage[i+pts[k,0],j+pts[k,1]]*kernel[i+ldrows,j+ldcols]
                
        
        result[k]=total
                
  
    return result

#========================================================================
#               cropnscaleImageDB
#========================================================================

def cropnscaleImageDB(imagedb,newimagedb,ox,oy,width,height,scale,suffix,verbose=False):
    """
    Applies a crop (region of interest) followed by a scale operation
    on a set of images listed on an image database.

    The feature points on the image databased are modified to reflect
    the operations.
        
    Key arguments:
    imagedb   -- filename/path of the image database
    newimagedb  -- name of the file that will be created
    ox -- x origin of the crop operation
    oy -- y origin of the crop operation
    width -- width of the region of interest
    height -- height of the region of interest
    scale -- used to resize the region of interest
    prefix -- prefix to be added to the transformed images
    verbose -- If True provides feedback about the images being processed
    
    """

    import procdb
    import os

    images,shapes,labels=procdb.processImageDB(imagedb)
    shapes=np.asarray(shapes)
    print shapes.shape

    if verbose==True:
        print str(len(images))+" images to process."
    
    
    newimagedb=open(newimagedb,'w')

    for i in range(len(images)):
        im=cv2.imread(images[i])
        im_cropped=crop(im,ox,oy,width,height)
        newheight=int(height*scale)
        newwidth=int(width*scale)
        im_resized=np.asarray(np.zeros((newheight,newwidth)))
        im_resized=cv2.resize(im_cropped,(newwidth,newheight),im_resized,scale,scale,cv2.INTER_AREA)
        fileName, fileExtension = os.path.splitext(images[i])
        retval=cv2.imwrite(fileName+suffix+fileExtension,im_resized)
        if retval==False:
            print "Problem to save modified image."
            return False
        shapes[i,:,0]=shapes[i,:,0]-ox
        shapes[i,:,1]=shapes[i,:,1]-oy
        shapes[i]=shapes[i]*scale

        newshapes=''
        for j in range(shapes.shape[1]):
            newshapes=newshapes+',('+str(shapes[i,j,0])+';'+str(shapes[i,j,1])+')'

        newlabels=''
        for k in range(len(labels[i])):
            newlabels=newlabels+','+str(labels[i][k])

        newimagedb.write(fileName+suffix+fileExtension+newlabels+newshapes+'\n')

        if verbose==True:
            print "Image "+str(i+1)+" successfully processed."
        
    newimagedb.close()

    return True
