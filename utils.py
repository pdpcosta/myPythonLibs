#============================================================================
#                                  UTILS
#============================================================================
# Script name: utils.py
# Created on: 10/10/2012
# Author: Paula D. Paro Costa
# Purpose: Collection of snnipets that may be useful
#          to someone that likes to play with image processing.
#
# Updates:
#  26/10/2012 - Added 'crop' function
#  11/01/2013 - Added 'addcoltofile' function
#  18/04/2013 - Added 'visualcheckImageDB' function
#  18/04/2013 - Added 'dumpMatrix2File' and 'loadMatrixFromFile'
#  07/06/2013 - Added 'applyKernelToPoints'
#  03/07/2013 - Added 'cropnscaleImageDB' function. 'crop' function modified
#               to reflect changes in the library procdb (shapes variable as list
#               of tuples)
#  05/07/2013 - Added functions: 'alignPairShapes', 'RST', 'alignNImages'
#  17/07/2013 - Added class 'Eigentextures'
#  23/07/2013 - Changed the 'loadMatrixFromFile' function to determine automatically
#               the size of the matrix if not provided.
#
# Notice:
# Copyright (C) 2013  Paula D. Paro Costa
#=============================================================================

import numpy as np
import numpy.linalg as la
import Image,ImageDraw
import cv2
import os



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
#        sv=[[x0,y0],[x1,y1],...,[xk,yk]]
#==================================================
def centroid(sv):
    sv=np.asarray(sv)
    sv_x=sv[:,0]
    sv_y=sv[:,1]
    xc=sum(sv_x)/float(len(sv_x))
    yc=sum(sv_y)/float(len(sv_y))
    return xc,yc

#========================================================================
#               alignPairShapes
#========================================================================
def alignPairShapes(s1,s2,weights):
    """
    Given two vector shapes, the function applies
    the minimum squared error to align s2 with s1.
    The implementation is based on the paper from
    Cootes et al., "Active Shape Models -- Their Training and Application", 1995
    See Appendix A.
    
    Key arguments:
    s1   -- array of tuples representing the first shape vector with n landmarks
            [(x1,y1),(x2,y2),...,(xn,yn)]
    s2   -- array of tuples representing the second shape vector with n landmarks
    weights -- vector of n weights that control how a landmark influences the alignment
               (greater weight values have greater impact on the alignment).

    Outputs:
    The coefficients of the affine Rotation, Scaling and Translation (RST) transform
    ax -- s.cos(theta)
    ay -- s.sin(theta)
    tx -- translation in x
    ty -- translation in y
    """


    s1=np.asarray(s1)
    s2=np.asarray(s2)
    
    x1k=s1[:,0]
    y1k=s1[:,1]
    x2k=s2[:,0]
    y2k=s2[:,1]

    X1=sum(x1k*weights) 
    X2=sum(x2k*weights)

    Y1=sum(y1k*weights)
    Y2=sum(y2k*weights)

    Z=sum(weights*(pow(x2k,2)+pow(y2k,2)))

    W=sum(weights)

    C1=sum(weights*(x1k*x2k+y1k*y2k))

    C2=sum(weights*(y1k*x2k-x1k*y2k))
    
    a=np.asarray([[X2,-Y2,W,0],[Y2,X2,0,W],[Z,0,X2,Y2],[0,Z,-Y2,X2]])
    b=np.asarray([X1,Y1,C1,C2])

    x=np.linalg.solve(a,b)

    ax=x[0]
    ay=x[1]
    tx=x[2]
    ty=x[3]
    return ax,ay,tx,ty

#===========================================================
#                   RST
#===========================================================
def RST(s,ax,ay,tx,ty):
    """
    Apply rotation, scale and translation to a shape vector,
    given the coefficients of the affine transformation matrix.
    
    Key arguments:
    s -- array of tuples representing the shape vector with n landmarks
         [(x1,y1),(x2,y2),...,(xn,yn)]
    The coefficients of the affine Rotation, Scaling and Translation (RST) transform:
    ax -- s.cos(theta)
    ay -- s.sin(theta)
    tx -- translation in x
    ty -- translation in y
    """
    
    svRST=np.asarray(np.zeros(s.shape))
    svRST[:,0]=ax*s[:,0]-ay*s[:,1]+tx
    svRST[:,1]=ay*s[:,0]+ax*s[:,1]+ty

    return svRST


#========================================================================
#                   alignNImages
#========================================================================
def alignNImages(images,shapes,weights,save_aligned_images=True):
    """
    
    Aligns a set of images according to their shapes.
    Together with functions 'alignPairShapes' and 'RST' this
    function implements the shape alignment algorithm used in
    the Active Shape Model (ASM). For additional references see:
    "Active Shape Models", Cootes et al., 1995
    "Active Appearance Models", Stegmann, 2000, Chapter 4, Section 4.4.2
    
    Key arguments:
    images  -- array of images filenames (N images)
    shapes  -- array of shapes corresponding to each image
    weights -- vector of weights that control how a landmark influences 
               the alignment (greater weight values have greater impact 
               on the alignment).
    save_aligned_images -- if True, aligned images are saved in the same
                           folder with "aligned" prefix. 
                           
    """             
        
    shapes=np.asarray(shapes)
    aligned_shapes=np.asarray(np.zeros(shapes.shape))
    aligned_shapes.astype(float)

    print "Starting alignment of "+str(len(images))+"."
    
    # Variables initialization
    it=0
    first=True
    mean_shape=shapes[0]
    print mean_shape.shape
    previous_mean_shape=np.asarray(np.zeros((shapes.shape[1],shapes.shape[2])))
    ax=np.asarray(np.zeros(shapes.shape[0]))
    ay=np.asarray(np.zeros(shapes.shape[0]))
    tx=np.asarray(np.zeros(shapes.shape[0]))
    ty=np.asarray(np.zeros(shapes.shape[0]))
    
    # The "while" loop checks the convergence of the alignment.
    # The convergence is checked measuring the difference of previous mean_shape
    # an the last calculated mean shape.    
    
    error=sum(abs(np.ravel(previous_mean_shape)-np.ravel(mean_shape)))
    print "error = "+str(error)              
    while (error>0.0001):

        print sum(abs(np.ravel(previous_mean_shape)-np.ravel(mean_shape)))
        print 'Iteration ',it
        it=it+1
        previous_mean_shape=np.copy(mean_shape)
     

        # Normalizing the mean shape to the first shape
        axm,aym,txm,tym=alignPairShapes(shapes[0],mean_shape,weights)
        mean_shape=RST(mean_shape,axm,aym,txm,tym)

        # Align all shapes to the mean shape
        for i in range(len(images)):
            #print 'Aligning shape '+str(i)
            ax[i],ay[i],tx[i],ty[i]=alignPairShapes(mean_shape,shapes[i],weights)
            aligned_shapes[i]=RST(shapes[i],ax[i],ay[i],tx[i],ty[i])

        # Calculate new mean shape
        mean_shape=np.add.reduce(aligned_shapes)/float(aligned_shapes.shape[0])
        #print mean_shape

        error=sum(abs(np.ravel(previous_mean_shape)-np.ravel(mean_shape)))
        print "error = "+str(error) 
        
    if save_aligned_images==True:
            for i in range(len(images)):
                im=cv2.imread(images[i])
                dsize=(im.shape[1],im.shape[0])
                T=np.asarray([[ax[i],-ay[i],tx[i]],[ay[i],ax[i],ty[i]]])
                im=cv2.warpAffine(im,T,dsize)
                fileName, fileExtension = os.path.splitext(os.path.basename(images[i]))
                cv2.imwrite(fileName+'_aligned'+fileExtension,im)
                            
    return mean_shape,aligned_shapes



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
    
    try:
        f=open(filename,'r+')
    except IOError:
        
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
#========================================================================
def loadMatrixFromFile(filename, rows=0,cols=0,sep=','):
    """

    Loads a numpy matrix from a text file, typically a CSV file.

    Key arguments:
    filename -- text
    rows -- (optional) specifies the number of rows of the matrix
    cols -- (optional) specifies the number of columns of the matrix
    sep -- separator of the columns, the default is a comma
    """
    datafile=open(filename,'r')
    if rows!=0 and cols!=0:    
        matrix=np.asarray(np.zeros((rows,cols)))
        for i in range(rows):
            aux=datafile.readline()
            aux=aux.split(sep)
            matrix[i]=np.asarray(aux,dtype=np.float64)
            #print 'Reading line '+str(i)+' of file '+filename
        return matrix
    else:
        aux=datafile.readline()
        aux=aux.split(sep)
        matrix=np.asarray([aux],dtype=np.float64)
        aux=datafile.readline()
        while aux!="":
            aux=aux.split(sep)
            #print aux
            matrix=np.append(matrix,np.asarray([aux],dtype=np.float64),0)
            aux=datafile.readline()
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

def cropnscaleImageDB(imagedb,newimagedb,ox,oy,width,height,scale,folder="",verbose=False):
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
    folder -- where the images are going to be saved; if not provided,
              a new directory is created automatically.
    verbose -- If True provides feedback about the images being processed
    
    """


    import procdb
    import os

    images,shapes,labels=procdb.processImageDB(imagedb)
    shapes=np.asarray(shapes)
    #print shapes.shape

    if verbose==True:
        print str(len(images))+" images to process."
    
    
    suffix="_"+str(int(width*scale))+"x"+str(int(height*scale))
    if folder=="":
        folder=str(int(width*scale))+"x"+str(int(height*scale))
        if not os.path.exists(folder): os.makedirs(folder)
    else:
        if not os.path.exists(folder):os.makedirs(folder)

    newimagedb=open(folder+"/"+newimagedb,'w')

    for i in range(len(images)):
        im=cv2.imread(images[i])
        im_cropped=crop(im,ox,oy,width,height)
        newheight=int(height*scale)
        newwidth=int(width*scale)
        im_resized=np.asarray(np.zeros((newheight,newwidth)))
        im_resized=cv2.resize(im_cropped,(newwidth,newheight),im_resized,scale,scale,cv2.INTER_AREA)
        fileName, fileExtension = os.path.splitext(images[i])
        
        retval=cv2.imwrite(folder+"/"+fileName+suffix+fileExtension,im_resized)
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

#=================================================================================
#               Eigentextures
#=================================================================================

class Eigentextures:
    '''

    This class implements the principal components analysis (PCA) or
    the eigendecomposition of high dimensional vectors.
    This class was designed having in mind its use for whole images or
    parts of images, or simply, textures.
    If the images contain faces, the algorithm implemented here is
    equivalent to the Eigenfaces algorithm presented by Matthew Turk and
    Alex Pentland.
    The variables names correspond to the variables present in the paper
    "Eigenfaces for Recognition", Matthew Turk and Alex Pentland, 1991.

    Key arguments:
    trainingset -   Array of rasterized textures. Each sample corresponds to
                    a row of the array.
    evr -           Explained variance ratio: indicates how much of the overall variance
                    is explained by the corresponding principal component or eigenvector.
    numpc -         This parameter is used to inform how many principal components the
                    function should consider.
    '''
    
    def __init__(self,trainingset,verbose=False):
        self.__verbose=verbose
        self.M=trainingset.shape[0]
        self.N=trainingset.shape[1]

        # STEP 1
        # Gamma is the matrix which columns are the rasterized pixels of each image of
        # the training set
        Gamma=np.transpose(trainingset)
    

        # STEP 2
        # Compute Psi, that is the average texture over the training set.
        Psi=Gamma.mean(1)
        self.Psi=Psi
        Psi=(Psi.round()).astype(np.int32)
        Psi=np.reshape(Psi,(Psi.shape[0],1))

        # STEP 3
        # Subtracts the average face from all samples, creating a zero mean
        # distribution Phi.
        self.__Phi=np.asarray(np.zeros(Gamma.shape),dtype=np.int32)
        self.__Phi=Gamma-Psi
        del Gamma
        del trainingset
        if self.__verbose==True: print "Eigentextures:\tPhi created successfully."

        # STEP 4
        # A minor product of the covariance matrix is calculated.
        Phi_t=np.transpose(self.__Phi)
        L=np.dot(Phi_t,self.__Phi)
        del Phi_t
        L=L/self.M
        if self.__verbose==True: print "Eigentextures:\tMinor product generated successfully."


        # STEP 5
        # Calculates the eigenvalues(w) and eigenvectors(v) of
        # the minor product L.
        self.__w,self.__v=la.eig(L)
    
        del L

        # STEP 6
        # Order the eigenvalues and their corresponding eigenvectors
        # in the descending order.
        indices=np.argsort(self.__w)
        indices=indices[::-1]           # descending order
        self.__w=self.__w[indices]
        self.__v=self.__v[:,indices]
        # Calculating the explained variance ratio.
        self.evr=self.__w/np.sum(self.__w)
        
        if self.__verbose==True: print "Eigentextures:\tObject created succesfully."
        return

    def getEigentextures(self,numpc="all"):
        
        # Calculates the eigenvectors of the original covariance matrix
        if numpc=='all':
            self.__u=np.asarray(np.zeros((self.N,self.M)))
            for col in range(self.M):
                if self.__verbose==True: print "Calculating eigentexture "+str(col+1)
                h=np.dot(self.__Phi,self.__v[:,col])
                h=h/la.norm(h)
                self.__u[:,col]=h
            return self.__u
        elif numpc>0 and numpc<=self.M:
            numpc=int(numpc+0.5)
            self.__u=np.asarray(np.zeros((self.N,numpc)))
            for col in range(numpc):
                if self.__verbose==True: print "Calculating eigentexture "+str(col+1)
                h=np.dot(self.__Phi,self.__v[:,col])
                h=h/la.norm(h)
                self.__u[:,col]=h
            return self.__u
        else:
            print "Eigentextures:\tInvalid value for numpc."
  
            return
    
    def getEigentexturesEVR(self,variance=1):
             
        # Calculates the eigenvectors of the original covariance matrix
        if variance>=1:
            self.__u=np.asarray(np.zeros((self.N,self.M)))
            for col in range(self.M):
                if self.__verbose==True: print "Calculating eigentexture "+str(col+1)
                h=np.dot(self.__Phi,self.__v[:,col])
                h=h/la.norm(h)
                self.__u[:,col]=h
            return self.__u
        elif variance<1 and variance>0:
            cols=np.where(np.cumsum(self.evr)<=variance)[0]
            self.__u=np.asarray(np.zeros((self.N,len(cols))))
            for col in cols:
                if self.__verbose==True: print "Calculating eigentexture "+str(col+1)
                h=np.dot(self.__Phi,self.__v[:,col])
                h=h/la.norm(h)
                self.__u[:,col]=h
            return self.__u
        else:
            print "Eigentextures:\t Invalid explained value ratio parameter."
            return
    
    def saveEigentextures2File(self,filename,numpc="all"):
        u=self.getEigentextures(numpc)
        dumpMatrix2File(u,filename)
        return
    
    def saveEVR2File(self,filename,variance=1):
        u=self.getEigentexturesEVR(variance)
        dumpMatrix2File(u,filename)
        return


    
#=================================================================================
#               PCA
#=================================================================================

class PCA:
    '''

    This class is a simple implementation of principal components analysis (PCA)
    through the computation of the eigenvectors of the covariance matrix of a training
    set of samples. For high dimensional vectors, see the class Eigentextures.

    Key arguments:
    trainingset -   Matrix of samples. Each sample corresponds to
                    a row of the array.
    evr -           Explained variance ratio: indicates how much of the overall variance
                    is explained by the corresponding principal component or eigenvector.
    numpc -         This parameter is used to inform how many principal components the
                    function should consider.
    '''
    
    def __init__(self,trainingset,verbose=False):
        self.__verbose=verbose
        self.N=trainingset.shape[0] # number of samples/trials
        self.M=trainingset.shape[1] # number of dimensions (size of the sample vector)

        # STEP 2
        # Compute Psi, that is the average vector considering the training set
        Psi=trainingset.mean(0)
        self.Psi=Psi
        

        # STEP 3
        # Subtracts the average from all samples, creating a zero mean
        # distribution Phi.
        Phi=np.asarray(np.zeros(trainingset.shape))
        Phi=trainingset-Psi
        
        if self.__verbose==True: print "PCA:\tPhi created successfully."

        # STEP 4
        # Computes the covariance matrix.
        # covariance=1/((N-1)*trainingset_t*trainingset)
        
        Phi_t=np.transpose(Phi) # M x N matrix
        covariance=(np.dot(Phi_t,Phi))/(self.N-1)
        self.cov=covariance
        
               
        self.__w,self.__v=la.eig(covariance)
        
        # The covariance is a positive semi-definite matrix
        # and all of its eigenvalues are positive.
        # However, the linalg.eig function may return small and
        # negative eigenvalues. Before calculating the explained variance ratio,
        # values below 1e-10 are made equal zero.
        self.__w=np.where(self.__w<=1e-10,0,self.__w)
               
        # Putting eigenvectors in the descending order of eigenvalues
        indices=np.argsort(self.__w)
        indices=indices[::-1]      
        self.__w=self.__w[indices]
        self.__v=self.__v[:,indices]
        # Calculating the explained variance ratio.
        self.evr=self.__w/np.sum(self.__w)
        
        if self.__verbose==True: print "PCA:\tObject created succesfully."
        return

    def getPC(self,numpc="all"):
        
        
        # Calculates the eigenvectors of the original covariance matrix
        if numpc=='all':
            return self.__v
        elif numpc>0 and numpc<=self.N:
            numpc=int(numpc)   
            return self.__v[:,0:numpc]
        else:
            print "PCA:\tInvalid value for numpc."
  
            return
    
    def getEVR(self,variance=1):
             
        # Calculates the eigenvectors of the original covariance matrix
        if variance>=1:
            return self.__v
        elif variance<1 and variance>0:
            cols=np.where(np.cumsum(self.evr)<=variance)[0]
            return self.__v[:,cols]
        else:
            print "PCA:\t Invalid explained variance ratio parameter."
            return
    
    def savePC2File(self,filename,numpc="all"):
        v=self.getPC(numpc)
        dumpMatrix2File(v,filename)
        return
    
    def saveEVR2File(self,filename,variance=1):
        v=self.getEVR(variance)
        dumpMatrix2File(v,filename)
        return


        
    

        
    
                
                
            
    
        
    
    
