import pygame
from time import sleep
import argparse
import sys



### IMAGE GLITCHING _____ _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _ _ __ _ _ __ _ _ __ _- _ _ -_ _ _
class Sorter:
	def __init__(self):
		pass

	def sort(self,image):
		return image

class repeat(Sorter):
	repeats=0
	sorter=None
	def __init__(self,repeats,sorter):
		self.sorter=sorter
		self.repeats=repeats

	def sort(self,image):
		for n in range(self.repeats):
			print ">> repeat %s"%(n)
			image=self.sorter.sort(image)
		return image

class Chain(Sorter):
	sorters=[]
	def __init__(self, *sorters):
		self.sorters=sorters

	def sort(self,image):
		for i in range(len(self.sorters)):
			print "> sorter %s"%(i)
			image=self.sorters[i].sort(image)
		return image

class rotate(Sorter):
	def __init__(self,turns):
		self.turns=turns

	def sort(self,image):
		return pygame.transform.rotate(image,90*self.turns)


class RelayScrambler(Sorter):
	def __init__(self, overwrite=True):
		self.overwrite=overwrite

	def sort(self,image):
		copysurf = image.copy()
		for channel in [0,1,2]:
			print "channel %s"%(channel)
			for x in range(image.get_width()):
				for y in range(image.get_height()):
					ay = int( y+image.get_at( (x,y) ).cmy[channel]*image.get_height() )%image.get_height()

					c = image.get_at((x,y))
					b = image.get_at((x,ay))
					if(self.overwrite):
						if(channel==0):
							b.r = c.r
						if(channel==1):
							b.g = c.g
						if(channel==2):
							b.b = c.b
					else:
						if(channel==0):
							n = b.r+c.r
							if(n>255):
								n=255
							b.r=n
						if(channel==1):
							n = b.g+c.g
							if(n>255):
								n=255
							b.g=n
						if(channel==2):
							n = b.b+c.b
							if(n>255):
								n=255
							b.b=n
					copysurf.set_at( (x,ay), b)

		return copysurf

class MeltIntoWall_broken_1(Sorter):

	def isWall(self,color):
		return color.hsva[2]<10

	def isLarger(self,colorA,colorB):
		return colorA.hsva[2]>colorB.hsva[2]

	def qsort(self,list):
	    if list == []:
	        return []
	    else:
	        pivot = list[0]
	        lesser = self.qsort([x for x in list[1:] if self.isLarger(pivot,x)])
	        greater = self.qsort([x for x in list[1:] if (x == pivot or self.isLarger(x,pivot))])
	        return lesser + [pivot] + greater


	def sort(self,image):
		out = image.copy();
		for x in range(image.get_width()):
			regionstart=0
			for y in range(image.get_height()):
				if(self.isWall(image.get_at( (x,y) )) or y==image.get_height()-1):
					order = range(regionstart,y)
					#b = order
					for i in range(len(order)):
						order[i]=image.get_at( (x,i) )
					self.qsort(order)
					#print (b,order)

					for i in range(len(order)):
						out.set_at( (x,regionstart+i), order[i] )

					regionstart=y
		return out

class MeltIntoWall_broken_2(Sorter):

	def isWall(self,color):
		return color.hsva[2]<10

	def isLarger(self,colorA,colorB):
		return colorA.hsva[2]>colorB.hsva[2]

	def qsort(self,list):
	    if list == []:
	        return []
	    else:
	        pivot = list[0]
	        lesser = self.qsort([x for x in list[1:] if self.isLarger(pivot,x)])
	        greater = self.qsort([x for x in list[1:] if (x == pivot or self.isLarger(x,pivot))])
	        return lesser + [pivot] + greater


	def sort(self,image):
		out = image.copy();
		for x in range(image.get_width()):
			regionstart=0
			for y in range(image.get_height()):
				if(self.isWall(image.get_at( (x,y) )) or y==image.get_height()-1):
					order = range(regionstart,y)
					#b = order
					for i in range(len(order)):
						order[i]=image.get_at( (x,order[i]) )
					order=self.qsort(order)
					#print (b,order)

					for i in range(len(order)):
						out.set_at( (x,regionstart+i), order[i] )

					regionstart=y
		return out




class MeltIntoWall(Sorter):

	def isWall(self,color):
		return color.hsva[2]<=10

	def isLarger(self,colorA,colorB):
		return colorA.hsva[2]>colorB.hsva[2]

	def sort(self,image):
		out = image.copy();
		for x in range(image.get_width()):
			regionstart=0
			for y in range(image.get_height()):
				if(self.isWall(image.get_at( (x,y) )) or y==image.get_height()-1):
					order = range(regionstart,y)
					for i in range(len(order)):
						order[i]=image.get_at( (x,order[i]) )

					order.sort(key = lambda x: x.hsva[2])
					#print (image.get_height(), len(order));

					for i in range(len(order)):
						out.set_at( (x,regionstart+i), order[i] )

					regionstart=y
		return out


class MeltByEh(MeltIntoWall):

	def isWall(self,color):
		return color.hsva[0]%10==0

	def isLarger(self,colorA,colorB):
		return colorA.hsva[1]>colorb.hsva[1]

class Melt(MeltIntoWall):

	def isWall(self,color):
		return False

	def isLarger(self,colorA,colorB):
		return colorA.hsva[1]>colorb.hsva[1]




class bitshift(Sorter):
	shift=0
	def __init__(self, shift):
		self.shift=shift

	def sort(self,image):
		buff = image.get_buffer().raw
		n=len(buff)
		new=""
		for b in range(n):
			if(1.0*b/n)%0.1==0:
				print(1.0*b/n)
			#TODO wraparound for better fx
			if(self.shift>0):
				new+=chr( abs( ord(buff[b])>>self.shift )%256 )
			else:
				new+=chr( abs( ord(buff[b])<< (-self.shift) )%256 )


		return pygame.image.fromstring(new,(image.get_width(), image.get_height()), "RGB", False)


class oradjacent(Sorter):
	shift=1
	def __init__(self,shift=1):
		self.shift=shift

	def sort(self,image):
		buff = image.get_buffer().raw
		n=len(buff)
		new=""
		for b in range(n):
			if(1.0*b/n)%0.1==0:
				print(1.0*b/n)
			new+=chr( ( ord(buff[b])|ord(buff[(b+self.shift)%len(buff)]) )%256 )


		return pygame.image.fromstring(new,(image.get_width(), image.get_height()), "RGB", False)

class xoradjacent(Sorter):
	def __init__(self,shift=1):
		self.shift=shift

	def sort(self,image):
		buff = image.get_buffer().raw
		n=len(buff)
		new=""
		for b in range(n):
			if(1.0*b/n)%0.1==0:
				print(1.0*b/n)
			new+=chr( ( ord(buff[b])^ord(buff[(b+self.shift)%len(buff)]) )%256 )


		return pygame.image.fromstring(new,(image.get_width(), image.get_height()), "RGB", False)

class andadjacent(Sorter):
	def __init__(self,shift=1):
		self.shift=shift

	def sort(self,image):
		buff = image.get_buffer().raw
		n=len(buff)
		new=""
		for b in range(n):
			if(1.0*b/n)%0.1==0:
				print(1.0*b/n)
			new+=chr( ( ord(buff[b])&ord(buff[(b+self.shift)%len(buff)]) )%256 )


		return pygame.image.fromstring(new,(image.get_width(), image.get_height()), "RGB", False)

class sortbycolumn(Sorter):
	columnwidth=0
	def __init__(self,columnwidth=0):
		self.columnwidth=columnwidth

	def evaluate(self,img):
		value = 0
		for x in range(img.get_width()):
			for y in range(img.get_height()):
				value+=self.ecol(img.get_at( (x,y) ))
		return value

	def ecol(self,color):
		return color.hsva[1]

	def sort(self,image):
		out = image.copy()
		blocks = []
		for i in range(image.get_width()/self.columnwidth):
			img = pygame.Surface( (self.columnwidth, image.get_height()) )
			img.blit(image, (-i*self.columnwidth,0) )
			blocks.append( img )
		if ( image.get_width()%self.columnwidth!=0 ):
			xoff = (image.get_width()/self.columnwidth)*self.columnwidth#because rounding
			i = pygame.Surface((
				image.get_width()-xoff,
				image.get_height() ))
			i.blit(image, (-xoff,0) )
			blocks.append(i)

		for i in range(len(blocks)):
			blocks[i] = (self.evaluate(blocks[i]),blocks[i])

		blocks.sort(key = lambda x: x[0])

		x=0
		for i in blocks:
			out.blit(i[1],(x,0))
			x+=i[1].get_width()

		return out

class sortbycolumn_hue(sortbycolumn):
	def ecol(self,color):
		return color.hsva[0]

class sortbycolumn_r(sortbycolumn):
	def ecol(self,color):
		return color.r


#### IMAGE PROCESSING {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
import cv2 #Opencv
import cv#Opencv
import numpy as np

import Image #Image from PIL
import glob
import os
from PIL import Image

imgs = []

def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    # modified from: http://www.lucaamore.com/?p=638

    #variables
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    # Equalize the histogram
    cv.EqualizeHist(image, image)

    # Detect the faces
    faces = cv.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

    # If faces are found
    if faces and returnImage:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0]  )
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)



def faceCrop(imagePattern,boxScale=1):
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load('haarcascade_frontalface_alt.xml')

    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print 'No Images Found'
        return

    for img in imgList:
        pil_im=Image.open(img)
        cv_im=pil2cvGrey(pil_im)
        faces=DetectFace(cv_im,faceCascade)
        if faces:
            n=1
            for face in faces:
                croppedImage=imgCrop(pil_im, face[0],boxScale=boxScale)
                fname,ext=os.path.splitext(img)
                croppedImage.save('crop'+str(n)+ext)
                imgs.append('crop'+str(n)+ext)


                n+=1
        else:
            print 'No faces found:', img


def test(imageFilePath):
    pil_im=Image.open(imageFilePath)
    cv_im=pil2cvGrey(pil_im)
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load('haarcascade_frontalface_alt.xml')
    face_im=DetectFace(cv_im,faceCascade, returnImage=True)
    img=cv2pil(face_im)
    # img.show()
    # img.save('test.png')














def main():
	pygame.init()

	print "working"



	print bgImage
	faceCrop(fgImage)
	background = Image.open(bgImage)
	fore = Image.open(imgs[0])

		# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 500;

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 140

	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.1

	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.87

	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.01

	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
	    detector = cv2.SimpleBlobDetector(params)
	else :
	    detector = cv2.SimpleBlobDetector_create(params)

		# Read image
	im = cv2.imread(bgImage, cv2.IMREAD_GRAYSCALE)

	# Set up the detector with default parameters.
	detector = cv2.SimpleBlobDetector()

	# Detect blobs.
	keypoints = detector.detect(im)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

	for key in keypoints:
	    # Show keypoints
	    x = key.pt[0]
	    y = key.pt[1]

	    background.paste(fore, (int(x),int(y)))


	background.paste(fore, (123,245))

	background.save('img.jpg')
	# sorter = Chain( sortbycolumn(40) )
	# sorter = Chain( rotate(1), sortbycolumn(20), rotate(-1) )
	# sorter = Chain( rotate(1), sortbycolumn(20), rotate(-1), sortbycolumn(20))
	# sorter = Chain( sortbycolumn(5), rotate(-1), sortbycolumn(5), rotate(1) )
	# sorter = Chain( rotate(2),sortbycolumn(20), rotate(-20), Melt())
	# sorter = Chain( rotate(1), sortbycolumn(10), rotate(-1), MeltByEh() )
	sorter = Chain( rotate(1), sortbycolumn(10), rotate(-1), RelayScrambler(), MeltByEh() )
	#sorter = Chain( rotate(1), sortbycolumn(10), rotate(-1), xoradjacent(), MeltByEh() )
	# sorter = Chain(rotate(-1), MeltByEh(), rotate(3), sortbycolumn(10) )


	raw = pygame.image.load('img.jpg')
	out = sorter.sort( raw )
	print "done"
	joined = pygame.Surface( (raw.get_width(), raw.get_height()) )
	joined.blit(out, (0,0));
	pygame.image.save(out,"test_out.jpg")
	screen = pygame.display.set_mode((joined.get_width(), joined.get_height()))
	running=True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running=False
		sleep(1.0/60)
		screen.blit(joined,(0,0) )
		pygame.display.flip()



##### MAIN MIAN MASND (((((((((((((((((((((((((({{{{[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]}}}}))))))))))))))))))))))))))
# print command line arguments
args = []
for arg in sys.argv[1:]:
	args.append(arg)
fgImage = args[0]
bgImage = args[1]
main()
