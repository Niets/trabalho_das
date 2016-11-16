
# coding: utf-8

# # Caminhos iniciais

# In[1]:

haarcascade_human = "/home/niets/Documents/Install-OpenCV/Ubuntu/OpenCV/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_alt2.xml"
haarcascade_cat = "/home/niets/Documents/Install-OpenCV/Ubuntu/OpenCV/opencv-2.4.13/data/haarcascades/haarcascade_frontalcatface_extended.xml"


# # Imports Iniciais

# In[2]:

import numpy as np
import cv2
import sys
import os
import json
import pickle
from scipy import misc

import matplotlib.pyplot as plt

# # Iniciando o Caffe

# In[3]:

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
home_dir = os.getenv("HOME")
caffe_root = os.path.join(home_dir, 'caffe')  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.


# In[4]:

if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system('~/caffe/scripts/download_model_binary.py ~/caffe/models/bvlc_reference_caffenet')


# In[5]:

caffe.set_mode_cpu()

model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# In[6]:

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# In[7]:

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


# In[8]:

# load ImageNet labels
labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
if not os.path.exists(labels_file):
    os.system(u'~/caffe/data/ilsvrc12/get_ilsvrc_aux.sh')

labels = np.loadtxt(labels_file, str, delimiter='\t')


# # Definir Classify

# In[9]:

def classify(image_filename):
    image = caffe.io.load_image(image_filename)
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

    return zip(output_prob[top_inds], labels[top_inds])


# # Funcoes Detect

# In[10]:

def detect_human(frame):
    height, width, depth = frame.shape

    # create grayscale version
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # equalize histogram
    cv2.equalizeHist(grayscale, grayscale)

    # detect objects
    classifier = cv2.CascadeClassifier(haarcascade_human)
    DOWNSCALE = 4
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    humans = 0
    if len(faces)>0:
        for i in faces:
            x, y, w, h = [ v*DOWNSCALE for v in i ]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
            humans = humans + 1
    print "Number of humans detected: " + str(humans)
    faces = np.multiply(faces,4)
    print faces
    return frame

def detect_cat(frame):
    height, width, depth = frame.shape

    # create grayscale version
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # equalize histogram
    cv2.equalizeHist(grayscale, grayscale)

    # detect objects
    classifier = cv2.CascadeClassifier(haarcascade_cat)
    DOWNSCALE = 4
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    cats = 0
    if len(faces)>0:
        for i in faces:
            x, y, w, h = [ v*DOWNSCALE for v in i ]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
            cats = cats + 1
    print "Number of cats detected: " + str(cats)
    faces = np.multiply(faces,4)
    print faces
    return frame


# ## Webcam

# In[11]:

def webcam():

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = frame.copy()
        # Call the function
        frame = detect_human(frame)
        #frame = detect_cat(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# ## Baixar Imagem

# In[12]:

def imagem(path):
    os.system(u'wget -O image.jpg $path')


# # Escolha (Webcam - Imagem)

# In[92]:

escolha = input('Digite: \n1 Para acessar a Webcam\n2 Para escolher a imagem da internet\n3 Para escolher uma imagem local\n')

if escolha == 1:
    webcam()
    path = 'image.png'
    #magem(path)

if escolha == 2:
    path = raw_input('Digite a URL da imagem: ')
    imagem(path)
    path = 'image.jpg'

if escolha == 3:
    path = raw_input('Digite o nome da imagem: ')


# # Verificar Imagem - OpenCV

# In[93]:

img = misc.imread(path)
faces = detect_cat(img.copy())
faces = detect_human(faces.copy())

plt.figure()
plt.imshow(faces)


# # Verificar Imagem - Caffe

# In[94]:

image = caffe.io.load_image(os.path.join(home_dir, 'Documents', 'DAS', str(path)))
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
plt.axis('on')

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


# In[95]:

# load ImageNet labels
labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
if not os.path.exists(labels_file):
    os.system(u'~/caffe/data/ilsvrc12/get_ilsvrc_aux.sh')

labels = np.loadtxt(labels_file, str, delimiter='\t')


# In[96]:

print 'output label:', labels[output_prob.argmax()]


# In[97]:

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])


# # Probabilidade Caninos & Felinos
#
#
# #### * Caninos ficam na lista do Caffe entre: [151:280]
# #### * Felinos ficam na lista do Caffe entre: [281:293]

# In[98]:

prob_cat = output_prob[281:293].argmax()
prob_cat = int(prob_cat) + 281

print 'Probability of the picture containing a cat is: ' + str(round(output_prob[prob_cat]*100,0))+'%'+'\n'

# ------
prob_dog = output_prob[151:280].argmax()
prob_dog = int(prob_dog) + 151

print 'Probability of the picture containing a dog is: ' + str(round(output_prob[prob_dog]*100,0))+'%'
