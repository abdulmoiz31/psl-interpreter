from testing.models import *
from .serializers import *
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import APIView
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import FileSystemStorage
import requests
import os
import time
import cv2
import testing
import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import KFold



class CountryView(APIView):
    #global sentences dictionary
    sentences_dict = {0:'None',1:'آج کیا تاریخ ہے', 2:'پولیس سکول میں جا رہی ہے', 3:'پاکستان میرا پیارا وطن ہے', 4:'واش روم کدھر ہے', 
                    5:'میری دو بہنیں ہیں',6:'میرے دو بھائ ہیں', 7:'میں طالب علم ہوں', 8:'مجھے راستہ بھول گیا ہے',
                    9:'کیا وقت ہوا ہے', 10:'بلی کو بھوک لگی ہے', 11:'اس بچے کو پانی چاہیے', 12:'آپ کا شکریہ', 
                    13:'آپ سے مل کر اچھا لگا', 14:'مجھے میرے امی ابو سے پیار ہے', 15:'میرے دوست کو چاکلیٹ پسند ہے', 
                    16:'میری بہن نے ٹرین نہیں دیکھی', 17:'براہ مہربانی مجھے جانے دیں', 18:'بھائ کمپیوٹر خریدنے کے لیے پیسے دیں', 
                    19:'ہم ملتان سے لاھور ٹرین میں گۓ', 20:'مجھے اب جانا چاہیے'}




    current_index = 0
    api_keys = ['NGwBiSug2jU9VtBPDSh3Uq3T','smQxGrccRwcAfhSX4J8dZoGJ','t4nfz82nZceR2ShZtQf4sJCk','vbqghvUMaQWejoJdM5eMuHPM','J1c76vFzMRUxkKtw8Gjx5LG8','uyyxmrVovaAbashxVa1Bx411','gooNMDBhJpnCHrjjwyzgZweh','CALehH5ekNSydSxiVZbqPpCS','qYnB1NAYCA5Ycezb84LR7xZZ','u8kiYiKfhJJzLBR87LFWZpkm','ZRrQLSwiyAb922AG322ppyRt','nwnVtTzEKTNqVPFj5q52wmYA','tBbMSbanQTwfoz8B1gB9Azvg','mB3jsSeurpDrB3zvmb86D693','75qbGDc4jxWRVeEuWyW3BjHt','FHZXnGUVrR69uecmjDMdUftf','JvpHtx7FtdEoMdZPbrVJPg42','auQdbFXvsZasCeyyhLpA4sEo','sWZntTLm7PGLdVpxoxoLQZN5']                                                                                                
    used = [True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
    parser_classes = (MultiPartParser,)
    files_processed = 0

    def remove_background(self):
        path = os.path.dirname(testing.__file__)
        path = path + "\\frames\\"

        dest_path = os.path.dirname(testing.__file__)
        dest_path = dest_path + "\\bgremoved\\"

        

        dir = os.listdir(path)
        self.files_processed = self.files_processed + len(dir)
        for x in range(len(self.used)):
            
            if self.used[x] == False:
                self.current_index = x
                break
        #if self.files_processed>45:
        #    self.files_processed = 0
        #    self.used[self.current_index] = True
        #    self.current_index = self.current_index + 1
        
        count = 0
        for x in range(len(dir)):
            image = dir[x]
            if count > 10:
                count = 0
                print("sleep")
                time.sleep(37)  # Sleep for 10 seconds
                print("awake")

            print(image)
            count+=1
            response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(path + str(image), 'rb')},
            data={'size': 'auto'},
            headers={'X-Api-Key': self.api_keys[self.current_index]},
             )
            if response.status_code == requests.codes.ok:
                with open(dest_path + '/' + str(image[:-4]+'.png'), 'wb') as out:
                    out.write(response.content)
            elif response.status_code == 402:
                    self.current_index+=1
                    self.used[self.current_index] = True
                    x-=1

            else:
                print("Error:", response.status_code, response.text)
                
                
                
                
    def cleanDirectories(self):
        pth = os.path.dirname(testing.__file__)
        basepath = pth + "\\frames"

        files = os.listdir(basepath)
        for f in files:
            os.remove(basepath+"\\"+f)

        
    


    
    def getFrame(self,sec, vidcap, count):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames, image = vidcap.read()
        #print("Here")
        if hasFrames:
            #image = cv2.resize(image, dsize=(120,120))              #resize image to 120x120
            

            pth = os.path.dirname(testing.__file__)
            basepath = pth + "\\frames\\"
            #print(basepath)
            
            
            filename = basepath + "%d.jpg" % count
            cv2.imwrite(filename, image)    # save frame as JPG file

        return hasFrames





    def createFrames(self,video):
        #print(video)
        
        self.cleanDirectories()

        vidcap = cv2.VideoCapture(str(video))
        sec = 0
        frameRate = 1/6  # it will capture image in each 1/6th second
        count = 1
            
        success = self.getFrame(sec, vidcap, count)
        
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = self.getFrame(sec, vidcap, count)



    def toNumpy(self,dir):
        img_array = np.empty([0,43200], np.uint8, 'C')
        print('Converting frames to Numpy array...')

        dirs = os.listdir(dir)
        for img in dirs:
        #print(img)
            path = dir+"/"+img

            if path.endswith(".jpg"):
                I = cv2.imread(path)
                I = cv2.resize(I, dsize=(120,120))
                im = (np.array(I))
                r = im[:, :, 0].flatten()
                g = im[:, :, 1].flatten()
                b = im[:, :, 2].flatten()
                a = [list(r) + list(g) + list(b)]
                b = np.asarray(a)
                img_array = np.append(img_array, b, axis=0)
        
        print('Reshaping array...')
        img_array = img_array.reshape(img_array.shape[0], 120, 120, 3)
        print('Normalizing array...')
        img_array = img_array.astype('float32')
        img_array = img_array/ 255.

        return img_array

    


    def predictSentence(self,frames_dir):

        #preprocess frames
        images_array = self.toNumpy(frames_dir)
        
        #define CNN model
        sign_model = Sequential()
        sign_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(120,120,3)))
        sign_model.add(LeakyReLU(alpha=0.1))
        sign_model.add(MaxPooling2D((2, 2),padding='same'))
        sign_model.add(Dropout(0.4))
        sign_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        sign_model.add(LeakyReLU(alpha=0.1))
        sign_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        sign_model.add(Dropout(0.4))
        sign_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        sign_model.add(LeakyReLU(alpha=0.1))                  
        sign_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        sign_model.add(Dropout(0.4))
        sign_model.add(Flatten())
        sign_model.add(Dense(128, activation='linear'))
        sign_model.add(LeakyReLU(alpha=0.1))           
        sign_model.add(Dropout(0.4))
        sign_model.add(Dense(21, activation='softmax'))

        # loading the trained weights
        model_path = os.path.dirname(testing.__file__)  + "\\cnnmodel\\"
        sign_model.load_weights(model_path + "CNN_weights_updated.hdf5")

        # compiling the model
        sign_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        
        #predict the video
        print('Pridicting Sentence...')
        predictions = sign_model.predict(images_array)
        predictions = np.argmax(np.round(predictions),axis=1)
        print(predictions)
        #print(predictions.shape)

        #maximum predicted sentence
        sentence_key = np.bincount(predictions).argmax()

        #take sentence from dictionary using predicted key
        
        
        sentence = self.sentences_dict[sentence_key]
        
        return 'Predicted sentence is: ' + sentence + ' key: ' + str(sentence_key) + ' predictions: ' + str(predictions)




    def post(self, request):
        #print("In Post")
        data = request.FILES['file']
        #print(data.file)

        fs = FileSystemStorage()
        filename = fs.save(data.name, data)
        fs.url(filename)

        #print(uploaded_file_url)
        #print(data.name)
        #print(data.file)
        #print("Here")
        self.createFrames(data)
        
        #self.remove_background()
        
        if os.path.exists(filename):
            os.remove(filename)

        frames_dir_path = os.path.dirname(testing.__file__)  + "\\frames"
         
        
        print(frames_dir_path)
        result = self.predictSentence(frames_dir_path)
        print(result)


        return Response(result)



    
def home(request,pk=None):
    if pk==None:
        instances = Country.objects.all()

            
        serializer = CountrySerializer(instances, many=True)
        return JsonResponse(serializer.data)

    try:
        instance = Country.objects.get(name=pk)
    except:
        return Response(status=status.HTTP_404_NOT_FOUND)

    serializer = CountrySerializer(instance)
    return Response(data=serializer.data, status=status.HTTP_200_OK)