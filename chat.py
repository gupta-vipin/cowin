import winsound
import operator

import string

import random
import json

import torch
import datetime
import wolframalpha


from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import requests

urljson = "http://waymorecreations.com/jsonFile.txt"
r = requests.get(urljson)
with open('jsonFile.json', 'wb') as f:
    f.write(r.content)


#import train
import os

from bs4 import BeautifulSoup
import platform

#import pyttsx3
import azure.cognitiveservices.speech as speechsdk

from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from facenet_pytorch import MTCNN, InceptionResnetV1
#import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import shutil,os



speech_key, service_region = "c28b8c900af745feab0bdda34977366b", "centralindia"

audio_config = AudioOutputConfig(use_default_speaker=True)

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

app_id = 'RKWQR9-K8A4VKWYEV'
client = wolframalpha.Client(app_id)


#engine = pyttsx3.init('sapi5')
#voice = engine.getProperty('voices')
#engine.setProperty('voice', voice[0].id)
#rate = engine.getProperty('rate')   # getting details of current speaking rate
#print (rate)                        #printing current voice rate
#engine.setProperty('rate', 125)     # setting up new voice rate


def speak(audio):
    #engine.say(audio)
    #print(audio)
    #engine.runAndWait()
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(audio)
    
def img_capture():
    vid = cv2.VideoCapture(0)
    ret,frame = vid.read()
    cv2.imwrite("NewPicture.jpg",frame)
    result = False
    vid.release()
    cv2.destroyAllWindows()




# os.system('python train.py')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder('photos') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
        embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list
        
data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file

def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load('data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))


def recognise():
    result = face_match('NewPicture.jpg', 'data.pt')
    if(result[1]>1):
        print("No match found in database")
        name = input("Enter your name: ")
        
        parent_dir = r"C:\Users\hp\Desktop\Deeplearning\pytorch_face_recognition\photos"
        path = os.path.join(parent_dir, name)
        mode = 0o666
        try:
            os.makedirs(path, exist_ok = True)
            print("Directory '%s' created successfully" % path)
        except OSError as error:
            print("Directory '%s' can not be created" % path)
        
        print(path)
        file = r'C:\Users\hp\Desktop\Deeplearning\pytorch_face_recognition\NewPicture.jpg'
        shutil.copy(file, path)
        return name
    
    else:
        print('Face matched with: ',result[0], 'With distance: ',result[1])
        return result[0]
     
    

#playsound('startup.wav')
winsound.PlaySound('startup.wav', winsound.SND_FILENAME)       

def wishMe():
    hour=datetime.datetime.now().hour
    if hour>=0 and hour<12:
        speak("Good Morning")
        print("Good Morning")
    elif hour>=12 and hour<18:
        speak("Good Afternoon")
        print("Good Afternoon")
    else:
        speak("Good Evening")
        print("Good Evening")



WAKE1 = "hey blueberry"
WAKE2 = "ok blueberry"
WAKE3 = "oye blueberry"
WAKE4 = "sun blueberry"
WAKE5 = "blueberry"


if __name__ == "__main__":
    print("Let's talk!")
    wishMe()
   

    while True:
        sentence = speech_recognizer.recognize_once()
        if sentence.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(sentence.text))
            sentence = sentence.text
            print(sentence)

            if (sentence.count(WAKE1) or sentence.count(WAKE2) or sentence.count(WAKE3) or sentence.count(WAKE4) or sentence.count(WAKE5)) > 0:
                winsound.PlaySound('trigger.wav', winsound.SND_FILENAME) 
                img_capture()
                person = recognise()
                
                speak(person)
                sentence1 = speech_recognizer.recognize_once()
                if sentence1.reason == speechsdk.ResultReason.RecognizedSpeech:
                    print("Recognized: {}".format(sentence1.text))
                    sentence1 = sentence1.text
                
                    question = sentence1
                    sentence1 = tokenize(sentence1)
                    X = bag_of_words(sentence1, all_words)
                    X = X.reshape(1, X.shape[0])
                    X = torch.from_numpy(X).to(device)

                    output = model(X)
                    _, predicted = torch.max(output, dim=1)

                    tag = tags[predicted.item()]

                    probs = torch.softmax(output, dim=1)
                    prob = probs[0][predicted.item()]
                    
                    if prob.item() > 0.99:
                        for intent in intents['intents']:
                            if tag == intent["tag"]:
                                speak(random.choice(intent['responses']))
                                print(random.choice(intent['responses']))

                    else: 
                        # res = client.query(f'{question}')
                        # answer = next(res.results).text
                        # #speak(answer)
                        # speak(f'{answer}')
                        URL = "https://www.google.co.in/search?q=" + question
                        headers = {
                        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                        }

                        page = requests.get(URL, headers=headers)
                        soup = BeautifulSoup(page.content, 'html.parser')
                        #answer = soup.find(class_='VwiC3b yXK7lf MUxGbd yDYNvb lyLwlc lEBKkf').get_text()
                        #wwUB2c PZPZlf
                        #answer = ""
                        #answer2 = ""
                        #print(soup)
                        
                        #VwiC3b yXK7lf MUxGbd yDYNvb lyLwlc lEBKkf
                        try: 
                            answer = soup.find(class_='Z0LcW XcVN5d').get_text()
                            speak(answer)
                            print(answer)
                        except AttributeError:
                            try:
                                answer = soup.find(class_='PZPZlf hb8SAc').get_text()
                                speak(answer)
                                print(answer)
                            except AttributeError: 
                                try:
                                    answer = soup.find(class_='zCubwf').get_text()
                                    speak(answer)
                                    print(answer)
                                except AttributeError: 
                                    try:
                                        answer = soup.find(class_='gsrt vk_bk dDoNo FzvWSb XcVN5d DjWnwf').get_text()
                                        speak(answer)
                                        print(answer)
                                    except AttributeError: 
                                        try:
                                            temp = soup.find(class_='wob_t TVtOme').get_text()
                                            mood = soup.find(class_='wob_dcp').get_text()
                                            phw = soup.find(class_='wtsRwe').get_text()
                                            print(temp)
                                            speak(temp)
                                            print(mood)
                                            speak(mood)
                                            print(phw)
                                            speak(phw)
                                        except AttributeError: 
                                            try:
                                                answer = soup.find(class_='wDYxhc NFQFxe viOShc LKPcQc').get_text()
                                                speak(answer)
                                                print(answer)
                                            except AttributeError: 
                                                try:
                                                    tag = soup.find(class_='co8aDb XcVN5d').get_text()
                                                    answer = soup.find(class_='RqBzHd').get_text()
                                                    
                                                    print(tag)
                                                    speak(tag)
                                                    print(answer)
                                                    speak(answer)
                                                except AttributeError: 
                                                    try:
                                                        answer = soup.find(class_='vk_bk dDoNo FzvWSb XcVN5d').get_text()
                                                        speak(answer)
                                                        print(answer)
                                                    except AttributeError: 
                                                        try:
                                                            answer = soup.find(class_='VwiC3b yXK7lf MUxGbd yDYNvb lyLwlc lEBKkf').get_text()
                                                            speak(answer)
                                                            print(answer)
                                                        except AttributeError:
                                                            pass
                                                        
                            try:
                                res = client.query(question)
                                answer = next(res.results).text
                                speak(answer)
                                print(answer)
                            except:
                                speak("Sorry I did not understand what you said, please try again")
                                pass
                            #finally:
                                #pass
                            '''speak("Sorry I did not understand what you said, please try again")
                            #pass'''

                                                            
                            winsound.PlaySound('complete.wav', winsound.SND_FILENAME)

                            
                        

        