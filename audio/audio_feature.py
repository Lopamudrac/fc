from cgi import test
from locale import normalize
import os
from xml.sax.handler import all_features
import pandas as pd
import numpy as np
from sklearn import feature_extraction
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim


#resNet model
def audio_model():
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    model = resnet18()
    #model=CNN()
    model.fc = nn.Linear(512,8)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(device)
    learning_rate = 2e-4
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return device,optimizer,loss_fn,learning_rate,model

# resNet ending here


#print(Ravdess_df.head(10))

# def plot_waveform(waveform, sr, title="Waveform"):
#     waveform = waveform.numpy()

#     num_channels, num_frames = waveform.shape
#     time_axis = torch.arange(0, num_frames) / sr

#     figure, axes = plt.subplots(num_channels, 1)
#     print("axex",axes)
#     axes.plot(time_axis, waveform[0], linewidth=1)
#     axes.grid(True)
#     figure.suptitle(title)
#     plt.show(block=False)


# def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
#     fig, axs = plt.subplots(1, 1)
#     axs.set_title(title or "Spectrogram (db)")
#     axs.set_ylabel(ylabel)
#     axs.set_xlabel("frame")
#     im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto") #Convert a power spectrogram (amplitude squared) to decibel (dB) units
#     fig.colorbar(im, ax=axs)
#     plt.show(block=False)


# def plot_fbank(fbank, title=None):
#     fig, axs = plt.subplots(1, 1)
#     axs.set_title(title or "Filter bank")
#     axs.imshow(fbank, aspect="auto")
#     axs.set_ylabel("frequency bin")
#     axs.set_xlabel("mel bin")
#     plt.show(block=False)

def stretch(data):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def feat_extraction(data,emo,sample_rate):
    #print("hi",emo)
    features=torch.tensor([])
    #SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(pth,normalize=True)
    
   # data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    #print('inp',pth,emo) 
    

    #spectogram starts here
    #plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform")
    #Audio(SPEECH_WAVEFORM.numpy(), rate=SAMPLE_RATE)
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels=128
    # Define transform
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    # Perform transform
   # spec = spectrogram(SPEECH_WAVEFORM)
   # features= np.hstack((spec,features))
    #print ('spec',features.shape)
    
    #plot_spectrogram(spec[0], title=emo)
    #spectogram ends here

    # Mel Spectrogram
    
   

    # mel_spectrogram = T.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     norm="slaney",
    #     onesided=True,
        
       
    # )
    mel_spectrogram1 = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            norm="slaney",
            onesided=True,
       
    )
    #melspec = mel_spectrogram(SPEECH_WAVEFORM)  #pure torch
    #melspec_librosa = librosa.feature.melspectrogram( # librosa feature, load  data from torchaudio 
    #
    libT=torch.from_numpy(librosa.feature.melspectrogram(y=data, sr=sample_rate)) # pure librosa, using this feature
   
    features= torch.hstack((torch.mean(libT.T, axis=0),features))
    #print("Features1 ",libT.shape,features.shape)
    #audT= mel_spectrogram1(torch.from_numpy(data)) #torch feature, load data from librosa  , using this feature
    
    # print(
    # #melspec.shape, melspec_librosa.shape, libT.shape, libT.T.shape, audT.shape, np.mean(melspec_librosa.T, axis=0).shape,
    #  torch.mean(libT.T, axis=0).shape, libT.shape
    # )
    # features= torch.hstack((torch.mean(libT.T, axis=0),features))
    # Mel Spectrogram ends here

    #MFCC starts here
    mfcc= torch.from_numpy(librosa.feature.mfcc(y=data, sr=sample_rate))
    features= torch.hstack((torch.mean(mfcc.T, axis=0),features))
    #print("Features2 ",mfcc.shape,features.shape)
    #MFCC ends here

    #chroma_stft
    chroma_stft = torch.from_numpy(librosa.feature.chroma_stft(y=data, sr=sample_rate))
    features= torch.hstack((torch.mean(chroma_stft.T, axis=0),features))
    #chroma ends 
    #print("Features3, ",chroma_stft.shape,features.shape)
    
    return features

def get_features():
    print("inside get features")
    RADVESS= "../DataSet/RADVESS/audio_speech_actors_01-24/"
    TESS="../DataSet/TESS/TESS Toronto emotional speech set data"

    radvess_dir_= os.listdir(RADVESS)
    tess_dir_= os.listdir(TESS)

    #print("radvess_dir",radvess_dir_)
    emotion_dict = {'01':'neutral', '02':'calm', "03":'happy', '04':'sad',
    '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}

    emotion=[]
    path=[]
    # calm=True, angry=True, neutral=True, '02':'calm', "03":'happy', '04':'sad',
    #  '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'

    for d in radvess_dir_:
        actors= os.listdir(RADVESS + d)
        for actor in actors:
            file_name= actor.split("-")
            emotion.append(emotion_dict[file_name[2]])
            path.append(RADVESS+d+'/'+actor)
    #print(path)
    emotion_df=pd.DataFrame(emotion, columns=['Emotion'])
    path_df=pd.DataFrame(path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df,path_df],axis=1)
    X, Y = [], []
    #feature_dataset=torch.tensor([])
    count = 0;
    for pth, emo in zip(Ravdess_df.Path, Ravdess_df.Emotion):
        data, sample_rate = librosa.load(pth,offset=0.6, duration=2.5 )
        feature = feat_extraction(data,emo,sample_rate)
        result =torch.tensor([])
        result=torch.hstack((result,feature))
        shift_data = shift(data)
        res2= feat_extraction(shift_data,emo,sample_rate)
        result= torch.vstack((result,res2))
        stretch_data = stretch(data)
        res3 = feat_extraction(stretch_data,emo,sample_rate)
        result = torch.vstack((result, res3))
        all_features=result #converted 2D to 3D
        #X.append(torch.reshape(all_features, (1, *tuple(all_features.size()))))
        #Y.append(emo)
        count +=1
        if count>40000:
            break

        for ele in all_features:
            X.append(torch.reshape(ele, (1,1, *tuple(ele.size()))))
                # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            Y.append(emo)
        #feature_dataset= torch.hstack((torch.tensor(X),feature_dataset))
        #feature_dataset= torch.hstack((torch.from_numpy(np.array(Y)),feature_dataset))
    return X,Y

def label_converter(y_data):
    emotion_dict = {'neutral':0,'calm':1, 'happy':2, 'sad':3,
    'angry':4, 'fear':5, 'disgust':6, 'surprise':7}
    labels=[]
    for y in y_data:
        #for y in y_list:
        labels.append(emotion_dict[y])
    return labels

def dataset_creator(x,y):
    #train_set,val_set,test_set = torch.utils.data.random_split(feature_dataset, [25000,11080, 10000],generator=torch.Generator().manual_seed(42))
   # val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=10, shuffle=False)
    # print("X,Y", len(X), len(Y),train_set, test_set, val_set)    

    # splitting data
    x_train,x_rem, y_train, y_rem = train_test_split(x, y, random_state=0,train_size=0.8, shuffle=True)
    # x_val,x_test, y_val, y_test = train_test_split(x_rem, y_rem, random_state=0,
    # train_size=0.5, shuffle=True)
    x_test=x_rem[0]
    y_test=y_rem[0]
    x_val=x_rem[1:]
    y_val=y_rem[1:]
    batch_size = 64
    training_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=False)
    training_y_loader = torch.utils.data.DataLoader(label_converter(y_train), batch_size=batch_size, shuffle=False)
    validation_y_loader = torch.utils.data.DataLoader(label_converter(y_val), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)
    test_y_loader = torch.utils.data.DataLoader(label_converter(y_test), batch_size=batch_size, shuffle=False)

    # print("X","Y",np.array(x_train).shape, np.array(y_train).shape,
    # np.array(x_val).shape, np.array(y_val).shape, np.array(x_test).shape, np.array(y_test).shape)
    #print("scikit", y_test,"input",x_test)
    # for (idx, batch) in enumerate(validation_y_loader):
    #     print("x",idx, batch)
    #for (idx, batch) in enumerate(validation_loader):
    #print("inp",idx, batch)
    #print("pyTorch", validation_loader, validation_y_loader)

    #print(resnet_model)

    return training_loader,training_y_loader,validation_loader,validation_y_loader,test_loader,test_y_loader
    #return x_train,y_train,x_val,y_val,x_test,y_test

def feature_extraction():
    torch.random.manual_seed(0)
    print("inside feature extraction function")
    X,Y= get_features()
    print("dataset_creator")
    training_loader,training_y_loader,validation_loader,validation_y_loader,test_loader,test_y_loader=dataset_creator(X,Y)
    device, optimizer,loss_fn,learning_rate,model= audio_model()
    return training_loader,training_y_loader,validation_loader,validation_y_loader,test_loader,test_y_loader,device, optimizer,loss_fn,learning_rate,model

# Features = pd.DataFrame(X)
# Features['labels'] = Y
# X = Features.iloc[: ,:-1].values
# Y = Features['labels'].values





