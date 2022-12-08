# 取得聲音
CHANNELS=1 # 聲道數
RATE=12000 # 取樣頻率，11025、12000、22050、24000、44100(CD)、48000(DVD)
CHUNK=1024 # 紀錄聲音的樣本區塊大小
RECORD_SECONDS=1 # 錄音秒數
import pyaudio # python -m pip install PyAudio
FORMAT=pyaudio.paInt16 # 樣本格式，可為paFloat32、paInt32、paInt24、paInt16、paInt8、paUInt8、paCustomFormat

mic=pyaudio.PyAudio()
print('開始錄音...')

# 開啟錄音串流
stream=mic.open(format=FORMAT,channels=CHANNELS,rate=RATE,frames_per_buffer=CHUNK,input=True)

# 建立聲音串流
frames=[]
for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
    data=stream.read(CHUNK)
    frames.append(data) # 將聲音紀錄到串流中

# 關閉所有裝置
stream.stop_stream() # 停止錄音
stream.close() # 關閉串流
mic.terminate()
print('錄音結束...')

import wave
wf = wave.open('action.wav','wb') # 開啟聲音記錄檔
wf.setnchannels(CHANNELS) # 設定聲道
wf.setsampwidth(mic.get_sample_size(FORMAT)) # 設定格式
wf.setframerate(RATE) # 設定取樣頻率
wf.writeframes(b''.join(frames)) # 存檔
wf.close()

import librosa
y,sr=librosa.load('action.wav',sr=None) # y：音頻時間序列，sr：y的取樣頻率
y=y[::3] # 0,3,6,...
mfccs=librosa.feature.mfcc(y=y,sr=sr) # 計算梅爾頻率倒譜係數，[20,*]，20個特徵，*視聲音長度，不可大於20
import numpy as np
mfccs=np.pad(mfccs,pad_width=((0,0),(0,40-mfccs.shape[1])),mode='constant') # [20,40]，import numpy as np
np.save('1_2.npy',mfccs) # 儲存成.npy檔