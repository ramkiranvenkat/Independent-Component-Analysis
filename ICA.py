import numpy as np
import matplotlib.pyplot as plt
import wave
import matplotlib.image as mpimg
import numpy.linalg as la
import os
import sys
import math

def readWavFile(filename):
	audio  = wave.open(filename,'r')
	params = audio.getparams()
	signal = audio.readframes(-1)
	#print(type(signal))
	signal = np.frombuffer(signal,'int16')
	signal = np.asarray(signal,dtype='double')
	audio.close()
	return signal, params, audio.getnframes()

def writeWavFile(data,filename,params,framecount):
	# saving audio file
	savefile = wave.open(filename,'wb')
	savefile.setparams(params)
	savefile.setnframes(framecount)
	wsignal = np.asarray(data,dtype='int16')
	savefile.writeframes(wsignal.tobytes()); #writeframes()
	savefile.close()

def plotAudio(data,title='title'):
	fig, ax = plt.subplots()
	#ax.plot(t, s)
	ax.plot(data)
	ax.set(xlabel='samples', ylabel='Signal', title=title)
	ax.grid()
	fig.savefig(title+".png")
	plt.show()

audio1, params1, frameCnt1 = readWavFile('../Data/wav/chase.wav')
#plotAudio(audio1,'audio file #1')
#frameCnt1Save = int(frameCnt1/2)
#audio1Save = audio1[:frameCnt1Save]
#print(audio1Save.size,frameCnt1Save)
#writeWavFile(audio1Save,'../Data/wav/TrumphetWrite.wav',params1,frameCnt1Save)

audio2, params2, frameCnt2 = readWavFile('../Data/wav/Trumphet.wav')
#plotAudio(audio2,'audio file #2')
#frameCnt2Save = int(frameCnt2/2)
#audio2Save = audio2[:frameCnt2Save]
#print(audio2Save.size,frameCnt2Save)
#writeWavFile(audio2Save,'../Data/wav/DrumWrite.wav',params2,frameCnt2Save)

# Mixing Signals
if frameCnt1<frameCnt2:
	frameCnt2 = frameCnt1
	audio2 = audio2[:frameCnt2]
	print(audio1.shape,audio2.shape)
else:
	frameCnt1 = frameCnt2
	audio1 = audio1[:frameCnt1]
	print(audio1.shape,audio2.shape)

maxa1 = np.max(np.array([abs(np.max(audio1)),abs(np.min(audio1))]))
maxa2 = np.max(np.array([abs(np.max(audio2)),abs(np.min(audio2))]))
nmax = max(maxa1,maxa2)
audio1 = audio1/maxa1*nmax
audio2 = audio2/maxa2*nmax

plotAudio(audio1,'Amp Normalized #1')
plotAudio(audio2,'Amp Normalized #2')

writeWavFile(audio1,'../Data/wav/norm1.wav',params1,frameCnt1)
writeWavFile(audio2,'../Data/wav/norm2.wav',params2,frameCnt2)

mix1 = 0.3*audio1 + 0.0*audio2
mix2 = 0.3*audio1 + 0.8*audio2

writeWavFile(mix1,'../Data/wav/mix1.wav',params1,frameCnt1)
writeWavFile(mix2,'../Data/wav/mix2.wav',params2,frameCnt2)

#ICA
m1 = np.mean(mix1)
m2 = np.mean(mix2)

#mix1max = np.max(mix1)
#mix1min = np.min(mix1)
#mix2max = np.max(mix2)
#mix2min = np.min(mix2)


#mean normalization
mix1mn = (mix1-m1)
mix2mn = (mix2-m2)

plotAudio(mix1mn,'Mix Mean Normalized #1')
plotAudio(mix2mn,'Mix Mean Normalized #2')

# Note these are PCA computations
#theta computation
theta = 0.5*math.atan(-2.0*np.sum(mix1mn*mix2mn)/np.sum(mix1mn*mix1mn-mix2mn*mix2mn))
Us = np.array([[math.cos(theta),math.sin(theta)],[-1.0*math.sin(theta),math.cos(theta)]])

#scaling
sig1 = np.sum((mix1mn*math.cos(theta) 		+ mix2mn*math.sin(theta))**2)
sig2 = np.sum((mix1mn*math.cos(theta-math.pi/2) + mix2mn*math.sin(theta-math.pi/2))**2)
Sigma = np.array([[1/math.sqrt(sig1),0],[0,1/math.sqrt(sig2)]])

#make probability separable
m1bar = (Us[0][0]*mix1mn + Us[0][1]*mix2mn)*Sigma[0][0]
m2bar = (Us[1][0]*mix1mn + Us[1][1]*mix2mn)*Sigma[1][1]

numerator   = -1.0*np.sum(2.0*(m1bar**3)*m2bar - 2.0*(m2bar**3)*m1bar)
denominator = np.sum(3.0*(m1bar**2)*(m2bar**2)-0.5*(m1bar**4)-0.5*(m2bar**4))
phi = 0.25*math.atan(numerator/denominator)

V = np.array([[math.cos(phi),math.sin(phi)],[-1.0*math.sin(phi),math.cos(phi)]])

s1bar = V[0][0]*m1bar+V[0][1]*m2bar
s2bar = V[1][0]*m1bar+V[1][1]*m2bar
s1bar = s1bar/np.max(abs(s1bar))*nmax + m1
s2bar = s2bar/np.max(abs(s2bar))*nmax + m2

plotAudio(s1bar,'ICA file #1')
plotAudio(s2bar,'ICA file #2')

writeWavFile(s1bar,'../Data/wav/ICA1.wav',params1,frameCnt1)
writeWavFile(s2bar,'../Data/wav/ICA2.wav',params2,frameCnt2)

