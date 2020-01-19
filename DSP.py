from __future__ import division  # so, 1/2 == 0.5 (forces real-valued division)
import struct
import scipy
import sounddevice as sd
from matplotlib import pyplot
from numpy.fft import fftfreq
from scipy import spatial
from scipy.fftpack import fft
from scipy.io.wavfile import write, read
from scipy.io import wavfile
import matplotlib.pyplot as plt
from playsound import playsound
import numpy as np
import os
import statistics
from scipy.spatial import distance
from numpy import linspace
import wave
import sys
import matplotlib

# initialize
seconds = 3  # duration of record
fs = 44000  # sample rate
# start recording for test and training data

for x in range(1,2):
    fileName = "test/GreenRed/%d.wav" % x
    print("start speaking for audio %d\n", x)
    myRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(fileName, fs, myRecording)

# # extract some features
# audioName = "train/off/off1.wav"  # Audio File
# rate, audioData = wavfile.read(audioName)  # extract sample frequency and audiodata
# print("Frequency sampling", rate)
# Len = len(audioData.shape)  # length of data
# audioLen = len(audioData)  # length of data
# print("length of audio", audioLen)
# print("Channels", Len)
# if Len == 2:
#     signal = audioData.sum(axis=1) / 2
# N = signal.shape[0]
# print("Complete Samplings N", N)
# # plt.plot(audioData) #plot  wave in time domain
# data = audioData.mean(axis=1)
# t = np.arange(data.shape[0]) / rate
# # plot everything
# plt.plot(t, data)
# plt.show()
#
# # fft
# # create some plots
# playsound('train/off/off1.wav')  # play the wave
# secs = N / float(rate)
# print("secs", secs)
# Ts = 1.0 / rate  # sampling interval in time
# print("Time step between samples Ts", Ts)
# t = scipy.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray
#
# FFT = abs(scipy.fft(signal))
# FFT_side = FFT[range(N // 2)]  # one side FFT range
# freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])
# fft_freqs = np.array(freqs)
# freqs_side = freqs[range(N // 2)]  # one side frequency range
# fft_freqs_side = np.array(freqs_side)
# plt.subplot(311)
# p1 = plt.plot(t, signal, "g")  # plotting the signal
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.subplot(312)
# p2 = plt.plot(freqs, FFT, "r")  # plotting the complete fft spectrum
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Count dbl-sided')
# plt.subplot(313)
# p3 = plt.plot(freqs_side, abs(FFT_side), "b")  # plotting the positive fft spectrum
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Count single-sided')
# plt.show()
#
# # Energy plot Em=∑n[s(n)w(m−n)]2
# assert fs % 1000 == 0
#
# sampsPerMilli = int(fs / 1000)
# millisPerFrame = 20
# sampsPerFrame = sampsPerMilli * millisPerFrame
# nFrames = int(len(audioData) / sampsPerFrame)  # number of non-overlapping _full_ frames
#
# print('samples/millisecond  ==> ', sampsPerMilli)
# print('samples/[%dms]frame  ==> ' % millisPerFrame, sampsPerFrame)
# print('number of frames     ==> ', nFrames)
#
# # process eaxh frame in turn
# STEs = []  # list of short-time energies
# for k in range(nFrames):
#     startIdx = k * sampsPerFrame
#     stopIdx = startIdx + sampsPerFrame
#     window = np.zeros(audioData.shape)
#     window[startIdx:stopIdx] = 1  # rectangular window
#     STE = sum((audioData ** 2) * (window ** 2))
#     STEs.append(STE)
# print('list of energies    ==> ', STEs)
# plt.plot(STEs)
# plt.title('Short-Time Energy')
# plt.ylabel('ENERGY')
# plt.xlabel('FRAME')
# pyplot.autoscale(tight='both')
# plt.show()


# --------Training Data --------

offFiles = []  # list of off files
onFiles = []  # list of on files
testFiles = []  # list of test files
testFilesG = []
onEnergy = []  # array of on files energy
offEnergy = []  # array of on files energy
on_ZCR_array = []  # list of  5 zero crossing rates of on files
off_ZCR_array = []  # list of 5 zero crossing rates of off files
on_avg_ZCR_array = []  # list of  5 zero crossing rates of on files
off_avg_ZCR_array = []  # list of 5 zero crossing rates of off files
on_features_array = []  # list of features of on files
off_features_array = []  # list of features of off files
on_avg_energy = 0
off_avg_energy = 0
fileName = ""

redOnFiles = []
redOnEnergy = []
redOn_ZCR_array = []
redOn_features_array = []
redOn_avg_ZCR_array = []
redOn_avg_energy = 0

redOffFiles = []
redOffEnergy = []
redOff_ZCR_array = []
redOff_features_array = []
redOff_avg_ZCR_array = []
redOff_avg_energy = 0

greenOnFiles = []
greenOnEnergy = []
greenOn_ZCR_array = []
greenOn_features_array = []
greenOn_avg_ZCR_array = []
greenOn_avg_energy = 0

greenOffFiles = []
greenOffEnergy = []
greenOff_ZCR_array = []
greenOff_features_array = []
greenOff_avg_ZCR_array = []
greenOff_avg_energy = 0

# -------------- Energy calculation---------------------
# Off
offPath = "train/off/"
for path in os.listdir(offPath):
    off_full_path = os.path.join(offPath, path)
    if os.path.isfile(off_full_path):
        offFiles.append(off_full_path)
for i in range(len(offFiles)):
    sr, data = wavfile.read(offFiles[i])  # set sampling rate to 44000
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    print("OFF energy", e)
    offEnergy.append(e)
# average Energy
off_avg_energy = np.mean(offEnergy)
print("OFF average energy ", off_avg_energy)
# On
onPath = "train/on/"
for path1 in os.listdir(onPath):
    on_full_path = os.path.join(onPath, path1)
    if os.path.isfile(on_full_path):
        onFiles.append(on_full_path)
for i in range(len(onFiles)):
    sr, data = wavfile.read(onFiles[i])  # set sampling rate to 44000
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    print("ON energy array", e)
    onEnergy.append(e)
# average Energy
on_avg_energy = np.mean(onEnergy)
print("ON average energy ", on_avg_energy)

# ---------Zero Crossing Rate taking 5 frames for each signal-----
print(onFiles)
print(offFiles)
test_ZCR_array = []
# on
for i in range(len(onFiles)):
    sr, data = wavfile.read(onFiles[i])
    length = len(data)
    ZCR1_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    ZCR_array = [ZCR1_on, ZCR2_on, ZCR3_on, ZCR4_on, ZCR5_on]
    on_ZCR_array.append(ZCR_array)
on_avg_ZCR_array = np.mean(on_ZCR_array, axis=0)
print(on_avg_ZCR_array)

# off
for i in range(len(offFiles)):
    sr, data = wavfile.read(offFiles[i])
    length = len(data)
    ZCR1_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    ZCR_array2 = [ZCR1_off, ZCR2_off, ZCR3_off, ZCR4_off, ZCR5_off]
    off_ZCR_array.append(ZCR_array2)
off_avg_ZCR_array = np.mean(off_ZCR_array, axis=0)
print(off_avg_ZCR_array)

# ---------------merging features in features array -----------------
# off word
for i in range(len(offFiles)):
    sr, data = wavfile.read(offFiles[i])
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    #print("OFF energy", e)
    offEnergy.append(e)
    ZCR1_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_off = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    features = [ZCR1_off, ZCR2_off, ZCR3_off, ZCR4_off, ZCR5_off, e]
    off_features_array.append(features)

off_features_array = np.mean(off_features_array, axis=0)
print("Features array of the word off ", off_features_array)

# on
for i in range(len(onFiles)):
    sr, data = wavfile.read(onFiles[i])
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    ZCR1_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_on = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    features = [ZCR1_on, ZCR2_on, ZCR3_on, ZCR4_on, ZCR5_on, e]
    on_features_array.append(features)

on_features_array = np.mean(on_features_array, axis=0)
print("Features array of the word on ", on_features_array)

# -----------------------red On-----------------------------
redOnPath = "train/redOn/"
for path in os.listdir(redOnPath):
    redOn_full_path = os.path.join(redOnPath, path)
    if os.path.isfile(redOn_full_path):
        redOnFiles.append(redOn_full_path)
for i in range(len(redOnFiles)):
    sr, data = wavfile.read(redOnFiles[i])
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    print("red energy", e)
    redOnEnergy.append(e)
    ZCR1_redOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_redOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_redOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_redOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_redOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    featuresRedOn = [ZCR1_redOn, ZCR2_redOn, ZCR3_redOn, ZCR4_redOn, ZCR5_redOn, e]
    redOn_features_array.append(featuresRedOn)

redOn_features_array = np.mean(redOn_features_array, axis=0)
#print("Features array of red On ", redOn_features_array)
# ------------------- red Off-----------------------------
redOffPath = "train/redOff/"
for path in os.listdir(redOffPath):
    redOff_full_path = os.path.join(redOffPath, path)
    if os.path.isfile(redOff_full_path):
        redOffFiles.append(redOff_full_path)
for i in range(len(redOffFiles)):
    sr, data = wavfile.read(redOffFiles[i])
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    print("red off energy", e)
    redOffEnergy.append(e)
    ZCR1_redOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_redOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_redOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_redOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_redOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    featuresRedOff = [ZCR1_redOff, ZCR2_redOff, ZCR3_redOff, ZCR4_redOff, ZCR5_redOff, e]
    redOff_features_array.append(featuresRedOff)

redOff_features_array = np.mean(redOff_features_array, axis=0)
#print("Features array of red Off ", redOff_features_array)

# -------------- green on -------------------
greenOnPath = "train/greenOn/"
for path in os.listdir(greenOnPath):
    greenOn_full_path = os.path.join(greenOnPath, path)
    if os.path.isfile(greenOn_full_path):
        greenOnFiles.append(greenOn_full_path)
for i in range(len(greenOnFiles)):
    sr, data = wavfile.read(greenOnFiles[i])
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    print("green On energy", e)
    greenOnEnergy.append(e)
    ZCR1_greenOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_greenOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_greenOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_greenOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_greenOn = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    features = [ZCR1_greenOn, ZCR2_greenOn, ZCR3_greenOn, ZCR4_greenOn, ZCR5_greenOn, e]
    greenOn_features_array.append(features)

greenOn_features_array = np.mean(greenOn_features_array, axis=0)
#print("Features array of the word green On", greenOn_features_array)

# ---------------------green Off -----------------------
greenOffPath = "train/greenOff/"
for path in os.listdir(greenOffPath):
    greenOff_full_path = os.path.join(greenOffPath, path)
    if os.path.isfile(greenOff_full_path):
        greenOffFiles.append(greenOff_full_path)
for i in range(len(greenOffFiles)):
    sr, data = wavfile.read(greenOffFiles[i])
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    print("green Off energy", e)
    greenOffEnergy.append(e)
    ZCR1_greenOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_greenOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_greenOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_greenOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_greenOff = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    features = [ZCR1_greenOff, ZCR2_greenOff, ZCR3_greenOff, ZCR4_greenOff, ZCR5_greenOff, e]
    greenOff_features_array.append(features)

greenOff_features_array = np.mean(greenOff_features_array, axis=0)
#print("Features array of the word green Off", greenOff_features_array)
# --------------------------------------------TEST DATA----------------------------------------------
testPath = "test/OnOff"
for path3 in os.listdir(testPath):
    test_full_path = os.path.join(testPath, path3)
    if os.path.isfile(test_full_path):
        testFiles.append(test_full_path)
# classified based on energy
# for i in range(len(testFiles)):
#     sr, data = wavfile.read(testFiles[i])  # set sampling rate to 44000
#     length = len(data)
#     e = 0
#     for j in range(1, length):
#         e = e + (data[j] * data[j])
#     e = e / length
#     e = np.mean(e)
#     if abs(e - on_avg_energy) < abs(e - off_avg_energy):
#         print("file classified as ON based on energy ", testFiles[i])
#     elif abs(e - on_avg_energy) > abs(e - off_avg_energy):
#         print("file classified as off based on energy ", testFiles[i])

# classified based on ZCR
test_ZCR_array = []
# for i in range(len(testFiles)):
#     sr, data = wavfile.read(testFiles[i])
#     length = len(data)
#     ZCR1_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
#     ZCR2_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
#     ZCR3_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
#     ZCR4_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
#     ZCR5_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
#     test_ZCR_array = [ZCR1_test, ZCR2_test, ZCR3_test, ZCR4_test, ZCR5_test]
#     if distance.euclidean(test_ZCR_array, on_avg_ZCR_array) < distance.euclidean(test_ZCR_array, off_avg_ZCR_array):
#         print("file classified as ON based on ZCR ", testFiles[i])
#     elif distance.euclidean(test_ZCR_array, on_avg_ZCR_array) > distance.euclidean(test_ZCR_array, off_avg_ZCR_array):
#         print("file classified as off based on ZCR ", testFiles[i])

# -----On and Off classified based on ZCR and energy------
for i in range(len(testFiles)):
    sr, data = wavfile.read(testFiles[i])  # set sampling rate to 44000
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    ZCR1_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    test_features = [ZCR1_test, ZCR2_test, ZCR3_test, ZCR4_test, ZCR5_test, e]
    if spatial.distance.cosine(on_features_array, test_features) < spatial.distance.cosine(off_features_array,
                                                                                           test_features):
        print("file classified as ON based on ZCR and energy ", testFiles[i])
    elif spatial.distance.cosine(on_features_array, test_features) > spatial.distance.cosine(off_features_array,
                                                                                             test_features):
        print("file classified as off based on ZCR and energy ", testFiles[i])

#---------Green and red , on and off ------------------
testPathG = "test/GreenRed"
for pathG in os.listdir(testPathG):
    testG_full_path = os.path.join(testPathG, pathG)
    if os.path.isfile(testG_full_path):
        testFilesG.append(testG_full_path)
for k in range(len(testFilesG)):
    sr, data = wavfile.read(testFilesG[k])  # set sampling rate to 44000
    length = len(data)
    e = 0
    for j in range(1, length):
        e = e + (data[j] * data[j])
    e = e / length
    e = np.mean(e)
    ZCR1_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[:len(data) // 5]))))
    ZCR2_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 5):int(len(data) // 2.5)]))))
    ZCR3_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 2.5):int(len(data) // 1.7)]))))
    ZCR4_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.7):int(len(data) // 1.25)]))))
    ZCR5_test = 0.5 * np.mean(np.abs(np.diff(np.sign(data[int(len(data) // 1.25):len(data)]))))
    testG_features = [ZCR1_test, ZCR2_test, ZCR3_test, ZCR4_test, ZCR5_test, e]
    if (spatial.distance.cosine(redOn_features_array, testG_features) < spatial.distance.cosine(redOff_features_array,testG_features)) \
            and (spatial.distance.cosine(redOn_features_array, testG_features) < spatial.distance.cosine(greenOff_features_array,testG_features))\
            and (spatial.distance.cosine(redOn_features_array, testG_features) < spatial.distance.cosine(greenOn_features_array,testG_features)):
        print("file classified as RED ON based on ZCR and energy ", testFilesG[k])
    elif (spatial.distance.cosine(redOff_features_array, testG_features) < spatial.distance.cosine(redOn_features_array,testG_features)) \
            and (spatial.distance.cosine(redOff_features_array, testG_features) < spatial.distance.cosine(greenOff_features_array,testG_features))\
            and (spatial.distance.cosine(redOff_features_array, testG_features) < spatial.distance.cosine(greenOn_features_array,testG_features)):
        print("file classified as RED OFF based on ZCR and energy ", testFilesG[k])
    elif (spatial.distance.cosine(greenOff_features_array, testG_features) < spatial.distance.cosine(redOn_features_array,testG_features)) \
            and (spatial.distance.cosine(greenOff_features_array, testG_features) < spatial.distance.cosine(redOff_features_array,testG_features))\
            and (spatial.distance.cosine(greenOff_features_array, testG_features) < spatial.distance.cosine(greenOn_features_array,testG_features)):
        print("file classified as GREEN OFF based on ZCR and energy ", testFilesG[k])
    elif (spatial.distance.cosine(greenOn_features_array, testG_features) < spatial.distance.cosine(redOn_features_array,testG_features)) \
            and (spatial.distance.cosine(greenOn_features_array, testG_features) < spatial.distance.cosine(redOff_features_array,testG_features))\
            and (spatial.distance.cosine(greenOn_features_array, testG_features) < spatial.distance.cosine(greenOff_features_array,testG_features)):
        print("file classified as GREEN ON based on ZCR and energy ", testFilesG[k])
    else :
        print("not classified" , testFilesG[k])