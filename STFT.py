# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

def STFT_single(x, framesamp=512):
    """
    Short-time Fourier Transform for single channel signal,
    where blocks are windowed with a cosine window.
    
    framesamp must be even.
    framesamp should be a power of 2 to be efficient.
    """
    hop = int(framesamp/2)
    x = np.concatenate((np.zeros(hop), x, np.zeros(framesamp)))
    w = np.sin(np.pi*((2*np.arange(framesamp)+1)/(2.0*framesamp)))
    idx = range(0, len(x)-framesamp, hop)
    X = np.array([np.fft.rfft(w*x[i:i+framesamp]) for i in idx])
    return X
    
def STFT(x, framesamp=512):
    """
    Short-time Fourier transform, calling STFT_single along the
    first dimension for each index of the second dimension.
    
    If the input is an 8-channel sound file of 10000 samples each,
    its shape should be
    """
    if len(x.shape)==1:
        return STFT_single(x, framesamp)
    elif len(x.shape)==2:
        a = np.array([STFT_single(x[:, i]) for i in range(0, x.shape[1])])
        return a.transpose(1, 2, 0)
    else:
        raise RuntimeError('STFT argument with more than 2 dimensions')
    
def ISTFT_single(X):
    """
    Inverse function for STFT_single().  Result will be longer
    than the input to STFT_single, and so must be trimmed
    appropiately.
    
    Input must be 2-D array.
    First dimension is time index, second is frequency.
    """
    framesamp = (X.shape[1]-1)*2
    hop = int(framesamp/2)
    w = np.sin(np.pi*((2*np.arange(framesamp)+1)/(2.0*framesamp)))
    bl = np.fft.irfft(X)
    x = sp.zeros(hop*(X.shape[0])+framesamp)
    for n, i in enumerate(range(0, len(x)-framesamp, hop)):
        x[i:i+framesamp] += bl[n, :]*w
    return x[hop:]
    
def ISTFT(X):
    """
    Inverse function for STFT().  Result will be longer
    than the input to STFT, and so must be trimmed
    appropiately.
    
    Input must be 3-D array.
    First dimension is time index, second is frequency.
    """
    if len(X.shape)==2:
        return ISTFT_single(X)
    elif len(X.shape)==3:
        a = np.array([ISTFT_single(X[:, :, i]) for i in range(0, X.shape[2])])
        return a.transpose(1, 0)
    else:
        raise RuntimeError('ISTFT argument with nonsense dimensions')
    
if __name__ == '__main__':
    # test script
    print('Testing the single channel functions...')
    for l in range(1020, 1030):
        x1 = np.random.randn(l)
        print('Length: '+str(l))
        x2 = STFT_single(x1)
        print('Resulting size: '+str(x2.shape))
        x3 = ISTFT_single(x2)
        print('Resulting IFFT size: '+str(x3.shape))
        err = x1 - x3[:l]
        print('Resulting RMS error: '+str(np.sqrt(np.sum(err**2.0))/l))
        print('Max abs error:'+str(np.max(abs(err))))
        
