import numpy as np
from functools import lru_cache

def pad_zeros(x, length=None):
    N = int(2**np.ceil(np.log2(len(x))))
    x_inp = np.zeros((N, N) if length == None else (length, length), dtype=np.float32)
    x_inp[:x.shape[0], :x.shape[1]] += x
    return x_inp

@lru_cache(maxsize=12)
def exp_arr(N):
    indices = np.arange(N/2, dtype=np.complex64)
    exp_arr = np.exp(-2.0j*np.pi/N*indices)
    return exp_arr

def fft1d(x):
    # Base Case
    if len(x) == 1: return np.complex64(x)

    # Divide
    x_inp = x.copy()
    N = len(x_inp)
    x_even = x_inp[::2]
    x_odd = x_inp[1::2]

    x_even_fft = fft1d(x_even) 
    x_odd_fft = fft1d(x_odd) 

    # Combine
    x_fft = np.zeros(N, dtype=np.complex64)
    exp_array = exp_arr(N)
    exp_odd_array = exp_array*x_odd_fft
    x_fft[:N//2] = x_even_fft + exp_odd_array
    x_fft[N//2:] = x_even_fft - exp_odd_array

    return x_fft

def test_fft1d():
    x = np.random.random(2**10)

    start_time = time.time()
    fft_numpy = np.fft.fft(x)
    numpy_time = time.time()-start_time

    start_time = time.time()
    fft_test = fft1d(x)
    fft_test_time = time.time()-start_time

    error = np.sum(np.absolute(fft_numpy-fft_test))/np.sum(np.absolute(fft_numpy))*100

    print("\tError:{}%".format(error))
    print("\tNumpy Time / fft1d Time:", numpy_time/fft_test_time)

def fft2d(x):
    # Base Case
    if len(x) == 1: return np.complex64(x)
        
    m,n = x.shape
    lm, ln = np.log2(m), np.log2(n)
    if(lm!=int(lm)):
        dm = int(2**np.ceil(lm))
        x = np.append(x, np.zeros((dm-m, n),dtype=np.int8), axis=0)
        m = dm
    if(ln!=int(ln)):
        dn = int(2**np.ceil(ln))
        x = np.append(x, np.zeros((m, dn-n),dtype=np.int8), axis=1)
        n = dn

    # Solve 1st dimension
    x_inp = x.copy()
    x_fft = np.zeros(x.shape, dtype=np.complex64)
    x_fft = np.array([fft1d(x_inp[i]) for i in range(len(x))], dtype=np.complex64)

    # Solve 2nd dimension
    x_fft = np.array([fft1d(x_fft[:,i]) for i in range(len(x))], dtype=np.complex64).transpose()

    return x_fft

def fft2d_img(x):
    x_outp = fft2d(x)
    N = len(x_outp)

    x_outp_img = np.zeros((N+1, N+1), dtype=np.complex64)
    x_outp_img[: N//2, : N//2] = x_outp[N//2 :, N//2 :]
    x_outp_img[N//2 : -1, : N//2] = x_outp[ : N//2, N//2 :]
    x_outp_img[: N//2, N//2 : -1] = x_outp[N//2 :, : N//2]
    x_outp_img[N//2 : -1, N//2 : -1] = x_outp[: N//2, : N//2]
    x_outp_img[N, :] = x_outp_img[0, :]
    x_outp_img[:, N] = x_outp_img[:, 0]
    x_outp_img[N, N] = x_outp_img[0, 0]

    return x_outp_img

def test_fft2d():
    x = np.random.random((2**10, 2**10))

    start_time = time.time()
    fft_numpy = np.fft.fft2(x)
    numpy_time = time.time()-start_time

    start_time = time.time()
    fft_test = fft2d(x)
    fft_test_time = time.time()-start_time

    error = np.sum(np.absolute(fft_numpy-fft_test))/np.sum(np.absolute(fft_numpy))*100

    print("\tError:{}%".format(error))
    print("\tNumpy Time / fft2d Time:", numpy_time/fft_test_time)

def ifft1d(x):
    x_inp = x.copy()
    N = len(x_inp)

    x_inp[1:] = x_inp[:0:-1]
    return fft1d(x_inp)/N

def test_ifft1d():
    x = np.random.random(2**10)

    x_fft = np.fft.fft(x)

    start_time = time.time()
    ifft_numpy = np.fft.ifft(x_fft)
    numpy_time = time.time()-start_time

    start_time = time.time()
    ifft_test = ifft1d(x_fft)
    ifft_test_time = time.time()-start_time

    error = np.sum(np.absolute(ifft_numpy-ifft_test))/np.sum(np.absolute(ifft_numpy))*100

    print("\tError:{}%".format(error))
    print("\tNumpy Time / ifft1d Time:", numpy_time/ifft_test_time)

def ifft2d(x):
    # Base Case
    if len(x) == 1: return np.complex64(x)

    # Solve 1st dimension
    x_inp = x.copy()
    x_ifft = np.zeros(x.shape, dtype=np.complex64)
    x_ifft = np.array([ifft1d(x_inp[i]) for i in range(len(x))], dtype=np.complex64)

    # Solve 2nd dimension
    x_ifft = np.array([ifft1d(x_ifft[:,i]) for i in range(len(x))], dtype=np.complex64).transpose()

    return x_ifft

def ifft2d_img(x):
    x_inp = x.copy()
    N = len(x_inp)-1
    
    x_inp = np.zeros((N,N), dtype=np.complex64)
    x_inp[: N//2, : N//2] = x[N//2 : -1, N//2 : -1]
    x_inp[N//2 : , : N//2] = x[0 : N//2, N//2 : -1]
    x_inp[: N//2, N//2 : ] = x[N//2 : -1, : N//2]
    x_inp[N//2 : , N//2 :] = x[: N//2, : N//2]

    return np.real(ifft2d(x_inp))

def test_ifft2d():
    x = np.random.random((2**10, 2**10))

    x_fft = np.fft.fft2(x)

    start_time = time.time()
    ifft_numpy = np.fft.ifft2(x_fft)
    numpy_time = time.time()-start_time

    start_time = time.time()
    ifft_test = ifft2d(x_fft)
    ifft_test_time = time.time()-start_time

    error = np.sum(np.absolute(ifft_numpy-ifft_test))/np.sum(np.absolute(ifft_numpy))*100

    print("\tError:{}%".format(error))
    print("\tNumpy Time / ifft1d Time:", numpy_time/ifft_test_time)

if __name__ == "__main__":
    import time

    print("Testing fft1d...")
    test_fft1d()

    print("Testing fft2d...")
    test_fft2d()

    print("Testing ifft1d...")
    test_ifft1d()

    print("Testing ifft2d...")
    test_ifft2d()