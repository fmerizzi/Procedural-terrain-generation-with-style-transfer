from scipy import signal
from scipy.ndimage import convolve
import numpy as np 

def matrix_average(mat1,mat2):
  return (mat1+mat2)/2

def upscale_matrix(mat,scale_factor = 2):
  return np.kron(mat, np.ones((scale_factor,scale_factor)))

def gkern(kernlen=21, std=3):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def blackmanharris_kern(kernlen=21):
    gkern1d = signal.blackmanharris(kernlen).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def bartlett_kern(kernlen=21):
    gkern1d = signal.bartlett(kernlen).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def cosine_kern(kernlen=21):
    gkern1d = signal.cosine(kernlen).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def barthann_kern(kernlen=21):
    gkern1d = signal.barthann(kernlen).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def gaussian_filter(image,kernel,sigma):
  return convolve(image, gkern(kernel,sigma))

def blackmanharris_filter(image,kernel):
  return convolve(image, blackmanharris_kern(kernel))

def bartlett_filter(image,kernel):
  return convolve(image, bartlett_kern(kernel))

def cosine_filter(image,kernel):
  return convolve(image, cosine_kern(kernel))

def barthann_filter(image,kernel):
  return convolve(image, barthann_kern(kernel))
    
def lin_interp(a,b,x):
    return a + x * (b-a)

def fade_function(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def perlin(x,y,seed=0):
    #create a 2d permutation table with numpy
    p = np.random.randint(0, 255, size=(512,), dtype=int)

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # internal coordinates
    xf = x - xi
    yf = y - yi

    # fade the coordinates 
    u = fade_function(xf)
    v = fade_function(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lin_interp(n00,n10,u)
    x2 = lin_interp(n01,n11,u) 
    return lin_interp(x1,x2,v) 


def generate_perlin(dim, space, seed):
    lin = np.linspace(0,space,dim,endpoint=False)
    x,y = np.meshgrid(lin,lin) 
    result = perlin(x,y,seed=seed)
    return result 



def generate_explicit(dim, kernel, layers, sigmas = [1,1,2,3]):
    fine_noise = np.random.rand(dim,dim)

    noises = np.zeros((layers,dim,dim))
    
    for i in range(layers):
        factor = (2)**i
        noises[i] =  upscale_matrix(np.random.rand(int(dim/factor),int(dim/factor)),factor)
        noises[i] = gaussian_filter(noises[i],kernel, sigma=sigmas[i])
    
    result = np.mean(noises, axis = 0)
    return result 