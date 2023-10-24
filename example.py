import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from scipy.ndimage import distance_transform_edt
from skimage import feature
# Generate noisy image of a square

def fom (edge_img, edge_gold):
    alpha = 1.0/9
    dist = distance_transform_edt(np.invert(edge_gold))
    fom = 1.0/np.maximum(np.count_nonzero(edge_img), np.count_nonzero(edge_gold))
    N,M = edge_img.shape

    for i in range(0, N):
        for j in range (0, M):
            if edge_img[i,j]:
                fom += 1.0/(1.0+dist[i,j]*dist[i,j]*alpha)
    fom /= np.maximum(np.count_nonzero(edge_img), np.count_nonzero(edge_gold))
    
    return fom

image=np.zeros((128, 128), dtype = float)
image[32:-32, 32:-32] = 1

image=ndi.rotate(image, 15, mode='constant')
image=ndi.gaussian_filter(image, 4)
image=random_noise(image, mode = 'speckle', mean = 0.1)

# Compute the Canny filter for two values of sigma
edges1=feature.canny(image)
edges2=feature.canny(image, sigma = 3)
# display results
fig,ax=plt.subplots(nrows = 1, ncols = 3, figsize = (8, 3))
ax[0].imshow(image,cmap = 'gray')
ax[0].set_title('noisy image', fontsize=20)
ax[1].imshow(edges1,cmap = 'gray')
ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)
ax[2].imshow(edges2,cmap = 'gray')
ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)
for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()

print(fom(edges1, edges2))