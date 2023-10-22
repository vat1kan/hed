import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from noise import gauss, sp, impulse

def edgeDetection(img):
    (H, W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), swapRB=False, crop=False)
    net = cv2.dnn.readNetFromCaffe("model/deploy.prototxt", "model/hed_pretrained_bsds.caffemodel")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    return hed

def plot(images, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 7))
    axes = axes.flatten()
    for i in range(len(images)):
        axes[i].imshow(cv2.cvtColor(images[i][0], cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f'{images[i][1]}{images[i][2]}',fontsize = 12)
    plt.tight_layout()
    plt.show()

def perfomance(edge_img):
    detected_edges = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY)[1]
    ground_truth_binary = (ground_truth > 0).astype(np.uint8)
    detected_edges_binary = (detected_edges > 0).astype(np.uint8)
    precision = np.round(precision_score(ground_truth_binary.flatten(), detected_edges_binary.flatten()),5)
    recall = np.round(recall_score(ground_truth_binary.flatten(), detected_edges_binary.flatten()),5)
    f1 = np.round(f1_score(ground_truth_binary.flatten(), detected_edges_binary.flatten()),5)

    return recall, precision,f1


#init
file_name = "results.csv"

img = cv2.imread('images/image.jpg')
ground_truth = edgeDetection(img)

stock_edge = [[img, "Stock image", " "], 
              [ground_truth, "HED detection", " "]]

#gauss noise
gauss_noice = gauss(img)

#salt and papper noise
salt_probability = 0.30
pepper_probability = 0.30
sp_noice = sp(img, salt_probability, pepper_probability)

#impulse noise
impulse_probability = 0.01
impulse_noice = impulse(img, impulse_probability)

#noised images edge detection
gaussEdge = edgeDetection(gauss_noice)
spEdge = edgeDetection(sp_noice)
impulseEdge = edgeDetection(impulse_noice)

gauss_result = perfomance(gaussEdge)
sp_result = perfomance(spEdge)
impulse_result = perfomance(impulseEdge)

noiced_edge = [[gauss_noice,'Gaussian Noise, standart deviatin = ', 75], 
               [sp_noice, 'Salt & Pepper Noise, noice probability = ', salt_probability, pepper_probability], 
               [impulse_noice, 'Impulse Noise, noice probabilty = ', impulse_probability],
               [gaussEdge, 'Edge detection, F1 = ', gauss_result[2]],
               [spEdge, 'Edge detection, F1 = ', sp_result[2]], 
               [impulseEdge, 'Edge detection, F1 = ', impulse_result[2]]]

plot(stock_edge,1,2)
plot(noiced_edge,2,3)


print('\nGaussian noiced image edge detection:')
print(f'Precision: {gauss_result[0]}')
print(f'Recall: {gauss_result[1]}')
print(f'F1 score: {gauss_result[2]}')

print('\nGaussian noiced image edge detection:')
print(f'Precision: {sp_result[0]}')
print(f'Recall: {sp_result[1]}')
print(f'F1 score: {sp_result[2]}')

print('\nGaussian noiced image edge detection:')
print(f'Precision: {impulse_result[0]}')
print(f'Recall: {impulse_result[1]}')
print(f'F1 score: {impulse_result[2]}')
