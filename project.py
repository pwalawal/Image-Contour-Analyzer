import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


Min_Descriptors = 20
Max_Train_Size = 200


def reconstruct(img):
    """Reconstruct the image from the contour"""
    rows, columns = img.shape   
    # Transforming the image
    DftColumns = cv2.getOptimalDFTSize(columns)
    DftRows = cv2.getOptimalDFTSize(rows)
    DftImg = np.zeros((DftRows, DftColumns))
    DftImg[:rows, :columns] = img
    croprow, cropcolumn = DftRows / 2 , DftColumns / 2    
    # Discrete Fourier Transform Calculation
    dft = cv2.dft(np.float32(DftImg), flags=cv2.DFT_COMPLEX_OUTPUT)
    DftShift = np.fft.fftshift(dft)
    # Masking the image
    mask = np.zeros((DftRows, DftColumns, 2), np.uint8)
    mask[croprow - 100:croprow + 100, cropcolumn - 100:cropcolumn + 100] = 1
    DftShift = DftShift * mask      
    # Using the inverse Fourier transform for image reconstruction
    newDft = np.fft.ifftshift(DftShift)
    result = cv2.idft(newDft)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])         
    # Plotting the result
    plt.title('Result')
    plt.imshow(result, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def findFourierDescriptor(img):
    """Find and return the Fourier-Descriptor of the image contour"""
    Imagecontour = []
    Imagecontour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        Imagecontour)
    contour_array = Imagecontour[0][:, 0, :]
    complexContour = np.empty(contour_array.shape[:-1], dtype=complex)
    complexContour.real = contour_array[:, 0]
    complexContour.imag = contour_array[:, 1]
    fourier_descriptor = np.fft.fft(complexContour) 
    return fourier_descriptor
 
    
def constructContour(descriptors, degree,i):
    """ Construct the contour of the image"""
    # truncate the long list of descriptors to certain length   
    descriptor_in_use = minimize_descriptor(descriptors, degree)
    constructContour = np.fft.ifft(descriptor_in_use)
    constructContour = np.array(
        [constructContour.real, constructContour.imag])
    constructContour = np.transpose(constructContour)
    constructContour= np.expand_dims(constructContour, axis=1)
    # normalizing the contour
    constructContour *= 500 / constructContour.max()
    # type cast to int32
    constructContour = constructContour.astype(np.int32, copy=False)
    constructContourImg = np.zeros((500, 500), np.uint8)
    # draw and save the contour image
    cv2.drawContours(constructContourImg, constructContour, -1, 255, thickness=-1)   
    cv2.imwrite("/Users/Pratik/Downloads/EEL 6562 Project/reconstruct_result1.tiff", constructContourImg)   
    return descriptor_in_use


def minimize_descriptor(descriptors, degree):
    """Minimize fourier descriptor array by removing unshifted cells in array"""
    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    descriptors = descriptors[
        center_index - degree / 2:center_index + degree / 2]
    descriptors = np.fft.ifftshift(descriptors)
    
    return descriptors

    
def train_set_generater(sample1, sample2):
    """this function generates training_set"""
    response = np.array([0, 1])
    response = np.tile(response, Max_Train_Size / 2)
    response = response.astype(np.float32)
    set = np.empty(
        [Max_Train_Size,Min_Descriptors], dtype=np.float32)
    for i in range(0, Max_Train_Size - 1, 2):
        descriptors_sample1 = findFourierDescriptor(sample1)
        descriptors_sample1 = minimize_descriptor(
            descriptors_sample1,
            Min_Descriptors)            
        set[i] = np.absolute(descriptors_sample1)        
        descriptors_sample2 = findFourierDescriptor(sample2)
        descriptors_sample2 = minimize_descriptor(
            descriptors_sample2,
            Min_Descriptors)             
        set[i + 1] = np.absolute(descriptors_sample2)
    return set, response

    
def test_set_generater(sample1):
    """this function generates test_set"""
    response = np.array([0, 1])
    response = np.tile(response, Max_Train_Size / 2)
    response = response.astype(np.float32)
    set = np.empty(
        [Max_Train_Size,Min_Descriptors], dtype=np.float32)
    for i in range(0, Max_Train_Size - 1, 2):
        fourier_descriptors_sample1 = findFourierDescriptor(sample1)
        descriptors_sample1 = minimize_descriptor(
            fourier_descriptors_sample1,
            Min_Descriptors)            
        set[i] = np.absolute(descriptors_sample1) 
    print 'Fourier Descriptors Calculated: ',np.count_nonzero(fourier_descriptors_sample1)
    print 'Minimum Descriptors Used: ',np.count_nonzero(descriptors_sample1)
    efficiency=np.count_nonzero(fourier_descriptors_sample1)*1.0/np.count_nonzero(descriptors_sample1)
    print 'Efficiency of the method:',efficiency
    return set, response


"""generate training_set"""
# import images and preprocess
sample1 = cv2.imread("/plane1.tiff", 0)
sample2 = cv2.imread("/airplanes.tiff", 0)
retval, sample1 = cv2.threshold(sample1, 127, 255, cv2.THRESH_BINARY_INV)
retval, sample2 = cv2.threshold(sample2, 127, 255, cv2.THRESH_BINARY_INV)
training_set, response = train_set_generater(sample1, sample2)
"""generate training_set"""

"""generate test_set from the reconstructed image"""
fourier_Descriptor1 = findFourierDescriptor(sample1)
constructContour1 = constructContour(fourier_Descriptor1, Min_Descriptors,1)

# import reconstructed image and preprocess
sample3 = cv2.imread("/reconstruct_result1.tiff", 0)
reconstruct(sample3)
retval, sample3 = cv2.threshold(sample3, 127, 255, cv2.THRESH_BINARY_INV)
test_set, correct_answer = test_set_generater(sample3)


"""K-nearest neighbor classifier"""
# Train K nearest Neighbour
knn_model = cv2.KNearest(training_set, response)
# Test K nearest Neighbour
ret, answer_KNN, neighbors, distance = knn_model.find_nearest(test_set, k=8)
prediction_KNN = np.sum(
    np.in1d(
        correct_answer,
        answer_KNN)) / Max_Train_Size
answer_KNN= np.array(answer_KNN)
neighbors=np.array(neighbors)
neighbors_acc = np.count_nonzero(neighbors)
mask = answer_KNN==correct_answer
correct = np.count_nonzero(mask)
cnMatrix=confusion_matrix(correct_answer,answer_KNN)
precison=(cnMatrix[0][0]*100.0/(cnMatrix[0][0]+cnMatrix[1][0]))
print '\n'
print 'K Nearest Neighbour'
print 'Precison :',precison
print 'Prediction :', prediction_KNN
print '\n'
""""K-nearest neighbor classifier"""


"""Support Vector Machine Classifier"""
# Define parameters for SVM
svm_parameters = dict(
    kernel_type=cv2.SVM_LINEAR,
    svm_type=cv2.SVM_C_SVC,
    C=1
)
# Train SVM
svm_model = cv2.SVM()
svm_model.train(training_set, response, params=svm_parameters)
# Test SVM
answer_SVM = [svm_model.predict(s) for s in test_set]
answer_SVM = np.array(answer_SVM)
prediction_SVM = np.sum(
    np.in1d(
        correct_answer,
        answer_SVM)) / Max_Train_Size
mask = answer_SVM==correct_answer
correct = np.count_nonzero(mask)
cnMatrix=confusion_matrix(correct_answer,answer_SVM)
cnMatrix=confusion_matrix(correct_answer,answer_SVM)
precison=(cnMatrix[0][0]*100.0/(cnMatrix[0][0]+cnMatrix[1][0]))
print 'SVM '
print 'Precison :',precison
print 'Prediction :', prediction_SVM
print '\n'
"""Support Vector Machine Classifier"""

"""Normal Bayes classifier"""
# Train Normal Bayes
bayes_model = cv2.NormalBayesClassifier()
bayes_model.train(training_set, response)

# Test Normal Bayes
retval, answer_bayes = bayes_model.predict(test_set)
prediction_bayes = np.sum(
    np.in1d(
        correct_answer,
        answer_bayes)) / Max_Train_Size
answer_bayes= np.array(answer_bayes)
mask = answer_bayes==correct_answer
correct = np.count_nonzero(mask)
cnMatrix=confusion_matrix(correct_answer,answer_bayes)
precison=(cnMatrix[0][1]*100.0/(cnMatrix[0][1]+cnMatrix[1][1]))
print 'Normay Bayes '
print 'Precison :',precison
print 'Prediction :', prediction_bayes
print '\n'
"""Normal Bayers classifier"""

