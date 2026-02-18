### DEEP LEARNING

## CNN - Convolutional Neural Networks
 They are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are designed to automatically and adaptively learn spatial hierarchies of features from input images. CNNs are particularly effective for tasks such as image classification, object detection, and image segmentation.

 They have layers like convolutional layers, pooling layers, and fully connected layers. The convolutional layers apply filters to the input image to extract features, while pooling layers reduce the spatial dimensions of the feature maps. Fully connected layers are used for classification tasks.

During feature extraction :
    image resolution is taken and then 


 In padding the formula used is :
  [n + 2p - f + 1] * [n +2p - f + 1]
  where n is the input size, p is the padding, and f is the filter size

Output size is calculated as :
    [(n - f + 2p) / s + 1]  * [ (n - f + 2p) / s + 1 ]
    where n is the input size, f is the filter size, p is the padding, and s is the stride.


# CNN - Edge Detection
 Edge detection is a technique used in image processing and computer vision to identify and locate sharp discontinuities in an image. These discontinuities typically correspond to edges of objects, boundaries, or changes in intensity. CNNs can be trained to perform edge detection by learning to recognize patterns in the pixel values of an image.

 The process of edge detection using CNNs involves training a model on a dataset of images with labeled edges. The model learns to identify features that correspond to edges, such as changes in intensity or color. Once trained, the CNN can be applied to new images to detect edges automatically.

 Common techniques for edge detection include the Sobel operator, Canny edge detector, and Laplacian of Gaussian (LoG). CNNs can learn to replicate or improve upon these traditional methods by leveraging their ability to capture complex patterns in the data.

 Calculating the number of parameters the formula is :
    (filter height * filter width * number of input channels + 1) * number of filters
    (fh * fw * Cin + 1) * N