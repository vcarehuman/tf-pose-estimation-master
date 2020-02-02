import numpy as np
#
#def sigmoid(x):
#    s = 1 /(1+np.exp(-x))
#    print(s)
#    return s
#x = np.array([1, 2, 3])
#print("Sigmoid of x is  = "  +str(sigmoid(x)) + "\n")
#
#def sigmoid_derivative(x):
#    s = sigmoid(x)
#    ds = s*(1-s)
#    print(ds)
#    return ds
#x = np.array([1, 2, 3])
#
#print("\n sigmoid_derivative of x  = "+ str(sigmoid_derivative(x)))
# GRADED FUNCTION: image2vector
#def image2vector(image):
#    """
#    Argument:
#    image -- a numpy array of shape (length, height, depth)
#    
#    Returns:
#    v -- a vector of shape (length*height*depth, 1)
#    """
#    
#    ### START CODE HERE ### (≈ 1 line of code)
#    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))
#    ### END CODE HERE ###
#    
#    return v
#image = np.array([[[ 0.67826139,  0.29380381],
#        [ 0.90714982,  0.52835647],
#        [ 0.4215251 ,  0.45017551]],
#
#       [[ 0.92814219,  0.96677647],
#        [ 0.85304703,  0.52351845],
#        [ 0.19981397,  0.27417313]],
#
#       [[ 0.60659855,  0.00533165],
#        [ 0.10820313,  0.49978937],
#        [ 0.34144279,  0.94630077]]])
#a = image.shape
#print(a)
#print (image, "\n")
#print ("image2vector(image) = " + str(image2vector(image)))
# GRADED FUNCTION: normalizeRows

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord =2,  axis = 1, keepdims = True )
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###

    return x
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))