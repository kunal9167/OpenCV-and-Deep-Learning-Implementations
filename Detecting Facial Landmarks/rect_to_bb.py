import numpy as np
def rect_to_bb(rect):
    # Take the bounding box created by dlib and convert it into (x,y,w,h)
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y

    return (x,y,w,h)

def shape_to_np(shape,dtype="int"):
    #initialise the list of (x,y) co-ordinates
    coords = np.zeros((68,2),dtype=dtype)

    #Loop over the 68 points and convert them
    for i in range(68):
        coords[i]=(shape.part(i).x,shape.part(i).y)

    return coords
