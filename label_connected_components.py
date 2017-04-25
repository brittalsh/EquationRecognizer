import numpy as np
from PIL import Image
import pandas as pd
import os
#import cv2
from matplotlib import pyplot as plt

#print cv2.__version__


def get_connected_components(bin_image, connectivity=4):
    '''
    Method: One component at a time connected component labeling
        Input: Binary Image (h x w array of 0 and non-zero values, will created connected components of non-zero)
        Returns: connected_array -(h x w array where each connected component is labeled with a unique integer (1:counter-1)
                counter-1 - integer, number of unique connected components
    '''
    h, w = bin_image.shape
    yc, xc = np.where(bin_image != 0)
    queue = []
    connected_array = np.zeros((h, w))  # labeling array
    counter = 1
    for elem in range(len(xc)):
        # iterate over all nonzero elements
        i = yc[elem]
        j = xc[elem]
        if connected_array[i, j] == 0:
            # not labeled yet proceed
            connected_array[i, j] = counter
            queue.append((i, j))
            while len(queue) != 0:
                # work through queue
                current = queue.pop(0)
                i, j = current
                if i == 0 and j == 0:
                    coords = np.array([[i, i + 1], [j + 1, j]])
                elif i == h - 1 and j == w - 1:
                    coords = np.array([[i, i - 1], [j - 1, j]])
                elif i == 0 and j == w - 1:
                    coords = np.array([[i, i + 1], [j - 1, j]])
                elif i == h - 1 and j == 0:
                    coords = np.array([[i, i - 1], [j + 1, j]])
                elif i == 0:
                    coords = np.array([[i, i, i + 1], [j - 1, j + 1, j]])
                elif i == h - 1:
                    coords = np.array([[i, i, i - 1], [j - 1, j + 1, j]])
                elif j == 0:
                    coords = np.array([[i, i + 1, i - 1], [j + 1, j, j]])
                elif j == w - 1:
                    coords = np.array([[i, i + 1, i - 1], [j - 1, j, j]])
                else:
                    coords = np.array([[i, i, i + 1, i - 1], [j - 1, j + 1, j, j]])

                for k in range(len(coords[0])):
                    # iterate over neighbor pixels, if  not labeled and not zero then assign current label
                    if connected_array[coords[0, k], coords[1, k]] == 0 and bin_image[coords[0, k], coords[1, k]] != 0:
                        connected_array[coords[0, k], coords[1, k]] = counter
                        queue.append((coords[0, k], coords[1, k]))
            counter += 1

    return connected_array, counter - 1


def cutimage(lefttop, rightbottom, points):
    if rightbottom[0] - lefttop[0] < 3 or rightbottom[1] - lefttop[1] < 3:
        return
    size = (rightbottom[1] - lefttop[1]+1, rightbottom[0] - lefttop[0]+1)
    img = Image.new('L', size)
    # box = (784, 22, 813, 56)
    # cut = img.crop(box)
    # cut.show()
    pix = img.load()
    for point in points:
        lt = point[0] - lefttop[0]
        rb = point[1] - lefttop[1]
        pix[rb, lt] = 255
    img.save(str(lefttop)+str(rightbottom) + '.png')


def findcomponent(img, connected_arr, width, height):
    lefttop = {}
    rightbottom = {}
    points = {}
    for row in range(height):
        for col in range(width):
            key = connected_arr[row][col]
            if key != 0:
                if not points.has_key(str(key)):
                    points[str(key)] = []
                points[str(key)].append([row, col])
                if not lefttop.has_key(str(connected_arr[row][col])):
                    lefttop[str(connected_arr[row][col])] = [row, col]
                    rightbottom[str(connected_arr[row][col])] = [row, col]
                else:
                    oripos1 = rightbottom[str(connected_arr[row][col])]
                    maxrow1 = max(row, oripos1[0])
                    maxcol1 = max(col, oripos1[1])
                    rightbottom[str(connected_arr[row][col])] = [maxrow1, maxcol1]
                    oripos2 = lefttop[str(str(connected_arr[row][col]))]
                    minrow2 = min(row, oripos2[0])
                    mincol2 = min(col, oripos2[1])
                    lefttop[str(connected_arr[row][col])] = [minrow2, mincol2]
    print lefttop
    print rightbottom
    for key in points.keys():
        cutimage(lefttop[key], rightbottom[key], points[key])


def blackwhite(img):
    width, height = img.size
    pix = img.load()
    for row in range(width):
        for col in range(height):
            color = pix[row,col]
            if color < 100:
                pix[row, col] = 0
            else:
                pix[row, col] = 255



if __name__ == "__main__":
    img = Image.open('/Users/lshbritta/learning~/brandeis/AI/Project/annotated/SKMBT_36317040717260_eq6.png')
    width, height = img.size
    print width
    print height
    img_convert = img.convert('L')
    img_convert.save('L.png')
    blackwhite(img_convert)
    img_convert.save('blackwhite.png')
    matrix = img_convert.getdata()
    matrix = np.reshape(matrix,(height,width))
    connected_arr, count = get_connected_components(matrix)
    # print connected_arr
    # for row in range(height):
    #     s = "".join(str(i) for i in connected_arr[row])
    #     print s
    findcomponent(img, connected_arr, width, height)