import cv2
import numpy as np

def prep_texture(filename, new_val, seed):
    image = cv2.imread(filename)
    image_cpy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #h = hsv[:,:,0]
    s = hsv[:,:,1]
    #v = hsv[:,:,2]
    cv2.imshow('Original image',image)

    gray_blur = cv2.medianBlur(gray,3)
    s_blur = cv2.medianBlur(s,3)

    #cv2.imshow('gray_blur', gray_blur)
    #cv2.imshow('s_blur', s_blur)

    canny_gray_blur = cv2.Canny(gray_blur, 0, 50)

    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    edges_smoothed = cv2.GaussianBlur(laplacian, ksize=(11,11), sigmaX=20)
    cv2.imshow('edges_smoothed', edges_smoothed)
    #merged = cv2.merge([laplacian], merged)
    #edges_smoothed = cv2.Canny(edges_smoothed, 0, 40)
    merged = cv2.addWeighted(edges_smoothed, 0.4, canny_gray_blur, 0.6, 0.0)
    #cv2.imshow('edges_smoothed', edges_smoothed)
    cv2.imshow('merged', merged)

    kernel = np.zeros((2,2), np.uint8)
    dilation = cv2.dilate(merged, kernel, iterations=1)
    #dilation = cv2.medianBlur(dilation,3)
    #cv2.imshow('dilation', dilation)

    h, w = dilation.shape[:2]

    resized = cv2.resize(dilation, (w+2, h+2), interpolation = cv2.INTER_AREA)
    cv2.floodFill(image, resized, seed, newVal=new_val, loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
    #cv2.circle(image, (0 ,0), 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
    #cv2.imshow('bef_image', image)
    dilation_v2 = cv2.dilate(image, kernel, iterations=1)
    #cv2.imshow('aft_image', dilation_v2)
    #merged_v = cv2.merge([v], dilation_v2)
    #merged_v = cv2.addWeighted(v, 0.5, dilation_v2, 0.5, 0.0)
    #cv2.imshow('merged_v', merged_v)

    final = cv2.addWeighted(image_cpy, 0.7, dilation_v2, 0.3, 0.0)
    #cv2.imshow('dilation_v2', dilation_v2)
    return final, dilation_v2



def mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (x, y)

        origin = cv2.imread(filename)
        hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        wall_light, wall_light_dilation = prep_texture(filename, (255, 255, 255), seed)
        wall_dark, wall_dark_dilation = prep_texture(filename, (0, 0, 0), seed)
        #texture = prep_texture(filename_texture, (0, 0, 0))
        texture = cv2.imread(filename_texture)
        h, w = origin.shape[:2]
        resized_texture = cv2.resize(texture, (w, h), interpolation = cv2.INTER_AREA)

        #cv2.imshow('wall_light', wall_light)
        cv2.imshow('wall_dark', wall_dark)
        #cv2.imshow('texture', texture)

        bit_xor = cv2.bitwise_xor(wall_light_dilation, wall_dark_dilation)
        bit_xor = cv2.GaussianBlur(bit_xor, ksize=(11,11), sigmaX=20)
        _, bit_xor = cv2.threshold(bit_xor, 10, 255, cv2.THRESH_BINARY)
        cv2.imshow('bit_xor', bit_xor)

        wallpaper = cv2.bitwise_and(bit_xor, resized_texture)
        cv2.imshow('wallpaper', wallpaper)
        #merged = cv2.merge([v], bit_or)
        #final = cv2.addWeighted(origin, 0.5, wallpaper, 0.5, 0.0)
        #wallpaper_with_wall = cv2.bitwise_xor(wall_dark_dilation, wallpaper)
        final = cv2.addWeighted(origin, 0.8, wallpaper, 0.2, 0.0)
        #cv2.imshow('wallpaper_with_wall', wallpaper_with_wall)

        cv2.imshow('finalfinal', final)

cv2.namedWindow('room')
cv2.setMouseCallback('room', mouse_callback)

filename = './room2.jpg'
filename_texture = './lui.jpg'
origin = cv2.imread(filename)
cv2.imshow('room', origin)


cv2.waitKey(0)
cv2.destroyAllWindows()
