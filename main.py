from flask import Flask, send_file, jsonify, request
import cv2
import base64
import numpy as np
import io
from io import BytesIO
import numpy as np
from scipy.signal import convolve2d
from PIL import ImageFont, ImageDraw, Image, ImageEnhance
from flask_cors import CORS

app = Flask(__name__)


# @app.after_request
# def add_headers(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers',
#                          'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     return response

CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})


@app.route('/upload_image', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        file.save('./Images/image.jpg')
        return jsonify({'success': True})


@app.route('/color')
def color():
    filename = './Images/image.jpg'
    img = cv2.imread(filename)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image})

@app.route('/gray')
def gray():
    filename = './Images/image.jpg'
    gray_image = cv2.imread(filename,0)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', gray_image)[1]).decode()
    return jsonify({'image': Encoded_Image})

@app.route('/binary')
def binary():
    color_image = cv2.imread("./Images/image.jpg",0)
    ret, binary_image = cv2.threshold(color_image, 127, 255, cv2.THRESH_BINARY)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', binary_image)[1]).decode()
    return jsonify({'image': Encoded_Image})

# GEOMETRIC TRANSFORMATIONOF IMAGES

# SCALING
@app.route('/scale')
def scale():
    img = cv2.imread('./Images/image.jpg')
    height = 0.5
    width = 0.5
    scaled_img = cv2.resize(img,None,fx=width, fy=height, interpolation =cv2.INTER_LINEAR)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', scaled_img)[1]).decode()
    return jsonify({'image': Encoded_Image})

# Rotation
@app.route('/rotation')
def Rotation():
    img = cv2.imread('./Images/image.jpg')
    angle = 30
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    M[0, 2] += (new_width / 2) - (width / 2)
    M[1, 2] += (new_height / 2) - (height / 2)
    rotated_img = cv2.warpAffine(img, M, (new_width, new_height))
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', rotated_img)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage, 'width': rotated_img.shape[1], 'height': rotated_img.shape[0]})


# Translation
@app.route('/translation')
def Translation():
    img = cv2.imread('./Images/image.jpg')
    rows, cols, x = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', dst)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


## SMOOTHING OF IMAGES
# Averaging
@app.route('/average')
def Averaging():
    img = cv2.imread('./Images/image.jpg')
    blur = cv2.blur(img, (10, 10))
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', blur)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Gaussian
@app.route('/gaussian')
def Guassian():
    img = cv2.imread('./Images/image.jpg')
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', blur)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Median
@app.route('/median')
def Median():
    img = cv2.imread('./Images/image.jpg')
    median = cv2.medianBlur(img, 5)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', median)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Bilateral
@app.route('/bilateral')
def Bilateral():
    img = cv2.imread('./Images/image.jpg')
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', blur)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



## MORPHOLOGICAL OPERATIONS OF IMAGE
# Erosion
@app.route('/erosion')
def Erosion():
    img = cv2.imread('./Images/image.jpg')
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', erosion)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Dilation
@app.route('/dilation')
def Dilation():
    img = cv2.imread('./Images/image.jpg')
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', dilation)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Opening
@app.route('/opening')
def Opening():
    img = cv2.imread('./Images/image.jpg')
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', opening)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Closing
@app.route('/closing')
def Closing():
    img = cv2.imread('./Images/image.jpg')
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', closing)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


## Histograms
# Histogram
@app.route('/histogram')
def Histogram():
    img = cv2.imread('./Images/image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    # Set the histogram parameters
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False

    # Compute the histograms for each color channel
    b_hist = cv2.calcHist([b], [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist([g], [0], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist([r], [0], None, [histSize], histRange, accumulate=accumulate)

    # Normalize the histograms to have values between 0 and 255
    cv2.normalize(b_hist, b_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Create an empty black image to draw the histogram
    hist_height = 256
    hist_width = 256*3
    hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    # Draw the histograms onto the black image
    bin_width = hist_width // histSize
    for i in range(histSize):
        x = i * bin_width
        b_y = int(hist_height - b_hist[i])
        g_y = int(hist_height - g_hist[i])
        r_y = int(hist_height - r_hist[i])
        cv2.rectangle(hist_image, (x, b_y), (x+bin_width, hist_height), (255, 0, 0), -1)
        cv2.rectangle(hist_image, (x+bin_width, g_y), (x+2*bin_width, hist_height), (0, 255, 0), -1)
        cv2.rectangle(hist_image, (x+2*bin_width, r_y), (x+3*bin_width, hist_height), (0, 0, 255), -1)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', hist_image)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


# Histogram Equalization
@app.route('/equalize')
def Equalization():
    img = cv2.imread('./Images/image.jpg')
    b, g, r = cv2.split(img)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    img_eq = cv2.merge((b_eq, g_eq, r_eq))
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', img_eq)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


# Thresholding the Histogram
@app.route('/threshold')
def Thresholding():
    img = cv2.imread('./Images/image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', thresh)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


# SHARPENING
@app.route('/sharpen')
def sharpen():
    img = cv2.imread('./Images/image.jpg')
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(img, -1, kernel)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', sharpened_image)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


@app.route('/laplacian')
def laplacian():
    img = cv2.imread('./Images/image.jpg')
    laplacian = cv2.Laplacian(img, cv2.CV_8U)
    sharp = cv2.addWeighted(img, 1.5, laplacian, -0.5, 0)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', sharp)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



# EDGE DETECTION
@app.route('/prewitt')
def prewitt():
    img = cv2.imread('./Images/image.jpg')
    gray = cv2.imread('./Images/image.jpg',cv2.IMREAD_GRAYSCALE)
    kernel_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    img_prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', img_prewitt_x)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


@app.route('/sobel')
def sobel():
    img = cv2.imread('./Images/image.jpg')
    gray = cv2.imread('./Images/image.jpg',cv2.IMREAD_GRAYSCALE)
    Gx = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])
    sobelx = cv2.filter2D(gray, -1, Gx)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', sobelx)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


@app.route('/robert')
def robert():
    img = cv2.imread('./Images/image.jpg')
    gray = cv2.imread('./Images/image.jpg',cv2.IMREAD_GRAYSCALE)
    robert_x = np.array([[1, 0], [0, -1]])
    filtered_x = cv2.filter2D(gray, -1, robert_x)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', filtered_x)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


# FREQUENCY DOMAIN FEATURES

@app.route('/lowpass')
def lowpass():
    img = cv2.imread('./Images/image.jpg',0)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows = img.shape[0]
    cols = img.shape[1]
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols), np.uint8)
    r = 80
    cv2.circle(mask, (ccol, crow), r, (1, 1, 1), -1)
    dft_shift_masked = dft_shift * mask
    dft_unshifted = np.fft.ifftshift(dft_shift_masked)
    filtered_img = np.fft.ifft2(dft_unshifted)
    filtered_img = np.abs(filtered_img)
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
    filtered_img = np.uint8(filtered_img)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', filtered_img)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


@app.route('/highpass')
def highpass():
    img = cv2.imread('./Images/image.jpg')
    d = 30
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    rows = gray_image.shape[0]
    cols = gray_image.shape[1]
    center_row, center_col = rows//2, cols//2
    mask = np.ones((rows,cols), np.uint8)
    mask[center_row-d:center_row+d, center_col-d:center_col+d] = 0
    fshift_filtered = fshift * mask
    f_filtered = np.fft.ifftshift(fshift_filtered)
    filtered_image = np.fft.ifft2(f_filtered)
    filtered_image = np.abs(filtered_image)
    filtered_image = np.uint8(filtered_image)
    filtered_img = np.uint8(filtered_image)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', filtered_img)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



# POINT OPERATIONS
@app.route('/negation')
def negation():
    img = cv2.imread('./Images/image.jpg')
    inverted_image = cv2.bitwise_not(img)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', inverted_image)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



@app.route('/identity')
def identity():
    img = cv2.imread('./Images/image.jpg')
    identity_transformed_image = img.copy()
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', identity_transformed_image)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



@app.route('/cstretching')
def cstretching():
    img = cv2.imread('./Images/image.jpg')
    b, g, r = cv2.split(img)
    lower_lim = 75
    upper_lim = 255
    b_stretch = cv2.normalize(b, None, alpha=lower_lim, beta=upper_lim, norm_type=cv2.NORM_MINMAX)
    g_stretch = cv2.normalize(g, None, alpha=lower_lim, beta=upper_lim, norm_type=cv2.NORM_MINMAX)
    r_stretch = cv2.normalize(r, None, alpha=lower_lim, beta=upper_lim, norm_type=cv2.NORM_MINMAX)
    stretched_image = cv2.merge((b_stretch, g_stretch, r_stretch))
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', stretched_image)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



@app.route('/logtransform')
def logtransform():
    img = cv2.imread('./Images/image.jpg')
    c = 255 / np.log(1 + np.max(img))
    log_transformed = c * (np.log(img + 1))
    log_transformed = np.uint8(log_transformed)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', log_transformed)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



@app.route('/powerlaw')
def powerlaw():
    img = cv2.imread('./Images/image.jpg')
    gamma = 4
    c = 1
    transformed_img = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', transformed_img)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})


@app.route('/watermark')
def watermark():
    img = cv2.imread('./Images/image.jpg')
    text = 'WaterMark'
    font_path = 'calibri.ttf'
    font_size = 60
    font_color = (100, 150, 100,64)
    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    text_size = draw.textbbox((0, 0), text, font=font)
    x = img.shape[1] - text_size[2] - 25
    y = img.shape[0] - text_size[3] - 25
    draw.text((x, y), text, font_color, font=font)
    watermarked = np.array(pil_image)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', watermarked)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})



@app.route('/brightness')
def brightness():
    img = cv2.imread('./images/image.jpg')
    beta = 100 # Brightness control
    adjusted = cv2.convertScaleAbs(img, beta=beta)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', adjusted)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Contrast

@app.route('/contrast')
def contrast():
    img = cv2.imread('./images/image.jpg')
    alpha = 4 # Contrast control
    beta = 10 # Brightness control
    adjusted = cv2.convertScaleAbs(img, alpha=alpha)
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', adjusted)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

# Cropping

@app.route('/crop')
def crop():
    img = cv2.imread('./images/image.jpg')
    h1,w1,z = img.shape
    x, y, w, h = 0, 0, w1//2, h1//2
    cropped_image = img[y:y+h, x:x+w]
    Encoded_Image = base64.b64encode(
        cv2.imencode('.jpg', cropped_image)[1]).decode()
    OriginalImage = base64.b64encode(
        cv2.imencode('.jpg', img)[1]).decode()
    return jsonify({'image': Encoded_Image, 'OriginalImage': OriginalImage})

if __name__ == '__main__':
    app.run()
