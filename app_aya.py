
# import libs
import streamlit as st
import cv2
import numpy as np
import skimage.io as io
import skimage
import matplotlib.pyplot as plt

def Otsu_metuod(image):
#segment using otsu
 # convert the image to grayscale
  gray_image = skimage.color.rgb2gray(image)
# blur the image to denoise
  blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
# perform automatic thresholding
  t = skimage.filters.threshold_otsu(blurred_image)
  binary_mask = blurred_image > t
  selection = image.copy()
  selection[~binary_mask] = 0
  return selection

# vars
DEMO_IMAGE = 'demo.jpg' # a demo image for the segmentation page, if none is uploaded
favicon = 'dog.png'

# main page
st.set_page_config(page_title='Otsu Segmentation - Aya Even Dar', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title('Image Segmentation using Otsu method, by Aya Even Dar')

# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)

st.sidebar.title('Segmentation Sidebar')
st.sidebar.subheader('Site Pages')

# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)
@st.cache()

# take an image, and return a resized that fits our page
def image_resize(image, width=20, height=20, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized

# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'Segment an Image'])

# About page
if app_mode == 'About App':
    st.markdown('In this app we will segment images using Otsu method')
    
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )



    st.markdown('''
                ## About the app \n
                Try segment your image by Otsu method using this webapp. \n
                Hope you will find it usefull! Aya


                ''')

# Run image
if app_mode == 'Segment an Image':
    
    st.sidebar.markdown('---') # adds a devider (a line)
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )

    # read an image from the user
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    # assign the uplodaed image from the buffer, by reading it in
    if img_file_buffer is not None:
        image = io.imread(img_file_buffer)
    else: # if no image was uploaded, then segment the demo image
        demo_image = DEMO_IMAGE
        image = io.imread(demo_image)

    # display on the sidebar the uploaded image
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    # call the function to segment the image
    segmented_image = Otsu_metuod(image)
    
    # Display the result on the right (main frame)
    st.subheader('Segmented Image')
    st.image(segmented_image, use_column_width=True)
