"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""

import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Default output
res = {"result": 0,
       "data": [], 
       "error": ''}

try:
    # Get post data
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

        # Convert data url to numpy array
        img_str = re.search(r'base64,(.*)', data).group(1)
        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        arr = np.array(im)[:,:,0:1]

        # Normalize and invert pixel values
        arr = arr / 255.

        # Load trained model
        fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9
    
        model=tf.keras.models.load_model(r'cgi-bin\models\fashionmodel.h5')

        y_hat=model.predict(arr.reshape(1,28,28,1))[0]
   
        predict_index = np.argmax(y_hat)
        result=fashion_mnist_labels[predict_index]

        # Return label data
        res['result'] = 1
        #res['data'] = result
        res['data'] = [float(num) for num in y_hat]

except Exception as e:
    # Return error data
    res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


