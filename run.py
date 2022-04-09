import tensorflow as tf
from tensorflow import keras
import numpy as np

from flask import Flask, redirect, url_for, request,render_template


app=Flask(__name__)

dict={0:"Dyskeratotic",1:"Koilocytotic",2:"Metaplastic",3:"Parabasal",4:"Superficial-Intermediate"}
model_dl = keras.models.load_model("model_dl.h5") 
def start(img_path):
    from tensorflow.keras.preprocessing import image
    batch_size = 32
    img_height = 64
    img_width = 64
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    classes = model_dl.predict_classes(image, batch_size=batch_size)
    return  str(dict[classes.item()])
#probabilities = model_dl.predict_proba(image, batch_size=batch_size)
#probabilities_formatted = list(map("{:.2f}%".format, probabilities[0]*100))
@app.route("/",methods=['POST','GET'])
def home():
    return render_template('index.html')
@app.route('/base',methods=['POST','GET'])
def base():
    if request.method == 'POST':
        img=request.files['a1']
        img_path = "static/" + img.filename
        img.save(img_path)
        p=start(img_path)
        return render_template('base.html',pre=p,im=img_path)

if __name__=='__main__':
    app.run(debug=True)
