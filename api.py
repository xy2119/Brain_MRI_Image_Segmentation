import os
import matplotlib.pyplot as plt
import matplotlib
plt.style.use("ggplot")
matplotlib.use('agg')

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
upload_folder = "./static"
device = "cpu"

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":

        img_file = request.files.getlist("picpath")

        image_path = []

        for image_file in img_file:

            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            image_name = os.path.basename(image_location)
            image_name = image_name.split('.')[0]
            print(image_name)
            print(image_location)
            print(os.path.realpath(image_location))

            image_path.append(os.path.realpath(image_location))

            from test import fun

        fun(image_path[0])

        return render_template("index.html", image_loc = ("Pred.png"))
            
    return render_template("index.html", prediction=0, image_loc=None)

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=12000, debug=True)
