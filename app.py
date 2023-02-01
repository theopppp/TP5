import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
from classes import class_label

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# home and input page
@app.route('/')
def home():
    return render_template('index.html')


# predict route
@app.route('/classify/<int_features>', methods=['POST'])
def predict(int_features):
    final_features = np.array(int_features.split(','), dtype=float)
    final_features = final_features / 255
    final_features = final_features.reshape(1,784)
    prediction = model.predict(final_features)
    # get highest probability
    # probability = np.max(prediction)
    # get class that has the highest probability
    id_class_pred = np.argmax(prediction)
    id_class_pred = int(id_class_pred)
    label_class_pred = class_label[id_class_pred]
    return jsonify(label_class_pred)
    # return render_template('results.html', prediction=label_class_pred, proba=probability)


if __name__ == "__main__":
    app.run(debug=True)