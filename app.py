import nltk
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from flask import Flask,render_template,url_for,request



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        comment = request.form['comment']
        msg = [comment]
        pickle_out = open('./model.pkl', 'rb')
        classifier = pickle.load(pickle_out)
        my_prediction = classifier.predict(msg)[0]
        pickle_out.close()
    return render_template('result.html',prediction = my_prediction)

def text_proc(mess):
    #remove puncituation
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = ''.join(nopunc)

    #remove stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') ]



if __name__ == '__main__':
    app.run(debug=True)
