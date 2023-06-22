from flask import Flask, render_template, request
#from flask_mysqldb import MySQL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from googletrans import Translator
translator = Translator()

app = Flask(__name__)

#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = ''
#app.config['MYSQL_DB'] = 'feedbackform'

#mysql = MySQL(app)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('Mergeddata.csv')
dataframe = dataframe.sample(frac=1, random_state=27)
x = dataframe['translated']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)

def fake_news_det(news):
    news = translator.translate(news, dest='en').text
    tfid_x_train = tfvect.fit_transform(x)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/', methods =["GET", "POST"])
def home():
    """if request.method == 'POST':
        details = request.form
        answer = details['answer']
        comment = details['comments']
        prompt = details['prompt']
        cur =mysql.connection.cursor()
        cur.execute("INSERT INTO `feedback` (`message`, `answer`, `comments`) VALUES (%s,%s,%s)", (prompt, answer, comment))
        mysql.connection.commit()
        
        # fetchdata = cur.fetchall()
        cur.close()
        return 'success'"""
        
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

@app.route('/gfg', methods =["GET", "POST"])
def gfg():
    if request.method == 'POST':
        answer = request.form['answer']
        #return "pocha " + answer
        return render_template('index.html', mydata=answer)

if __name__ == '__main__':
    app.run(debug=True)
