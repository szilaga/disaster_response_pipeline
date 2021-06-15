import json
import plotly
import pandas as pd
import joblib
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Scatter


from sklearn.externals import joblib
from sqlalchemy import create_engine


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


app = Flask(__name__)

def tokenize(text):
     #remove unwanted characters:
    text = re.sub(r'[^a-zA-Z0-9]',' ', text)

    #change a sentence into tokens (words)
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w  not in stopwords.words('english')]

    #create object
    lemmatizer = WordNetLemmatizer()

    #intiate empty list
    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens
   
# load data
database = str('../data/DisasterResponse.db')
engine = create_engine('sqlite:///'+database )
df = pd.read_sql_table('disaster_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    records = df['message'].count()
    
    # Plot 2 / 3
    cat_list = ['related', 'request', 'offer', 'aid_related', 'medical_help',
   'medical_products', 'search_and_rescue', 'security', 'military',
   'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
   'missing_people', 'refugees', 'death', 'other_aid',
   'infrastructure_related', 'transport', 'buildings', 'electricity',
   'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
   'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
   'other_weather', 'direct_report']
    
    
    # 5 Most tagged categories
    tagged = df[cat_list]
    most_tagged_cats = tagged.sum().sort_values()[len(cat_list)-5:].index.tolist()
    
    df_most_tagged = tagged[most_tagged_cats]
    most_tagged_values = df_most_tagged.sum().tolist()
    
   
    # 5 least Tagged categories
    least_tagged_cats = tagged.sum().sort_values()[:5].index.tolist()
    
    df_least_tagged = tagged[least_tagged_cats]
    least_tagged_values = df_least_tagged.sum().tolist()
    
  
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=genre_names,
                    y=genre_counts,
                    opacity = 0.9,
                    line={'color': 'rgba(238,59,59, 1)'},
                    marker={'color' : 'rgba(0,0,205, 1)'},
               
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x= most_tagged_cats,
                    y= most_tagged_values,
                    opacity = 0.9,
                  
                )
            ],

            'layout': {
                'title': 'Most tagged categories',
                'yaxis': {
                    'title': "Value Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x= least_tagged_cats,
                    y= least_tagged_values,
                    opacity = 0.9,
                  
                )
            ],

            'layout': {
                'title': 'Least tagged categories',
                'yaxis': {
                    'title': "Value Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    '''
    'cliponaxis', 'connectgaps', 'customdata', 'customdatasrc', 'dx',
    'dy', 'error_x', 'error_y', 'fill', 'fillcolor', 'hoverinfo',
    'hoverinfosrc', 'hoverlabel', 'hoveron', 'hovertext', 'hovertextsrc',
    'ids', 'idssrc', 'legendgroup', 'line', 'marker', 'mode', 'name',
    'opacity', 'r', 'rsrc', 'showlegend', 'stream', 't', 'text',
    'textfont', 'textposition', 'textpositionsrc', 'textsrc', 'tsrc',
    'type', 'uid', 'visible', 'x', 'x0', 'xaxis', 'xcalendar', 'xsrc', 'y',
    'y0', 'yaxis', 'ycalendar', 'ysrc'
     ''' 
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host = '127.0.0.1', port=3001, debug=True)
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()