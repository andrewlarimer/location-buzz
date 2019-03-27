from flask import Flask, render_template, request, session, Response, make_response
import os
import meta_wordcounts
#import get_similarities
import location_analyzer

# Connect to Redis
app = Flask(__name__)

app.secret_key = b'eiu23#@$u32irn2k/3.,2/.,23'

@app.errorhandler(500)
def page_not_found(e):
    return render_template('error500.html'), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/meta_wordcount', methods=['POST'])
def conduct_wordcout():
    search_phrase = request.form['search_phrase']
    results = meta_wordcounts.evaluate(search_phrase)
    session['wordcount_results'] = results
    return render_template('count_results.html', results=results)

@app.route('/chain_analyzer', methods=['POST'])
def chain_analyzer():
    chain_name = request.form['chain_name']
    city_name = request.form['city_name']
    loc_results = location_analyzer.evaluate(CHAIN_NAME=chain_name, CITY_NAME=city_name)
    return render_template('location_results.html', lr=loc_results)

@app.route('/intent_viz', methods=['POST'])
def intent_viz():
    search_phrase = request.form['search_phrase']
    intent1 = request.form['intent1']
    intent2 = request.form['intent2']
    intent3 = request.form['intent3']
    session['similarity_results'] = get_similarities.evaluate(search_phrase, intent1, intent2, intent3)
    return render_template('intent_viz.html')

@app.route('/counts_as_csv')
def counts_as_csv():
    results = session.get('wordcount_results', None)
    generator = (f"{line['token']},{line['count']} \n" for line in results['wordcounts'])
    return Response(generator,
                       mimetype="text/plain",
                       headers={"Content-Disposition":
                                    "attachment;filename=wordcountresults.csv"})

@app.route('/img/similarity_density/<intent_id>')
def render_similarity_img(intent_id):
    results = session.get('similarity_results', None)
    img_str = render_similarity.from_results_and_id(results, intent_id)
    resp = make_response(img_str)
    resp.content_type = "image/png"
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
