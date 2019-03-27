import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import seaborn as sns

def from_results_and_id(result_json, intent_id):

    results = json.loads(result_json)

    intent_key = str(int(intent_id.split('_')[1]) - 1)

    #print("intent key:" + str(intent_key))
    #print("results:" + str(results))

    sim_list = results['similarities'][intent_key]

    figures = dict()

    figures[intent_key] = plt.figure(num=1, figsize=(6, 1), facecolor='w', edgecolor='w')
    #fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    #plt.rcParams["figure.figsize"] = (7,3)
    ax = figures[intent_key].gca()
    #ax.figsize=(8,0.5)
    # (or if you have an existing figure)
    # fig = plt.gcf()
    ax.set_xlim(0,30)
    ax.set_ylim(-3,3)
    ax.axis('off')
    #for i, val in enumerate(sim_list):
    #circle = plt.Circle((1.45, 0), 0.2, color='g')
    plt.title(f"Similarity of '{results['search_phrase']}' search results vs '{results[intent_id]}'")
    for val in sim_list:
        plt.plot(val, (np.random.rand(1)-.5)*1.2, linestyle='--', marker='o', \
            markersize=20, alpha=0.2, color=sns.color_palette("husl", 3)[int(intent_key)])

    #
    # sns.set_style(style='white')
    # intent_figure = plt.figure(figsize=(6,2))
    # for val in sim_list:
    #     plt.Circle((val, 0), 0.2, color='r')
    #temp_plot = sns.distplot(np.array(sim_list).astype(float), \
        #kde=False, color=sns.color_palette("husl", 3)[int(intent_key)])
    # sns.despine(left=True)
    # temp_plot.set(yticks=[])
    # temp_plot.set_title(f"Similarity of '{results['search_phrase']}' search results vs '{results[intent_id]}'")
    #axes = temp_plot.axes
    #temp_fig = temp_plot.get_figure()
    #axes.set_xlim(1.4,2)
    # axes.set_ylim(1,4)
    figures[intent_key].tight_layout()
    #temp_fig.savefig("tmp/intent{0}.png".format(intent_key+1), transparent=True);
    canvas=FigureCanvas(figures[intent_key])
    png_output = io.BytesIO()
    canvas.print_png(png_output)

    figures[intent_key].clf()

    return png_output.getvalue()
