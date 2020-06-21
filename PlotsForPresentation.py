from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline


x = np.arange(4)
money = [1/10, 1/17, 1/230,1/1000]


def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)



formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(formatter)
plt.title("Relative Risk of Death to Travel 1 Mile")
plt.bar(x, money, color=(1.0, 0.0, 0.0, 0.6))
ax.get_yaxis().set_ticks([])
plt.xticks(x, ('Bicycling', 'Walking', 'Driving','Flying'))

plt.show()




x = np.arange(7)
money = [39.69,41.55,42.35,42.34,44.02,45.83,47.88]


def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)



formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(formatter)
plt.title("Number of Bicyclist in USA (in millions)")
plt.bar(x, money)
plt.xticks(x, ('2006','2008','2010','2012','2014','2016','2018'))

plt.show()



men_means, men_std = (358, 231), (0, 0)

ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.8  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind , men_means, width, yerr=men_std,
                color='SkyBlue', label='Men')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Mean Absolute Error of Traffic Models (number of cyclists)')
ax.set_xticks(ind)
ax.set_xticklabels(('Median (baseline)', 'Model'))


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], .55*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "center")

plt.show()



men_means, men_std = (0.059, 0.328,0.349,0.369), (0, 0,0,0)

ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.8  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind , men_means, width, yerr=men_std,
                color='Lime', label='Men')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Distance Correlation')
ax.set_title('Distance Correlation of Accident Models')
ax.set_xticks(ind)
ax.set_xticklabels(('Number of Cyclists', 'Accidents', 'Empirical Ratio','Model'))


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], .5*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "center")

plt.show()


labels = ['Safest', 'Balance']
men_means = [65, 18]
women_means = [-64, -58]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10,5))
rects1 = ax.bar(x - width/2, men_means, width, label='Average Percent Change in Distance')
rects2 = ax.bar(x + width/2, women_means, width, label='Average Percent Change in Accident Probability')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Change relative to shortest')
ax.set_title('Comparing Routes to Shortest Path')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

import pandas
df_traffic = pd.read_csv('./RawData/DVRPC_Bicycle_Counts.csv')


df_traffic.head()
df_traffic['Year of Study'] = df_traffic['setyear']
df_traffic['Number of Bicyclists Per Day'] = df_traffic['aadb']
df_traffic.plot.scatter(x='Year of Study',y='Number of Bicyclists Per Day')

df_traffic['Number of Bicyclists Per Day'].hist()
