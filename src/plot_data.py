import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(hue, data):
    for i, col in enumerate(data.columns):
        plt.figure(i)
        sns.set(rc={'figure.figsize':(11.7, 8.27)})
        ax = sns.countplot(x=data[col], hue=hue, data=data)
        plt.show()

def main():
    DATAPATH = '../data/mushrooms.csv'

    data = pd.read_csv(DATAPATH)

    # plot counts of each class
    hue = data['class']
    p_data = data.drop('class', 1)

    plot_data(hue, p_data)

main()
