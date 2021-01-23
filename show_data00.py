import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
from matplotlib import pyplot as plt
import collections
import seaborn as sns

sns.set()

iris = sns.get_dataset_names()
iris

iris = sns.load_dataset("iris")
iris["species"]
sns.pairplot(iris)
plt.show()
