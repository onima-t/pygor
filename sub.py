import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
#from matplotlib import pyplot as plt
#import matplotlib.patches as patches


data = pd.read_table("test_data.txt",sep=" |\t|,",engine="python")

class Table_window:
    T_df=type(pd.DataFrame([[]]))
    T_se=type(pd.Series([0]))
    def run(self,D):
        if type(D) == self.T_se:
            D=pd.DataFrame(D)

        layout = [[sg.Table(values=D.values.tolist(),
            auto_size_columns=True, headings=D.columns.values.tolist(), justification="right",
            background_color="#aaaaaa",
            alternating_row_color="#888888"
            #def_col_width=6
            )]]

        window = sg.Window("Data Table",layout=layout,finalize=True)
        while True:
            event, v = window.read()

            if event in ("Exit","Quit",None):
                break
        window.close()
        return 0



T=Table_window()
T.run(data)
