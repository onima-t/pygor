import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
import collections

#改善点：tableだけ別windowで出せば,table_updateで一々更新のためのコードを使わなくてよさそう

APP_NAME="Data utilities"


class Data_controler:
    def __init__(self):
        None

def layout(df):
    col = df.columns
    gui=[[sg.Table(
        key='table',
        values=df.values.tolist(),
        headings=df.columns,
        justification='right',
        max_col_width=50,
        def_col_width=8,
        auto_size_columns=False,
        enable_events=True,
        select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
        right_click_menu=["", ["Select all", "My fit"]],
        background_color='#aaaaaa',
        alternating_row_color='#888888',
        display_row_numbers = True)]]

    window = sg.Window(APP_NAME,layout=gui,finalize=True)

    while True:
        event, values =window.read()

        if event in ("Exit", sg.WIN_CLOSED, None):
            window.close()
            return None
        elif event == "table":
            None
        elif event == "Delete":
            None
        elif event == "Plot":
            None
