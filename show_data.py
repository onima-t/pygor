import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
import collections

#改善点：tableだけ別windowで出せば,table_updateで一々更新のためのコードを使わなくてよさそう

APP_NAME="Data utilities"
d=pd.DataFrame([[1,2,4],[6,8,10]],columns=["a","b","c"])


class Data_controler:
    def __init__(self):
        None

def Data_controler(df):
    global current_df,dn_idx
    def GUI_layout(snap_df):
        global current_df,dn_idx
        current_df = snap_df.loc[SNAP,:]
        col = current_df.columns.tolist()
        dn_idx=col.index("data_no")
        vm=[False if i in ["snap_no","data_no"] else True for i in col]

        def sel(axis):
            return sg.Frame("", border_width=0, element_justification="center", pad=(0,10), layout=[
                [sg.Text(axis)],
                [sg.Listbox(col, size=(6, 6),
                 select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True,
                 default_values=[col[0]], key=axis)]
            ])

        controler = [
            [
                sel("d1"),sel("d2"),sel("d3")
            ],
            [
                sg.Button("Test_1"),
                sg.Button("Test_2"),
                sg.Button("Delete point", key="del" ,disabled=True),
                sg.Button("Plot d1 vs d2", key="plt" ,disabled=True)
            ],
        ]

        return [
            [
                sg.Frame("",layout=controler, relief=sg.RELIEF_RAISED)
            ],
            [
                sg.Table(
                    key='table',
                    values=current_df.values.tolist(),
                    headings=col,
                    visible_column_map=vm,
                    justification='right',
                    max_col_width=50,
                    def_col_width=8,
                    auto_size_columns=True,
                    enable_events=True,
                    select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
                    right_click_menu=["", ["Select all", "My fit"]],
                    background_color='#aaaaaa',
                    alternating_row_color='#888888',
                    display_row_numbers = True)
            ]
        ]

    SNAP=0
    SNAPSHOTS=pd.DataFrame()

    df.reset_index(drop=True)
    df["data_no"]=df.index
    df["snap_no"]=SNAP
    SNAPSHOTS=SNAPSHOTS.append(df.set_index(["snap_no","data_no"]))

    window = sg.Window(APP_NAME,layout=GUI_layout(SNAPSHOTS),finalize=True)

    while True:
        event, values =window.read()

        if event in ("Exit", sg.WIN_CLOSED, None):
            window.close()
            return None

        elif event == "table":
            if len(values["table"])==0:
                window["del"].update(disabled=True)
                window["plt"].update(disabled=True)
            elif len(values["table"])==1:
                window["del"].update(disabled=False)
                wondow["plt"].update(disabled=True)
            else:
                window["del"].update(disabled=False)
                window["plt"].update(disabled=False)

        elif event == "del":
            data_numbers=[]
            for i in values["table"]:
                data_numbers.append(window["table"].get()[i][dn_idx])
            current_df = current_df.reset_index().query("data_no not in @data_numbers")

            SNAP+=1
            current_df["snap_no"] = SNAP
            SNAPSHOTS = SNAPSHOTS.append(current_df.set_index(["snap_no","data_no"]))
            current_df = SNAPSHOTS.loc[SNAP,:]
            window["table"].update(values=current_df)

        elif event == "plt":
            None

#Data_controler(d)

d
d["snap_no"]=0
d["index"]=d.index
d.set_index(["snap_no","index"])
d
pack = pd.DataFrame()
pack1=pack.append(d.set_index(["snap_no","index"]))
pack1
d
d.at[0,"a"]=3
d
d["snap_no"]=1
d
pack2=pack1.append(d.set_index(["snap_no","index"]))
pack2
pack2.loc[0,:]
pack2.loc[(slice(None),1),:]
pack2
d
d.at[0,"c"]=16
d["snap_no"]=2
pack3=pack2.append(d.set_index(["snap_no","index"]))
d=pack3.loc[1,:]
pack3
pack3.unstack(level="index")
set(pack3.reset_index()["snap_no"].tolist())
pack3.reset_index()["snap_no"].max()
pack3.query("snap_no < 2")
pack3["snap_no"]
A=[2,8]
pack3.query("b in @A")
pack3.loc[(slice(None),0),:].reset_index()
d
d["a"]=1/d["a"]
d
