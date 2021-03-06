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
sns.set_style("whitegrid")

APP_NAME="Data utilities"

def inverse(d):
    return 1/d[0]

def sum(d):
    return d[0] + d[1]

def prod(d):
    return d[0]*d[1]

def differentiate(d):
    x=d[0].values
    y=d[1].values
    d1 = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d2 = (d1[1:] + d1[:-1]) * 0.5
    Diff = np.array([d1[0]])
    Diff = np.append(Diff, d2)
    Diff = np.append(Diff, [d1[-1]])
    return pd.Series(Diff)

DATA_CNT={
"inverse" : inverse,
"sum" : sum,
"prod" : prod,
"diff" : differentiate
}




def apply_methods(df,l):
    _df=df.copy()
    try:
        for d in l:
            d_set=[]
            for i in d["col"]:
                d_set.append(_df[i])
            _df[d["name"]]=d["method"](d_set)
    except NameError as e:
        print(e)
        return df
    return _df


def Data_controler(df):
    global SNAP,SNAPSHOTS,current_df,dn_idx
    def save_snapshot(df_added):
        global SNAP,SNAPSHOTS,current_df
        SNAP+=1
        df_added["snap_no"] = SNAP
        SNAPSHOTS = SNAPSHOTS.append(df_added.set_index(["snap_no","data_no"]))
        current_df = SNAPSHOTS.loc[SNAP,:].reset_index()

    def GUI_layout(snap_df):
        global current_df,dn_idx
        current_df = snap_df.loc[SNAP,:].reset_index().dropna(how="all",axis=1)
        col = current_df.columns.tolist()
        dn_idx=col.index("data_no")
        vm=[False if i in ["snap_no","data_no"] else True for i in col]
        col_ = current_df.drop("data_no",axis=1).columns.tolist()

        def sel(axis):
            return sg.Frame("", border_width=0, element_justification="center", pad=(0,10), layout=[
                [sg.Text(axis)],
                [sg.Listbox(col_, size=(6, 6),
                 select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True,
                 default_values=[col_[0]], key=axis)]
            ])

        methods_list=[i for i in DATA_CNT.keys()]

        methods = sg.Frame("",border_width=0, element_justification="center", layout=[
            [
            sg.Text("Methods")
            ],
            [
            sg.Listbox(methods_list, size=(12,6), key="method_list",
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, default_values=[methods_list[0]])
            ]
        ])

        name_input = sg.Frame("",border_width=0, element_justification="right", layout =[
            [
            sg.Text("New column name")
            ],
            [
            sg.InputText("",key="new_col", size=(15,1))
            ]
        ])

        controler = [
            [
                sel("d0"),sel("d1"),sel("d2"),methods,name_input
            ],
            [
                sg.Button("Apply method", key="method"),
                sg.Button("Plot d0 vs d1", key="plt"),
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
                    auto_size_columns=False,
                    enable_events=True,
                    select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
                    background_color='#aaaaaa',
                    alternating_row_color='#888888',),
                    sg.Button("Delete point", key="del" ,disabled=True),
                    sg.Button("Undo"),
            ],
            [sg.Button("OK"),sg.Button("Apply all"),sg.Button("Cancel")]
        ]

    SNAP=0
    SNAPSHOTS=pd.DataFrame()
    _df = df.copy()
    _df.reset_index(drop=True)
    _df["data_no"]=_df.index
    _df["snap_no"]=SNAP
    SNAPSHOTS=SNAPSHOTS.append(_df.set_index(["snap_no","data_no"]))
    current_df,dn_idx=None,None

    APPLIED_METHODS_RECORD=[]

    window = sg.Window(APP_NAME,location=(0,0),layout=GUI_layout(SNAPSHOTS),finalize=True)
    print(current_df)
    print(dn_idx)
    while True:
        event, values =window.read()

        if event in ("Cancel", sg.WIN_CLOSED, None):
            FLAG_APPLY_METHODS_OTHER_DATA=False
            window.close()
            return  df,APPLIED_METHODS_RECORD, FLAG_APPLY_METHODS_OTHER_DATA

        elif event == "OK":
            FLAG_APPLY_METHODS_OTHER_DATA=False
            window.close()
            return current_df.drop("data_no",axis=1) ,APPLIED_METHODS_RECORD, FLAG_APPLY_METHODS_OTHER_DATA

        elif event == "Apply all":
            FLAG_APPLY_METHODS_OTHER_DATA=True
            window.close()
            return current_df.drop("data_no",axis=1) ,APPLIED_METHODS_RECORD, FLAG_APPLY_METHODS_OTHER_DATA


        elif event == "table":
            if len(values["table"])==0:
                window["del"].update(disabled=True)
            elif len(values["table"])==1:
                window["del"].update(disabled=False)
            else:
                window["del"].update(disabled=False)

        elif event == "del":
            data_numbers=[]
            for i in values["table"]:
                data_numbers.append(window["table"].get()[i][dn_idx])

            current_df = current_df.query("data_no not in @data_numbers")

            save_snapshot(current_df)
            window["table"].update(values=current_df.values.tolist())

        elif event == "plt":
            fig = plt.figure(figsize=(10, 7), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(current_df[values["d0"][0]].values, current_df[values["d1"][0]].values, edgecolor="black", linewidth=0.5, alpha=0.5)
            ax.set_title(values["d0"][0] + " vs " + values["d1"][0])
            plt.show()

        elif event == "method":
            sel_data=[current_df[values["d0"][0]], current_df[values["d1"][0]], current_df[values["d2"][0]] ]
            sel_col = [ values["d0"][0], values["d1"][0], values["d2"][0] ]
            new_col_name = "col_0" if values["new_col"]=="" else values["new_col"]
            for i in range(1000):
                if new_col_name in current_df.columns.tolist():
                    new_col_name = new_col_name[:-1] + str(i+1)
                else:
                    break
            current_df[new_col_name] = DATA_CNT[values["method_list"][0]](sel_data)
            APPLIED_METHODS_RECORD.append({"name":new_col_name, "col":sel_col, "method":DATA_CNT[values["method_list"][0]]})
            save_snapshot(current_df)

            window.close()
            window = sg.Window(APP_NAME,location=(0,0),layout=GUI_layout(SNAPSHOTS),finalize=True)

        elif event == "Undo":
            if SNAP>0:
                SNAP-=1
                SNAPSHOTS = SNAPSHOTS.loc[slice(SNAP),:]
                window.close()
                window = sg.Window(APP_NAME,location=(0,0),layout=GUI_layout(SNAPSHOTS),finalize=True)
