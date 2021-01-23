import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
import collections

#改善点：tableだけ別windowで出せば,table_updateで一々更新のためのコードを使わなくてよさそう

app_name="Columns setting"
deci_examples = ["1","1.2","1.23","1.234","1.2345"]
expo_examples = ["1e-3","1.2e-3","1.23e-3","1.234e-3","1.2345e-3"]
DEFAULT_DECI = "1.23"
DEFAULT_EXPO = "1.23e-3"
DEFAULT_REP = "{:.2f}"


class Appearence_controler:
    def __init__(self):
        self.num_col = ["counts"]
        self.obj_col = []
        self.map={"counts":"{:.0f}"}

    def renew_data(self,df,on_col,new_rep=DEFAULT_REP):
        pnc=self.num_col
        self.df=df.copy()
        _df=df.copy()
        col=self.df.columns.tolist()
        self.num_col=[]
        self.obj_col=[]
        for i in on_col:
            if self.df[i].dtype !="O":#文字などでないとき
                self.num_col.append(i)
            else:
                self.obj_col.append(i)

        self.map=dict(zip( self.num_col,[self.map[i] if (i in pnc) else new_rep for i in self.num_col] ))
        for k,v in self.map.items():
            _df[k]=_df[k].map(lambda x: v.format(x))
        self.df_show=_df[on_col]

    def renew_n(self, values):
        if values["rep"]==True:
            style = "{:." + str(deci_examples.index(values["style_deci"])) + "f}"
        else:
            style = "{:." + str(expo_examples.index(values["style_expo"])) + "e}"

        for i in values["appearence_num"]:
            self.map[i]=style
            self.df_show[i] = self.df[i].map(lambda x: style.format(x))



def extract_stats(df,col_name):
    return df[col_name].describe().iloc[1:]#countを除外


def col_cnt(df_latest, Dset, visible_column):
    global df_stats, apc, stats_l, df_LLL
    on_col=visible_column
    off_col=list(set(df_latest.columns.tolist()) - set(on_col) - set(["upd_num","f_path"]))
    off_col.sort()

    apc = Appearence_controler()
    apc.renew_data(df_latest,on_col)
    apc.df_LLL = apc.df


    col_set = set()
    df_all = pd.DataFrame()
    for i in Dset.values():
        col_set = col_set | set(i[1].columns.tolist())
        df_all = df_all.append(i[1])

    col_set = list(col_set)
    col_set.sort()

    def preview_update(window):

        sel_data1 = window["data1"].get_indexes()
        sel_stats = window["stats"].get_indexes()
        sel_on = window["on_col"].get_indexes()
        sel_off = window["off_col"].get_indexes()
        sel_an = window["appearence_num"].get_indexes()
        sel_ao = window["appearence_obj"].get_indexes()
        stats_l =window["stats"].get_list_values()
        radio_b =window["rep"].get()
        sd =window["style_deci"].get()
        se =window["style_expo"].get()

        window.close()
        new_window = sg.Window(app_name,col_controler(on_col, off_col, stats = stats_l), finalize=True)

        new_window["data1"].update(set_to_index=sel_data1)
        new_window["stats"].update(set_to_index=sel_stats)
        new_window["on_col"].update(set_to_index=sel_on)
        new_window["off_col"].update(set_to_index=sel_off)
        new_window["appearence_num"].update(set_to_index=sel_an)
        new_window["appearence_obj"].update(set_to_index=sel_ao)
        if radio_b:
            new_window["rep"].update(value=True)
        else:
            new_window["rep_e"].update(value=True)
        new_window["style_deci"].update(value=sd)
        new_window["style_expo"].update(value=se)
        return new_window



    def col_controler(on_col, off_col, stats=[]):
        appearence = sg.Frame("",layout=[
                [
                    sg.Frame("",element_justification="left",border_width=0,layout=[
                        [sg.Text("String")],
                        [sg.Listbox(apc.obj_col,key="appearence_obj", size=(15,8), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)]
                    ]),
                    sg.Frame("",element_justification="left",border_width=0,layout=[
                        [sg.Text("Number")],
                        [sg.Listbox(apc.num_col,key="appearence_num", size=(15,8), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)]
                    ]),
                    sg.Frame("", layout=[
                            [sg.Radio("Deci", key = "rep", group_id="rep", default=True),sg.Spin(values=deci_examples, size=(8,1), key="style_deci")],
                            [sg.Radio("Expo", key = "rep_e", group_id="rep", default=False),sg.Spin(values=expo_examples, size=(8,1), key="style_expo")],
                            [sg.Button("Apply")]
                    ])
                ]
            ]
        )
        return [
            [
                appearence, sg.Table(apc.df_show.values.tolist(),headings=on_col)
            ],
            [
                sg.Frame("", element_justification="left",
                         layout=[
                            [sg.Text("Data")],
                            [sg.Listbox(col_set,key="data1",size=(8,5), enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
                         ]),
                sg.Frame("", element_justification="left",
                         layout=[
                            [sg.Text("Stats")],
                            [sg.Listbox(stats, key="stats" ,size=(8,5), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
                            [sg.Combo(("a", "b", "hoge"), visible=False, key="col_data2")],
                            [sg.Button("Add>>")]
                         ]),
                sg.Frame("",element_justification="left", layout=[
                    [sg.Text("Visible columns")],
                    [sg.Listbox(values=on_col, key="on_col",size=(20, 8), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)]
                ]),
                sg.Frame("",element_justification="center", layout=[
                    [sg.Button("<-Add")],
                    [sg.Button("Remove->")],
                    [sg.Button("Rename",disabled=True)]
                ]),
                sg.Frame("",element_justification="left" ,layout=[
                    [sg.Text("Invisible columns")],
                    [sg.Listbox(values=off_col, key="off_col", size=(20, 8), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)]
                ])
            ],
            [
            sg.Button("Cancel"),sg.Button("OK")
            ]
        ]

    window=sg.Window(app_name, layout=col_controler(on_col, off_col),finalize=True)
    print(type(window))

    while True:
        event, values = window.read()
        print(event)

        if event in ("OK","Cancel", sg.WIN_CLOSED, None):
            window.close()
            return apc, on_col

        elif event == "data1":
            st=extract_stats(df_all,values["data1"]).dropna(how="any")
            window["stats"].update(values=st.index.tolist())

        elif event == "Add>>":
            stats_columns=["f_path"]
            for j in values["data1"]:
                for k in values["stats"]:
                    new_stat_name = j+"_"+k
                    stats_columns.append(new_stat_name)
                    if j+"_"+k in df_latest.columns.tolist():
                        df_latest = df_latest.drop(new_stat_name, axis=1)

            stats = []
            for i in Dset.items():
                stats_i = []
                dsr = i[1][1].describe()
                for j in values["data1"]:
                    for k in values["stats"]:
                        stats_i.append(dsr.at[k,j])
                stats_i=[i[0]] + stats_i
                stats.append(stats_i)
            df_stats = pd.DataFrame(stats,columns=stats_columns)
            df_latest = pd.merge(df_latest, df_stats, on="f_path")

            A=collections.Counter(stats_columns[1:] + off_col)
            for k,v in A.items():
                if v >1:
                    off_col.remove(k)#重複した要素を排除
            on_col = list(dict.fromkeys(on_col + stats_columns[1:]))
            apc.renew_data(df_latest,on_col)
            window = preview_update(window)

        elif event == "<-Add":
            for i in values["off_col"]:
                off_col.remove(i)
            on_col = on_col + values["off_col"]
            apc.renew_data(df_latest,on_col)
            window = preview_update(window)


        elif event == "Remove->":
            for i in values["on_col"]:
                on_col.remove(i)
            off_col = off_col + values["on_col"]
            apc.renew_data(df_latest,on_col)
            window = preview_update(window)

        elif event == "Apply":
            apc.renew_n(values)
            window = preview_update(window)
