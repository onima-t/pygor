import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import PySimpleGUI as sg
import pandas as pd
from scipy.optimize import curve_fit
import os
import tkinter
from tkinter import filedialog
import picture
import random
import add_column as ac
import fitting as fg
import cnt_data as cd
import pickle as pkl



APP_NAME = "pygor"

cm = plt.cm.get_cmap('gist_rainbow_r')
T_oya = pd.DataFrame()
Vcol = ["f_path", "filename", "counts"]
apc=ac.Appearence_controler()
FG = fg.Fit_GUI(fg.FIT_FUNCS)

DATA_MODES = ["Raw data","1st diff","2nd diff","3rd diff","Integral"]

SEP_CHARACTOR = " |\t|,"

AX_FONTSIZE = 18
TICK_LABELSIZE = 15




#folder_path=sg.PopupGetFolder("Choose a Folder")

def differentiate(x, y):
    d1 = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d2 = (d1[1:] + d1[:-1]) * 0.5
    Diff = np.array([d1[0]])
    Diff = np.append(Diff, d2)
    Diff = np.append(Diff, [d1[-1]])
    return Diff

def n_diff(x, y, n):
    diff = np.copy(y)
    for i in range(n):
        diff=differentiate(x,diff)
    return diff



def path_list(path, fe=None):
    files = os.listdir(path)
    if fe != None:
        files_dir = [f for f in files if (os.path.isfile(
            os.path.join(path, f)) and f[-len(fe):] == fe)]
    else:
        files_dir = [f for f in files if os.path.isfile(os.path.join(path, f))]
    return files_dir


def Init_ax(fig, xlabel, ylabel, ymin=0, ymax=1.5, place=111, _3D=False):
    if _3D == False:
        ax = fig.add_subplot(place)
    else:
        ax = fig.add_subplot(place, projection = "3d")
    ax.set_xlabel(xlabel, fontsize=AX_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AX_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    # ax.set_ylim(ymin,ymax)
    ax.grid(color='black', linestyle='dashed', linewidth=0.5)
    return ax

def re_init_ax(ax,xlabel,ylabel):
    ax_VF.cla()
    ax.set_xlabel(xlabel, fontsize=AX_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AX_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.grid(color='black', linestyle='dashed', linewidth=0.5)


def pl_remove(pl, pl_bool):
    if pl_bool == True:
        pl.remove()
        pl_bool = False


def pl_dynes(ax, min, max, params, color, num_p=300, LS="-", label=""):
    x = np.linspace(min, max, num_p)
    y = dynes(x, params[0], params[1], params[2], params[3])
    pl = ax.plot(x, y, color=color, linestyle=LS, label=label)
    return pl

def ref_data_mode(df, xname, yname, mode=None):
    if mode==None:
        M=values["data_mode"]
    else:
        M=mode
    ["Raw data","1st diff","2nd diff","3rd diff","Integral"]
    x,y = df[xname].values,df[yname].values
    if M == "Raw data":
        return data, yname
    elif M == "1st diff":
        y = n_diff(x,y,1)
    elif M == "2nd diff":
        y = n_diff(x,y,2)
    elif M == "3rd diff":
        y = n_diff(x,y,3)
    elif M == "Integral":
        return 0#工事中
    new_yname = M + " " + yname
    df[new_yname] = pd.Series(y)
    #print(y)
    return df,new_yname

def ref_fit_info():
    path=window["-TABLE-"].get()[values["-TABLE-"][0]][0]
    T = latest_df(T_oya).query("f_path == @path")
    l = T.columns
    info = ""
    for col in FG.func.columns_data + FG.func.columns_init:
        if col in T.columns:
            info = info + col + " : " + str(T[col].values[0]) + "\n"
    info = info[:-1]
    window["info"].update(value=info)

def check_bounds(name_c):
    name = name_c[:-2]
    if values[name_c] == True:
        M = float(values[name + "_Max"])
        m = float(values[name + "_Min"])
        window[name].update(range=(m, M))
    else:
        window[name].update(range=Prange[name])




def prepare_fit(x, y):
    if f_range_c == True:
        fmin = values["Min"]
        fmax = values["Max"]
        y_f = y[x > fmin]
        x_f = x[x > fmin]
        return x_f[x_f < fmax], y_f[x_f < fmax]
    else:
        return x, y

def latest_df(df,idx_s="f_path", sort_order=True):
    if df.values.shape !=(0,0):
        idx = df.groupby("f_path")["upd_num"].transform(max) == df["upd_num"]  # 最新のindexを取得
        return df[idx].sort_values(idx_s, ascending=not sort_order)
    else:
        return df

def interface_update():
    global window, values
    val_x=window["x"].get_list_values()
    val_y=window["y"].get_list_values()
    val_z=window["z"].get_list_values()
    sel_x=window["x"].get_indexes()
    sel_y=window["y"].get_indexes()
    sel_z=window["z"].get_indexes()
    sel_f=window["func_list"].get_indexes()[0]

    FG.func.save_params(values)
    window.close()
    window = sg.Window(APP_NAME, layout(Vcol,values["func_list"]), location=(0, 0), finalize=True)

    window["x"].update(values=val_x,set_to_index=sel_x)
    window["y"].update(values=val_y,set_to_index=sel_y)
    window["z"].update(values=val_z,set_to_index=sel_z)
    window["func_list"].update(set_to_index=sel_f)
    window["fe"].update(value=values["fe"])
    window["sort"].update(value=values["sort"])
    window["sort_order"].update(value=values["sort_order"])
    window["frange_min"].update(value=values["frange_min"])
    window["frange_max"].update(value=values["frange_max"])
    window["rep_frange_max"].update(value=values["rep_frange_max"])
    window["rep_frange_max"].update(value=values["rep_frange_max"])
    window["rep_frange_max"].update(value=values["rep_frange_max"])
    window["plt_mode"].update(value=values["plt_mode"])
    window["data_mode"].update(value=values["data_mode"])


def table_update_contents(idx_s, sel=None, sort_order=True):
    global window
    a=latest_df(T_oya, idx_s, sort_order)
    apc.renew_data(a,Vcol, "{:.2e}")
    a=apc.df_show

    interface_update()

    window["-TABLE-"].update(values=a[Vcol].values.tolist(), select_rows=sel)
    window["Visual fit"].update(disabled=True)

def table_update_contents_(idx_s, sel=None, sort_order=True):
    global window
    a=latest_df(T_oya, idx_s, sort_order)
    apc.renew_data(a,Vcol, "{:.2e}")
    a=apc.df_show

    window["-TABLE-"].update(values=a[Vcol].values.tolist(), select_rows=sel)
    window["Visual fit"].update(disabled=True)

def table_update_order(idx_s, sel=None, sort_order=True):
    global window
    a=latest_df(T_oya, idx_s, sort_order)
    apc.renew_data(a,Vcol)
    a=apc.df_show

    window["-TABLE-"].update(values=a[Vcol].values.tolist(), select_rows=sel)
    window["Visual fit"].update(disabled=True)

def fg_update():
    global window, values
    t=window["-TABLE-"].get()
    sel_data=values["-TABLE-"]
    interface_update()
    window["-TABLE-"].update(values=t, select_rows=sel_data)
    try:
        FG.frange_update(window, STACK_FROM_TABLE)
    except NameError as e:
        FG.frange_update(window)

TABLE_SELECTED=[]
def ref_data_col():
    global STACK_FROM_TABLE, TABLE_SELECTED
    if TABLE_SELECTED!=values["-TABLE-"]:
        TABLE_SELECTED = values["-TABLE-"]
        l = []
        for i in values["-TABLE-"]:
            data = Dset[window["-TABLE-"].get()[i][0]][1]
            l.append(data)
        STACK_FROM_TABLE = pd.concat(l, join="inner")
        col = STACK_FROM_TABLE.columns.values.tolist()

        def upd(i):
            if values[i] != []:
                if values[i][0] in col:
                    window[i].update(set_to_index=col.index(
                        values[i][0]), values=col)
                else:
                    window[i].update(set_to_index=0, values=col)
            else:
                window[i].update(set_to_index=0, values=col)

        upd("x")
        upd("y")
        upd("z")
        FG.frange_update(window, STACK_FROM_TABLE, values["frange_min"], values["frange_max"])



def get_data(paths):
    global upd_num, T_oya, Dset, D
    Tadd = []
    for fp in paths:
        D = pd.read_csv(fp, sep=SEP_CHARACTOR, engine="python")
        Dset[fp] = [fp.split("/")[-1], D]
        Tadd.append([upd_num, fp, fp.split("/")[-1], len(D)])
    _T = pd.DataFrame(Tadd, columns=["upd_num", "f_path", "filename", "counts"])
    T_oya = T_oya.append(_T)
    table_update_order(values["sort"], sel=values["-TABLE-"],sort_order=values["sort_order"])
    upd_num += 1


def undo():
    global upd_num, T_oya
    T_oya = T_oya[T_oya["upd_num"] != upd_num]
    table_update_contents_(values["sort"])
    upd_num -= 1

#def CLICK_TABLE():


def save_data():
    global folder_path, f_cnd
    df_cnd = pd.DataFrame(f_cnd, columns=columns_cnd)
    df_ = pd.merge(df, df_cnd, on=df.columns[:2].tolist())

    idx = df_.groupby("filename")["fit num"].transform(
        max) == df_["fit num"]  # 最新のfitのindexを取得
    a = df_[idx].sort_values("T")
    a = a.dropna(how="any")

    root = tkinter.Tk()
    root.withdraw()
    dir = tkinter.filedialog.askdirectory()
    try:
        a.to_csv(dir + "/fit_res.csv", sep="\t", index=False)
        sg.popup("The data is saved.\n" + dir + "/fit_res.csv")
    except PermissionError as e:
        print(e)
        sg.popup("Failed to save the data.")



def layout(col,sel_func=[]):
    """
    f_path(一番左の列)のみ非表示にしておく
    """

    vm = [True for i in range(len(col))]
    vm[0] = False

    """
    layoutの部品を先に定義
    """
    Fit_range_button = sg.Button('Off', size=(
        3, 1), button_color='white on red', key='-B-')

    menu_def = [['&File', ["&Save as csv", "Save as pkl", "Load pkl" ,"Undo", 'E&xit']]]

    Browse = [  # sg.Text('Your Folder', size=(15, 1), justification='right'),
        sg.InputText('', key="path", enable_events=True), sg.FolderBrowse(
            key="File")
    ]

    S = (12, 1)


    buttons = [
        [sg.Button("Column Setting", size=S)],
        [sg.Button("Check fit", size=S, disabled=True)],
        [sg.Button("Modify data",key="modify",size=S, disabled=True)],
        [sg.Button("test2")]
    ]

    def sel(axis):
        return sg.Frame("", border_width=0, element_justification="center", pad=(0,10), layout=[
            [sg.Text(axis)],
            [sg.Listbox([], size=(6, 6), select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, default_values="", key=axis)]
        ])

    data_sel = [
        [
        sel("x"),sel("y"),sel("z"),sg.CBox("y_offset?",enable_events=True)
        ],
        [
        sg.OptionMenu(["Nomal", "Color", "3D"],key="plt_mode"),
        sg.OptionMenu(DATA_MODES,key="data_mode"),
        sg.Button("Plot",disabled=True)
        ],
    ]

    plot_menu = sg.Frame("Plot menu",layout=[
        [
            sel("_z")
        ],
    ])


    Fit_controler = FG.layout(sel_func)

    finfo = sg.Frame("Fitting Info", border_width=0, layout = [[sg.MLine(default_text="", size=(30, 10), key="info", disabled=True)]])



    """
    GUI のレイアウトの設定
    """
    return [[sg.Menu(menu_def, tearoff=False)],
            [
                sg.Text("拡張子", size=(5, 1)), sg.Input(".txt", key="fe", size=(5, 1), disabled=True),
                sg.Input("", key="folder", enable_events=True, visible=False),
                sg.FolderBrowse(button_text="Add folder", key="add_folder"),
                sg.FilesBrowse(button_text="Add data files",
                            key="add_files", target="names", enable_events=True),
                sg.Text("Sort by"),
                sg.Combo(Vcol, enable_events=True,
                      default_value="filename", key="sort"),
                sg.CBox("Reverse order", key="sort_order", enable_events=True),
                sg.Input("", key="names", enable_events=True, visible=False), sg.Button("test")
            ],
            # Browse,
            [
                sg.Frame("", buttons, key="buttons", border_width=1,),
                sg.Table(
                    key='-TABLE-',
                    values=[],
                    headings=col,
                    visible_column_map=vm,
                    col_widths=[23, 23, 5],
                    # row_colors=[(0, "red", "white"), (4, "white", "#aaaaff")],
                    justification='right',
                    max_col_width=50,
                    def_col_width=8,
                    auto_size_columns=False,
                    enable_events=True,
                    select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
                    right_click_menu=["", ["Select all", "My fit"]],
                    background_color='#aaaaaa',
                    alternating_row_color='#888888',
                    display_row_numbers = True), finfo
            ],
            [sg.HorizontalSeparator()],
            [
                sg.Frame("Data menu", data_sel, element_justification="center"),
                plot_menu
            ],
            [
                sg.Frame("Fit panel", Fit_controler, relief=sg.RELIEF_RAISED, border_width=5)
            ]
        ]


window = sg.Window(APP_NAME, layout(Vcol), location=(0, 0), finalize=True)


Selected_points = []


def onclick(event):  # グラフ上でクリックしたときに実行される関数
    global pl_select_show, pl_select
    if event.button == 3:  # 右クリック(左クリックにするには 1 に変えればよい)
        xmin, xmax, ymin, ymax = plt.axis()
        Xwidth = xmax - xmin
        Ywidth = ymax - ymin

        x = event.xdata
        y = event.ydata
        R = ((Diff_vis - y) / Ywidth)**2 + ((V_vis - x) / Xwidth)**2
        num = np.argmin(R)
        Selected_points.append(num)
        if pl_select_show == True:
            pl_select.remove()
        else:
            pl_select_show = True
        pl_select = ax.scatter(V_vis[Selected_points], Diff_vis[Selected_points],
                               c='pink', marker='.', s=200, edgecolors='red')
        plt.pause(0.1)
        window["Delete"].update(disabled=False)


f_range_c = graphic_off = False  # 同じ値を複数の変数に同時に代入
pl_p_show = False
pl_f_show = False
pl_f_lr_show = False
pl_d_show = False
pl_select_show = False
pl_f_range_show = False
ax = None
fit_num = 0
f_cnd = []
Dset = {}
upd_num = 0

fig_VF = plt.figure(figsize=(10, 7), dpi=100)
ax_VF=Init_ax(fig_VF,"","")
fig_VF.show()
plt.close(fig_VF)#位置を初期化したいだけなのですぐ閉じる

while True:
    event, values = window.read()

    if event == "__TIMEOUT__":
        continue
    else:
        print(event)

    # Exitボタンが押されたら、またはウィンドウの閉じるボタンが押されたら終了
    if event in ('Exit', sg.WIN_CLOSED, None):
        window.close()
        break

    elif event == "names":
        if values["names"] != "":
            new = values["names"].split(";")
            get_data(new)

    elif event == "folder":
        if values["folder"] != "":
            new = [values["folder"] + "/" +
                   i for i in path_list(values["folder"], fe=values["fe"])]
            get_data(new)

    elif event == "modify":
        selected_data = window["-TABLE-"].get()[values["-TABLE-"][0]][0]
        new, methods, flag_apply = cd.Data_controler(Dset[selected_data][1])
        if flag_apply:
            for k in Dset.keys():
                Dset[k][1] = cd.apply_methods(Dset[k][1],methods)
        else:
            Dset[selected_data][1] = new

    elif event == "Fit":
        x_sel = values["x"][0]
        y_sel = values["y"][0]
        ffn=values["func_list"][0] #fitting function name
        fr_m  = float(values["rep_frange_min"])
        fr_M  = float(values["rep_frange_max"])
        res = []
        _df = pd.DataFrame(columns=T_oya.columns)
        for i in values["-TABLE-"]:
            sel_table =window["-TABLE-"].get()[i]
            data = Dset[window["-TABLE-"].get()[i][0]][1].copy()
            data, new_yname = ref_data_mode(data, x_sel, y_sel)
            data=data[data[x_sel]>=fr_m]
            data=data[data[x_sel]<=fr_M]
            try:
                res_i, init_i = FG.func.fit(data[x_sel].values ,data[new_yname].values ,values)
            except ValueError as e:
                sg.PopupError("Fitting Error!\n"+e)
                break
            res.append([sel_table[0]] + res_i + [x_sel,y_sel,values["data_mode"],fr_m,fr_M] +init_i)

            _df = _df.append(latest_df(T_oya).query("f_path == @sel_table[0]"))
        else:#when for-loop finishes without any fitting error
            res = pd.DataFrame(res, columns=["f_path"]+FG.func.columns+FG.func.columns_data+FG.func.columns_init).sort_values("f_path").drop("f_path", axis=1)
            _df = _df.sort_values("f_path").reset_index(drop=True)
            _df[FG.func.columns+FG.func.columns_data+FG.func.columns_init] = res
            _df["upd_num"] = upd_num
            _df = _df.replace({None:np.nan})#fitがうまく行かなかったら欠損値にしておく
            T_oya = T_oya.append(_df)

            add_col=False
            for i in FG.func.columns:
                if i not in T_oya.columns:
                    print("test!")
                    T_oya[i]=np.nan
                if i not in Vcol:
                    Vcol.append(i)
                    add_col=True

            if add_col:
                table_update_contents(values["sort"], sel=values["-TABLE-"], sort_order=values["sort_order"])
            else:
                table_update_contents_(values["sort"], sel=values["-TABLE-"], sort_order=values["sort_order"])

            upd_num+=1

            #結果のプロット

    elif event == "Check fit":
        d4c={}
        for i in df4check.columns:
            d4c[i]=df4check[i].values[0]
        data = Dset[d4c["f_path"]][1].copy()
        fig = plt.figure(figsize=(8, 7), dpi=100)
        ax = Init_ax(fig, d4c[ffn4c + "_xdata"], d4c[ffn4c+"_Data_type"]+" "+d4c[ffn4c + "_ydata"], place=111)
        data, new_yname = ref_data_mode(data, d4c[ffn4c + "_xdata"], d4c[ffn4c + "_ydata"], mode=d4c[ffn4c+"_Data_type"])
        xdata,ydata = data[d4c[ffn4c + "_xdata"]],data[new_yname]
        ax.scatter(xdata,ydata,edgecolors="black",linewidth=0.5, alpha=0.40, label="Data points")

        xmin,xmax = np.min(xdata.values), np.max(xdata.values)
        fmin,fmax=d4c[ffn4c+"_fit_range_min"],d4c[ffn4c+"_fit_range_max"]
        FG.fit_dict[ffn4c].plot_result(ax,d4c,xmin,xmax,fmin,fmax)

        ax.legend()
        plt.pause(0.1)

    elif event == "Column Setting":
        if Dset!={}:
            #AAA = latest_df(T_oya, values["sort"], values["sort_order"])
            apc, Vcol[1:] = ac.col_cnt(latest_df(T_oya, values["sort"], values["sort_order"]), Dset, Vcol[1:])
            apc.df["upd_num"] = upd_num
            T_oya = T_oya.append(apc.df)
            table_update_contents(values["sort"], sel=values["-TABLE-"], sort_order=values["sort_order"])

            upd_num+=1
        else:
            sg.PopupError("No data exists")

    elif event == "-TABLE-":
        if len(values["-TABLE-"]) == 0:
            window["Visual fit"].update(disabled=True)
            window["Plot"].update(disabled=True)
            window["Check fit"].update(disabled=True)
            window["modify"].update(disabled=True)
        else:
            ref_data_col()
            if len(values["-TABLE-"]) == 1:
                window["Visual fit"].update(disabled=False)
                window["Plot"].update(disabled=False)
                ffn4c=values["func_list"][0] #fitting function name
                window["Check fit"].update(disabled=True)
                window["modify"].update(disabled=False)
                ref_fit_info()

                if ffn4c+"_xdata" in T_oya.columns:
                    df4check=latest_df(T_oya)
                    df4check=df4check[df4check["f_path"] == window["-TABLE-"].get()[ values["-TABLE-"][0] ][0]]
                    if not df4check[ffn4c+"_xdata"].isnull().values[0]:#選択中の関数でのfit結果が出力されているか？
                        window["Check fit"].update(disabled=False)

            else:
                window["Visual fit"].update(disabled=True)
                window["Plot"].update(disabled=False)
                window["Check fit"].update(disabled=True)
                window["modify"].update(disabled=True)

    elif event in ["sort", "sort_order"]:
        table_update_order(values["sort"], sort_order=values["sort_order"])

    elif event == "Select all":
        window["-TABLE-"].update(select_rows=np.arange(len(window["-TABLE-"].get())))

    elif event == "Visual fit":
        VF_x,VF_y,VF_dx,VF_dy=fig_VF.canvas.manager.window.geometry().getRect()
        fig_VF.clf()
        plt.close(fig_VF)
        del fig_VF,ax_VF

        x_sel = values["x"][0]
        y_sel = values["y"][0]
        data = Dset[window["-TABLE-"].get()[values["-TABLE-"][0]][0]][1].copy()
        data, new_yname = ref_data_mode(data, x_sel, y_sel)
        #fig_VF = plt.figure(figsize=(10, 7), dpi=100)

        fig_VF = plt.figure(figsize=(10, 7), dpi=100)
        fig_VF.canvas.manager.window.setGeometry(VF_x,VF_y,VF_dx,VF_dy)
        ax_VF=Init_ax(fig_VF, x_sel, new_yname)
        FG.func.VF_init(ax_VF, data[x_sel].values, data[new_yname].values, values)


    elif event == "x":
        try:
            FG.frange_update(window, STACK_FROM_TABLE)
        except NameError:
            FG.frange_update(window)


    elif event == "Plot":
        x_sel = values["x"][0]
        y_sel = values["y"][0]
        z_sel = values["z"][0]
        fig = plt.figure(figsize=(8, 7), dpi=100)
        ax = Init_ax(fig, x_sel, y_sel, place=111)
        if values["plt_mode"]=="Nomal":
            for i in values["-TABLE-"]:
                data = Dset[window["-TABLE-"].get()[i][0]][1].copy()
                data, new_yname = ref_data_mode(data, x_sel, y_sel)
                x = data[x_sel].values
                y = data[new_yname].values
                ax.scatter(x,y,edgecolors="black",linewidth=0.5, alpha=0.40, label=Dset[window["-TABLE-"].get()[i][0]][0])
                ax.legend()
        else:
            data_set = pd.DataFrame()
            for i in values["-TABLE-"]:
                data = Dset[window["-TABLE-"].get()[i][0]][1].copy()
                data, new_yname = ref_data_mode(data, x_sel, y_sel)
                data_set = data_set.append(data)
            if values["plt_mode"] == "Color":
                pl = ax.scatter(data_set[x_sel].values, data_set[new_yname].values, c=data_set[z_sel].values, cmap=cm, edgecolor="black",linewidth=0.40,alpha=0.3)
                fig.colorbar(pl,ax=ax)
            elif values["plt_mode"] == "3D":
                ax = Init_ax(fig, x_sel, y_sel, place=111, _3D=True)
                pl = ax.scatter3D(data_set[x_sel].values, data_set[new_yname].values, zs=data_set[z_sel].values, alpha=0.5)

        ax.set_ylabel(new_yname)
        plt.pause(0.1)

    elif event == "func_list":
        fg_update()

    elif event in [i+"_slider" for i in FG.func.Pnames]:
        FG.par_slider_move(window,values,event)
        if values["Active"]:
            FG.func.VF_active_fit(values)
        else:
            FG.func.VF_par_slider_move(values)


    elif event in FG.par_bounds_names:
        FG.par_range_overwritten(window,values,event)

    elif event in ("frange_max", "frange_min"):
        FG.frange_slider_move(window,values["frange_max"],values["frange_min"],event)
        if values["Active"]:
            FG.func.VF_frange_slider_move_(values)
            FG.func.VF_active_fit(values)
        else:
            FG.func.VF_frange_slider_move(values)

    elif event =="Active":
        if values["Active"]:
            FG.func.VF_active_fit(values)

        else:
            FG.func.VF_par_slider_move(values)

    elif event == 'Undo':
        if upd_num >= 0:
            undo()

    elif event in FG.func.Pnames:
        FG.init_par_overwritten(window,values,event)

    elif event == "Save as csv":
        table_df=pd.DataFrame(window["-TABLE-"].get(),columns=Vcol).drop("f_path", axis=1)
        name=filedialog.asksaveasfilename(title = "Save as csv",filetypes =  [("csv file","*.csv")])
        if name !="":
            table_df.to_csv(name,index=False)

    elif event == "Save as pkl":
        name=filedialog.asksaveasfilename(title = "Save as pkl",filetypes =  [("pkl file","*.pkl")])
        if name != "":
            with open(name, "wb") as f:
                pkl.dump({
                "T_oya" : T_oya,
                "Dset" : Dset,
                "apc" : apc,
                "Vcol":Vcol
                },f)

    elif event == "Load pkl":
        name = filedialog.askopenfilename(title = "Load pkl")
        print(name)
        if name != "":
            with open (name,"rb") as f:
                loaded = pkl.load(f)
            print(loaded["Vcol"])
            T_oya = loaded["T_oya"]
            Dset = loaded["Dset"]
            apc = loaded["apc"]
            Vcol = loaded["Vcol"]
            upd_num = T_oya["upd_num"].max() + 1
            table_update_contents(values["sort"])


[['&File', ["&Save as csv", "Save as pkl", "Undo", 'E&xit']]]
