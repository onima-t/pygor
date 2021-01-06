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
import sympy as sm
import add_column as ac

sm.init_printing()


app_name = "pygor"
D_norm = 0.00046373
ax_Ymax = 1.5
ax_Ymin = 0
Pname = ["Delta", "Gamma", "Coef", "Offset"]
Pname_c = [i + "_c" for i in Pname]
# initial parameter range which user can select by slider
Prange = {"Delta": (0, 4000), "Gamma": (0, 5000),
          "Coef": (0.5, 1.5), "Offset": (-0.5, 0.5)}
cm = plt.cm.get_cmap('gist_rainbow_r')
df = pd.DataFrame(columns=["fit num", "filename", "points", "T",
                           "Delta", "Gamma", "Coef", "Offset", "s_D", "s_G", "s_C", "s_O"])
T_oya = pd.DataFrame()
Vcol = ["f_path", "filename", "counts"]
apc=ac.Appearence_controler()

columns_cnd = ["fit num", "filename", "fmax", "fmin", "init_D", "init_G", "init_C", "init_O",
               "bm_D", "bm_G", "bm_C", "bm_O", "bM_D", "bM_G", "bM_C", "bM_O"]
c4table = df.columns[1:12]


I_ax_label = r"V ($\mu \rm{eV}$)"
V_ax_label = r"I ($\mu \rm{A}$)"
ax_fontsize = 18
tick_labelsize = 15

#folder_path=sg.PopupGetFolder("Choose a Folder")


def dynes(E, delta, gamma, coef=1, offset=0):
    T = type(np.zeros(1))
    if type(E) != T and type(gamma) != T:
        A = complex(E, -gamma)
    else:
        A = np.empty_like(E).astype(np.complex)
        A.real = E
        A.imag = -gamma
    return coef * np.abs(np.real(A / (A**2 - delta**2)**0.5)) + offset


def differentiate(x, y):
    d1 = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d2 = (d1[1:] + d1[:-1]) * 0.5
    Diff = np.array([d1[0]])
    Diff = np.append(Diff, d2)
    Diff = np.append(Diff, [d1[-1]])
    return Diff / D_norm


def path_list(path, fe=None):
    files = os.listdir(path)
    if fe != None:
        files_dir = [f for f in files if (os.path.isfile(
            os.path.join(path, f)) and f[-len(fe):] == fe)]
    else:
        files_dir = [f for f in files if os.path.isfile(os.path.join(path, f))]
    return files_dir


def Init_ax(fig, xlabel, ylabel, ymin=0, ymax=1.5, place=111):
    ax = fig.add_subplot(place)
    ax.set_xlabel(xlabel, fontsize=ax_fontsize)
    ax.set_ylabel(ylabel, fontsize=ax_fontsize)
    ax.tick_params(labelsize=tick_labelsize)
    # ax.set_ylim(ymin,ymax)
    ax.grid(color='black', linestyle='dashed', linewidth=0.5)
    return ax


def pl_remove(pl, pl_bool):
    if pl_bool == True:
        pl.remove()
        pl_bool = False


def pl_dynes(ax, min, max, params, color, num_p=300, LS="-", label=""):
    x = np.linspace(min, max, num_p)
    y = dynes(x, params[0], params[1], params[2], params[3])
    pl = ax.plot(x, y, color=color, linestyle=LS, label=label)
    return pl


def check_bounds(name_c):
    name = name_c[:-2]
    if values[name_c] == True:
        M = float(values[name + "_Max"])
        m = float(values[name + "_Min"])
        window[name].update(range=(m, M))
    else:
        window[name].update(range=Prange[name])


def get_bounds():
    m = []
    M = []
    for i in Pname:
        m.append(values[i + "_Min"] if values[i + "_c"] else -np.inf)
        M.append(values[i + "_Max"] if values[i + "_c"] else np.inf)
    return m, M


def get_params():
    return [values['Delta'], values['Gamma'], values['Coef'], values['Offset']]


def prepare_fit(x, y):
    if f_range_c == True:
        fmin = values["Min"]
        fmax = values["Max"]
        y_f = y[x > fmin]
        x_f = x[x > fmin]
        return x_f[x_f < fmax], y_f[x_f < fmax]
    else:
        return x, y

def latest_df(df,idx_s, sort_order):
    if df.values.shape !=(0,0):
        idx = df.groupby("f_path")["upd_num"].transform(
            max) == df["upd_num"]  # 最新のindexを取得
        return df[idx].sort_values(idx_s, ascending=not sort_order)
    else:
        return df

def table_update_data(idx_s, sel=None, sort_order=True):
    global window
    a=latest_df(T_oya, idx_s, sort_order)
    apc.renew_data(a,Vcol)
    a=apc.df_show

    window.close()
    window = sg.Window(app_name, layout(Vcol), location=(0, 0), finalize=True)

    window["-TABLE-"].update(values=a[Vcol].values.tolist(), select_rows=sel)
    window["Visual fit"].update(disabled=True)

def table_update_order(idx_s, sel=None, sort_order=True):
    global window
    a=latest_df(T_oya, idx_s, sort_order)
    apc.renew_data(a,Vcol)
    a=apc.df_show

    window["-TABLE-"].update(values=a[Vcol].values.tolist(), select_rows=sel)
    window["Visual fit"].update(disabled=True)

def ref_data_col():
    l = []
    for i in values["-TABLE-"]:
        data = Dset[window["-TABLE-"].get()[i][0]][1]
        l.append(data)
    stacked = pd.concat(l, join="inner")
    col = stacked.columns.values.tolist()

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


def get_data(paths):
    global upd_num, T_oya, Dset, D
    Tadd = []
    for fp in paths:
        D = pd.read_csv(fp, sep=" |\t|,", engine="python")
        Dset[fp] = [fp.split("/")[-1], D]
        Tadd.append([upd_num, fp, fp.split("/")[-1], len(D)])
    _T = pd.DataFrame(
        Tadd, columns=["upd_num", "f_path", "filename", "counts"])
    T_oya = T_oya.append(_T)
    table_update_data(values["sort"], sel=values["-TABLE-"],
                      sort_order=values["sort_order"])
    upd_num += 1


def undo():
    global upd_num, df
    df = df[df["fit num"] != fit_num]
    table_update(values["sort"])
    fit_num -= 1


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


def pl_delta(e_c=True):
    idx = df.groupby("filename")["fit num"].transform(max) == df["fit num"]
    latest = df[idx]
    latest = latest[["T", "Delta", "Gamma", "s_D", "s_G"]].dropna(
        how="any")  # 行に一つでも欠損値があればlatestから削除
    Temp = latest["T"].to_numpy()
    Delta = latest["Delta"].to_numpy()
    s_D = latest["s_D"].to_numpy()**(1 / 2)
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = Init_ax(fig, "T (K)", r"$\Delta \ (\mu \rm{eV})$", ymin=0, ymax=np.max(
        Delta) * 1.1, place=111)
    if e_c == True:
        ax.errorbar(x=Temp, y=Delta, yerr=s_D, fmt="o",
                    capsize=2, capthick=0.5, ecolor="black")
    elif e_c == False:
        ax.errorbar(x=Temp, y=Delta, fmt="o")
    plt.pause(0.1)


def pl_check():
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = Init_ax(fig, V_ax_label, "norm_dI/dV", place=111)
    sel_num = values["-TABLE-"][0]

    idx = df.groupby("filename")["fit num"].transform(
        max) == df["fit num"]  # 最新のfitのindexを取得
    a = df[idx]
    pa = a[a["filename"] == window["-TABLE-"].get()[sel_num][0]
           ].values[0][4:8].tolist()
    a = a[a["filename"] == window["-TABLE-"].get()[sel_num][0]
          ].values[0][-2:].tolist()

    pl_num = fn_l.index(window["-TABLE-"].get()[sel_num][0])
    T = data[pl_num][0].to_numpy()
    V = data[pl_num][1].to_numpy()
    I = data[pl_num][2].to_numpy()
    Diff = differentiate(V, I)
    ax.scatter(V, Diff, c="#c8326d", vmin=1, vmax=9,
               label="Data points", edgecolors="black", linewidth=0.5)
    pl_dynes(ax, a[1], np.min(V), pa, "y")
    pl_dynes(ax, np.max(V), a[0], pa, "y")
    pl_dynes(ax, a[0], a[1], pa, "blue", label="Fitting curve")
    ax.legend()
    plt.pause(0.1)


def slider(name, range, init, width, S=(15, 15), D=False):
    S = sg.Slider(range, init, width, orientation='h',
                  size=S, key=name, enable_events=True, disable_number_display=D)
    T = sg.Text(name, size=(6, 1))
    return [T, S]


def slider_(name, range, init, width, S=(15, 15)):
    S = sg.Slider(range, init, width, orientation='h', tick_interval=5000,
                  size=S, key=name, disable_number_display=True, enable_events=True)
    T = sg.Text(name, size=(6, 1))
    return T, S


def input_range(title, defaulut_min, default_max):
    t = sg.Text(title, size=(6, 1))
    in1 = sg.InputText(default_text=defaulut_min, size=(
        5, 15), key=title + '_Min', pad=((0, 10), (0, 0)))
    in2 = sg.InputText(default_text=default_max,
                       size=(5, 15), key=title + "_Max")
    c = sg.CBox("", key=title + "_c", enable_events=True)
    return [t, in1, in2, c]


def layout(col):
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

    menu_def = [['&File', ["&Save", "Undo", 'E&xit']]]

    Browse = [  # sg.Text('Your Folder', size=(15, 1), justification='right'),
        sg.InputText('', key="path", enable_events=True), sg.FolderBrowse(
            key="File")
    ]

    S = (8, 1)


    buttons = [[sg.Button('Plot', size=S, disabled=True)],
               [sg.Button("Visual fit", size=S, disabled=True)],
               [sg.Button("Multi fit", size=S)],
               [sg.Button("Check", size=S, disabled=True)],
               [sg.Button("Delta plot", size=S)],
               [sg.CBox("Error?", key="de_c", default=True)]
               #[sg.Button("My fit")],
               ]

    def sel(axis):
        return sg.Frame("", border_width=0, element_justification="center", pad=(0,10), layout=[
            [sg.Text(axis)],
            [sg.Listbox([], size=(6, 6), select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, default_values="", key=axis)]
        ])

    sel_xyz = [
        [
        sel("x"),sel("y"),sel("z")
        ],
        [
        sg.OptionMenu(["Nomal", "Color", "3D"],key="plt_mode"),
        sg.Button("B-Plot")
        ]
    ]

    Init_s = [slider("Delta", Prange["Delta"], 1000, 1),
              slider("Gamma", Prange["Gamma"], 500, 1),
              slider("Coef", Prange["Coef"], 1, 0.01),
              slider("Offset", Prange["Offset"], 0, 0.01)]

    Set_p_bound = [[sg.Text("Min", pad=((70, 20), (0, 0))), sg.Text("Max")],
                   input_range("Delta", 0, 0),
                   input_range("Gamma", 0, 0),
                   input_range("Coef", 0, 0),
                   input_range("Offset", 0, 0)]

    Text, Sl = slider_("Min", (-10000, 10000), 0, 1, S=(30, 15))
    Set_f_range = [slider("Max", (-10000, 10000), 0, 1, S=(30, 15), D=True),
                   [Text, Sl, Fit_range_button]]

    Fit_controler = [[sg.Frame("Initial parameter", Init_s, pad=((0, 5), (0, 5))), sg.Frame("Parameter bounds", Set_p_bound)],
                     [sg.Frame("Fit range", Set_f_range)],
                     [sg.Button('Delete', size=(10, 1), disabled=True), sg.Button('Fit', size=(10, 1)), sg.CBox("Active fit?", key="act_f_c")]]



    """
    GUI のレイアウトの設定
    """
    return [[sg.Menu(menu_def, tearoff=False)],
            [sg.Text("拡張子", size=(5, 1)), sg.Input(".txt", key="fe", size=(5, 1)),
             sg.Input("", key="folder", enable_events=True, visible=False),
             sg.FolderBrowse(button_text="Add folder", key="add_folder"),
             sg.FilesBrowse(button_text="Add data files",
                            key="add_files", target="names", enable_events=True),
             sg.Text("Sort by"),
             sg.Combo(Vcol, enable_events=True,
                      default_value="filename", key="sort"),
             sg.CBox("Reverse order", key="sort_order", enable_events=True),
             sg.Input("", key="names", enable_events=True, visible=False), sg.Button("test")],
            # Browse,
            [sg.Table(
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
                display_row_numbers = True), sg.Button("test2"), sg.Button("test3")],
            [sg.Text('_' * 120)],
            [
            sg.Frame("", buttons, key="buttons", border_width=1,),
            sg.Frame("", sel_xyz, element_justification="center"),
            sg.Frame("", Fit_controler, border_width=0)
            ]
        ]


window = sg.Window(app_name, layout(Vcol), location=(0, 0), finalize=True)


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

    elif event == "test":
        print(values["names"])
        print(values["x"])
        window["x"].update(set_to_index=0)

    elif event == "test2":
        Vcol.append("hoge")
        T["hoge"] = T["f_path"]
        table_update_data(values["sort"], sel=values["-TABLE-"], sort_order=values["sort_order"])

    elif event == "test3":
        if Dset!={}:
            #AAA = latest_df(T_oya, values["sort"], values["sort_order"])
            apc, Vcol[1:] = ac.col_cnt(latest_df(T_oya, values["sort"], values["sort_order"]), Dset, Vcol[1:])
            apc.df["upd_num"] = upd_num
            T_oya = T_oya.append(apc.df)
            table_update_data(values["sort"], sel=values["-TABLE-"], sort_order=values["sort_order"])

            upd_num+=1
        else:
            sg.PopupError("No data exists")

    elif event == "-TABLE-":
        if len(values["-TABLE-"]) == 0:
            window["Visual fit"].update(disabled=True)
            window["Plot"].update(disabled=True)
            window["Check"].update(disabled=True)
        else:
            if len(values["-TABLE-"]) == 1:
                window["Visual fit"].update(disabled=False)
                window["Plot"].update(disabled=False)
                """
                if window["-TABLE-"].get()[ values["-TABLE-"][0] ][3] >0:
                    window["Check"].update(disabled=False)
                else:
                    window["Check"].update(disabled=True)
                """
            else:
                window["Visual fit"].update(disabled=True)
                window["Plot"].update(disabled=False)
                window["Check"].update(disabled=True)

            ref_data_col()

    elif event in ["sort", "sort_order"]:
        table_update_order(values["sort"], sort_order=values["sort_order"])

    elif event == "Select all":
        window["-TABLE-"].update(select_rows=np.arange(len(window["-TABLE-"].get())))

    elif event == "Visual fit":
        vf_row = values["-TABLE-"][0]
        vf_name = window["-TABLE-"].get()[vf_row][0]
        d_num = fn_l.index(vf_name)
        data_f = data[d_num]
        V_vis = data_f[1].to_numpy()
        I = data_f[2].to_numpy()
        Diff_vis = differentiate(V_vis, I)

        fig = plt.figure(figsize=(10, 7), dpi=100)
        ax = Init_ax(fig, V_ax_label, "norm_dI/dV")
        fig.canvas.mpl_connect('button_press_event', onclick)

        pl_d = ax.scatter(V_vis, Diff_vis, c="#5fd989", edgecolors="black")
        w_V = np.max(V_vis) - np.min(V_vis)
        ax.set_xlim(np.min(V_vis) - w_V * 0.05, np.max(V_vis) + w_V * 0.05)
        plt.pause(0.1)

    elif event == 'Plot':
        fig = plt.figure(figsize=(14, 7), dpi=100)
        ax_I = Init_ax(fig, values["x"][0], values["y"][0], place=121)
        ax_d = Init_ax(fig, values["x"][0], "dy/dx", place=122)
        data_set = pd.DataFrame()

        for i in values["-TABLE-"]:
            data = Dset[window["-TABLE-"].get()[i][0]][1].copy()
            xy = data[[values["x"][0], values["y"][0]]].values
            data["diff"] = pd.Series(differentiate(
                xy[:, 0], differentiate(xy[:, 0], xy[:, 1])))
            data_set = data_set.append(data)

        data_set = data_set.sort_values(values["z"][0]).dropna(how="any")
        x = data_set[values["x"][0]].values
        y = data_set[values["y"][0]].values
        z = data_set[values["z"][0]].values
        Diff = data_set["diff"]

        pl_d = ax_d.scatter(x, Diff, c=z, cmap=cm,edgecolors="black", linewidth=0.5)
        pl_I = ax_I.scatter(x, y, c=z, cmap=cm,edgecolors="black", linewidth=0.5)

        fig.colorbar(pl_I, ax=ax_I)
        plt.pause(0.1)

    elif event == "B-Plot":
        fig = plt.figure(figsize=(8, 7), dpi=100)
        ax = Init_ax(fig, values["x"][0], values["y"][0], place=111)
        for i in values["-TABLE-"]:
            data = Dset[window["-TABLE-"].get()[i][0]][1].copy()
            x = data[values["x"][0]].values
            y = data[values["y"][0]].values
            z = data[values["z"][0]].values
            if values["plt_mode"] == "Nomal":
                ax.scatter(x,y,edgecolors="black",linewidth=0.5, alpha=0.40, label=Dset[window["-TABLE-"].get()[i][0]][0])
                ax.legend()
            if values["plt_mode"] == "Color":
                ax.scatter(x,y,c=z,cmap=cm,edgecolor="black",linewidth=0.40,alpha=0.3)

        if values["plt_mode"] == "Color":
            plt.colorbar(ax=ax)
        plt.pause(0.1)



    elif event in Pname:
        if ax != None:
            if pl_p_show == True:
                pl_p[0].remove()
            else:
                pl_p_show = True
            pl_p = pl_dynes(ax, np.min(V_vis), np.max(
                V_vis), get_params(), "blue")
            plt.pause(0.1)

    elif event in Pname_c:
        check_bounds(event)

    elif event == 'Undo':
        if fit_num > 0:
            undo()
        else:
            sg.popup("There is no fitting result.")

    elif event == 'Delete':
        pl_select_show = False
        Diff_vis = np.delete(Diff_vis, Selected_points)
        V_vis = np.delete(V_vis, Selected_points)
        pl_d.remove()
        pl_select.remove()
        pl_d = ax.scatter(V_vis, Diff_vis, c="#5fd989", edgecolors="black")
        Selected_points = []
        plt.pause(0.1)
        window["Delete"].update(disabled=True)

    elif event == '-B-':                # if the normal button that changes color and text
        if f_range_c == True:
            if pl_f_range_show == True:
                pl_f_v.remove()
                pl_f_rect.remove()
                plt.pause(0.1)
                pl_f_range_show = False
        f_range_c = not f_range_c
        window['-B-'].update(text='On' if f_range_c else 'Off',
                             button_color='white on green' if f_range_c else 'white on red')

    elif event in ("Max", "Min"):
        if (f_range_c == True) and (ax != None):
            if pl_f_range_show == True:
                pl_f_v.remove()
                pl_f_rect.remove()
            fmin = float(values["Min"])
            fmax = float(values["Max"])
            pl_f_v = ax.vlines([fmin, fmax], ax_Ymin, ax_Ymax, color="blue")
            pl_f_rect = ax.add_patch(patches.Rectangle(xy=(
                fmin, 0), width=fmax - fmin, height=ax_Ymax - ax_Ymin, fc="b", fill=True, alpha=0.1))

            plt.pause(0.1)
            pl_f_range_show = True
            if values["act_f_c"] == True:
                try:
                    V_f, Diff_f = prepare_fit(V_vis, Diff_vis)
                except NameError as e:
                    print(e)
                    continue
                if pl_p_show == True:
                    pl_p[0].remove()
                    pl_p_show = False

                if pl_f_show == True:
                    pl_f_in[0].remove()
                    pl_f_show = False
                    if pl_f_lr_show == True:
                        pl_f_r[0].remove()
                        pl_f_l[0].remove()
                        pl_f_lr_show = False

                b_m, b_M = get_bounds()

                try:
                    popt, pcov = curve_fit(
                        dynes, V_f, Diff_f, p0=get_params(), bounds=(b_m, b_M))
                except (RuntimeError, ValueError) as e:
                    print(e)
                except TypeError as e:
                    print(e)
                else:
                    print(popt)
                    fmax, fmin = np.max(V_f), np.min(V_f)

                    pl_f_in = pl_dynes(ax, fmin, fmax, popt, color="r")
                    pl_f_show = True
                    if f_range_c == True:
                        pl_f_l = pl_dynes(ax, np.min(V_vis),
                                          fmin, popt, color="y", LS="-")
                        pl_f_r = pl_dynes(ax, fmax, np.max(
                            V_vis), popt, color="y", LS="-")
                        pl_f_lr_show = True
                    plt.pause(0.1)

    elif event == 'Fit':
        fit_num += 1
        try:
            V_f, Diff_f = prepare_fit(V_vis, Diff_vis)
        except NameError as e:
            print(e)
            continue
        if pl_p_show == True:
            pl_p[0].remove()
            pl_p_show = False

        if pl_f_show == True:
            pl_f_in[0].remove()
            pl_f_show = False
            if pl_f_lr_show == True:
                pl_f_r[0].remove()
                pl_f_l[0].remove()
                pl_f_lr_show = False

        init_p = get_params()
        b_m, b_M = get_bounds()

        try:
            popt, pcov = curve_fit(
                dynes, V_f, Diff_f, p0=init_p, bounds=(b_m, b_M))
        except (RuntimeError, ValueError) as e:
            print(e)
        else:
            print(popt)
            fmax, fmin = np.max(V_f), np.min(V_f)

            pl_f_in = pl_dynes(ax, fmin, fmax, popt, color="r")
            pl_f_show = True
            if f_range_c == True:
                pl_f_l = pl_dynes(ax, np.min(V_vis), fmin,
                                  popt, color="y", LS="-")
                pl_f_r = pl_dynes(ax, fmax, np.max(V_vis),
                                  popt, color="y", LS="-")
                pl_f_lr_show = True
            plt.pause(0.1)

            f_res = [fit_num] + table[d_num] + \
                popt.tolist() + np.diag(pcov).tolist()
            f_cnd.append([fit_num] + table[d_num][:1] + [np.max(V_f), np.min(V_f)] +
                         init_p + b_m + b_M)
            df = df.append(pd.Series(f_res, index=df.columns),
                           ignore_index=True)

            # visual fit時のfilenameが現在何列目にあるか？
            a = np.where(
                np.array(window["-TABLE-"].get())[:, 0] == vf_name)[0][0]
            table_update(values["sort"], sel=[a],
                         sort_order=values["sort_order"])

    elif event == "Multi fit":
        fit_num += 1
        selected_row = values["-TABLE-"]
        b_m, b_M = get_bounds()
        init_p = get_params()
        f_res = []
        for i in selected_row:
            pl_num = fn_l.index(window["-TABLE-"].get()[i][0])
            T = data[pl_num][0].to_numpy()
            V = data[pl_num][1].to_numpy()
            I = data[pl_num][2].to_numpy()
            Diff = differentiate(V, I)

            V_f, Diff_f = prepare_fit(V, Diff)
            try:
                popt, pcov = curve_fit(
                    dynes, V_f, Diff_f, p0=init_p, bounds=(b_m, b_M))
            except (RuntimeError, ValueError) as e:
                print(e)
            else:
                f_res.append([fit_num] + table[pl_num] + popt.tolist() +
                             np.diag(pcov).tolist())
                f_cnd.append([fit_num] + table[pl_num][:1] + [np.max(V_f), np.min(V_f)] +
                             init_p + b_m + b_M)
        df = df.append(pd.DataFrame(
            f_res, columns=df.columns), ignore_index=True)
        table_update(values["sort"], sel=selected_row,
                     sort_order=values["sort_order"])

    elif event == "Delta plot":
        pl_delta(values["de_c"])

    elif event == "Check":
        pl_check()

    elif event == "Save":
        save_data()

    elif event == "Change":
        img_ch = random.uniform(0, 100)  # 0から100の間で乱数を生成
        if img_ch < 48:
            window["img"].update(data=picture.B)
        elif img_ch < 96:
            window["img"].update(data=picture.A)
        else:
            window["img"].update(data=picture.C)


"""
    elif event == "My fit":
        fit_num+=1
        G=get_params()
        AAA=np.array(window["-TABLE-"].get())
        a=pd.DataFrame(AAA[values["-TABLE-"],:3],columns=df.columns[1:4])#Tまでget
        a["fit num"]=fit_num
        a=a.assign(Delta=G[0],Gamma=G[1],Coef=G[2],Offset=G[3])
        a[df.columns[-6:]]=0
        a=a.astype({"points":float, "T":float})
        df = df.append(a,ignore_index=True)
        table_update(values["sort"],sel=values["-TABLE-"],sort_order=values["sort_order"])
"""
