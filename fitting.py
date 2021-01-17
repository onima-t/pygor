import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
import lmfit as lf
from matplotlib import pyplot as plt

SLIDER_DEVIDE = 1000 #sliderを刻む数
PLOT_RESOLUTION = 1000


class Fit:
    def __init__(self, func, name, par_info):
        self.func = func#関数
        self.name = name#表示名
        self.Pnames=[]#names of the parameters ["p0", "p1", ...]
        self.init_params={}#fittingの初期条件["p0":p0, p1":p1, ...]
        self.slider_params={}#sliderの最大最小["p0":[p0min,p0max], p1":[p1min,p1max], ...]
        self.vary = {}#the parameter is variable or bounded ["p0":True, "p1":False, ...]
        for i in par_info:
            self.Pnames.append(i[0])
            self.init_params[i[0]] = i[1]
            self.slider_params[i[0]] = i[2:]
            self.vary[i[0]] = True
        self.columns=self.Pnames + [i+"_stderr" for i in self.Pnames]
        self.columns_init=[i+"_init_val" for i in self.Pnames] + [i+"_fix" for i in self.Pnames] \
            + [i+"_bound_min" for i in self.Pnames] + [i+"_bound_max" for i in self.Pnames]
        self.ax=None

    def fit(self, x_data, y_data, values):
        model = lf.Model(self.func)
        params = model.make_params()
        for i in model.param_names:
            params[i].set(
                value=float(values[i]),
                min=float(values[i+"_Min"]),
                max=float(values[i+"_Max"]),
                vary=True if values[i+"_Fix"]=="Bound" else False
            )
        self.result = model.fit(x=x_data, data=y_data, params=params, method='leastsq')
        res=self.result.result.params
        best_vals=[]
        errors=[]
        init=[]
        fix=[]
        bm=[]
        bM=[]
        for i in model.param_names:
            best_vals.append(res[i].value)
            errors.append(res[i].stderr)
            init.append(self.result.init_params[i].value)
            fix.append(not self.result.init_params[i].vary)
            bm.append(self.result.init_params[i].min)
            bM.append(self.result.init_params[i].max)

        return best_vals + errors, init + fix + bm + bM

    def plot_result(self,ax,params,xmin,xmax,fmin,fmax):
        P={}
        for i in self.Pnames:
            P[i]=params[i]

        x_l=np.linspace(xmin,fmin,PLOT_RESOLUTION)
        x_m=np.linspace(fmin,fmax,PLOT_RESOLUTION)
        x_r=np.linspace(fmax,xmax,PLOT_RESOLUTION)
        y_l=self.func(x_l,**P)
        y_m=self.func(x_m,**P)
        y_r=self.func(x_r,**P)

        ax.plot(x_l,y_l, color="black")
        ax.plot(x_m,y_m, color="r", label ="Fitting result")
        ax.plot(x_r,y_r, color="black")

    def save_params(self,values):
        for i in self.Pnames:
            self.init_params[i] = float(values[i])
            self.slider_params[i] = [float(values[i+"_Min"]), float(values[i+"_Max"])]
            self.vary[i] = True if values[i+"_Fix"]=="Bound" else False

    def VF_init(self,ax,xdata,ydata,values):
        self.ax=ax
        ax.scatter(xdata,ydata,edgecolors="black",linewidth=0.5, alpha=0.40)
        self.xmin, self.max = np.min(xdata), np.max(xdata)
        self.x=np.linspace(np.min(xdata),np.max(xdata),PLOT_RESOLUTION)
        par={}
        for i in self.Pnames:
            par[i] = float(values[i])
        self.pl = ax.plot(self.x, self.func(self.x,**par), color="b")[0]
        plt.pause(0.1)

    def VF_slider_move(self,values):
        if self.ax != None:
            self.pl.remove()
            par={}
            for i in self.Pnames:
                par[i] = float(values[i])

            self.pl = self.ax.plot(self.x, self.func(self.x,**par), color="b")[0]
            plt.pause(0.1)




class Fit_GUI:
    def __init__(self, Fit_list):
        self.func_list=[]
        self.fit_dict={}
        for i in Fit_list:
            self.func_list.append(i.name)
            self.fit_dict[i.name] = i

        self.frange_slider_max=1
        self.frange_slider_min=0

    def layout(self, sel_func):
        if sel_func == []:
            F=self.fit_dict[self.func_list[0]]
        else:
            F=self.fit_dict[sel_func[0]]

        self.func=F

        self.par_bounds_names=[]
        self.pre_bounds={}
        def param_setting(param_name):
            sp=F.slider_params[param_name]
            ip=F.init_params[param_name]
            slider_width = sp[1] - sp[0]
            T_s=(7, 1)
            self.par_bounds_names.append(param_name + "_Min")
            self.par_bounds_names.append(param_name + "_Max")
            self.pre_bounds[param_name+"_Min"]=sp[0]
            self.pre_bounds[param_name+"_Max"]=sp[1]
            return sg.Frame(param_name, layout=[
                    [
                        sg.Text("Init param", size=T_s),
                        sg.Slider(sp, ip ,slider_width/SLIDER_DEVIDE, orientation = "h",
                        size=(20,15), key=param_name, enable_events=True)
                    ],
                    [
                        sg.Text("Min/Max", size=T_s),
                        sg.InputText(default_text=sp[0], size=(5,15), enable_events=True, key=param_name + "_Min", pad=((0,10),(0,0))),
                        sg.InputText(default_text=sp[1], size=(5,15), enable_events=True, key=param_name + "_Max"),
                        sg.OptionMenu(["Bound","Fix","Free"], default_value="Bound" if F.vary[param_name] else "Fix", key=param_name + "_Fix")
                    ]
                ]
            )

        if len(F.Pnames)%2==0:
            controler=np.array([param_setting(i) for i in F.Pnames]).reshape(-1,2).tolist()
        else:
            controler=np.array([param_setting(i) for i in F.Pnames[:-1]]).reshape(-1,2).tolist()
            controler.append([param_setting(F.Pnames[-1])])

        set_f_range=[
            [
                sg.Text("Max", size=(5,1)),
                sg.Slider([0,1], 1, 2/SLIDER_DEVIDE, size=(30,15), disable_number_display=True,
                    key="frange_max", orientation="h", enable_events=True),
                sg.Input("1", size=(5,1), key="rep_frange_max", )
            ],
            [
                sg.Text("Min", size=(5,1)),
                sg.Slider([0,1], 0, 2/SLIDER_DEVIDE, size=(30,15), disable_number_display=True,
                    key="frange_min", orientation="h", enable_events=True),
                sg.Input("0", size=(5,1), key="rep_frange_min", )
            ]
        ]

        output=[
            [sg.CBox("Active fit", key="Active")],
            [sg.Button("Fit")]
        ]

        return [
            [
                sg.Listbox(values=self.func_list, default_values=self.func_list[0], size=(6,6), key="func_list",
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True),
                sg.Frame("", element_justification="left", border_width=0, layout=[
                        [sg.Frame("", controler, element_justification="left", border_width=0)],
                        [
                            sg.Frame("Fit range", set_f_range, element_justification="left"),
                            sg.Frame("", output, element_justification="right",border_width=0)
                        ]
                    ]
                )
            ]
        ]

    def frange_update(self, window, df=None, smin=None, smax=None):
        if type(df)!=type(None) and window["x"].get()!=[]:#tableとxデータが選択されているか
            sel_x = window["x"].get()[0]
            if df[sel_x].dtype!=np.object:#数字として扱えるデータが選択されているか
                dsr = df.describe()
                self.frange_slider_max = dsr.at["max",sel_x]
                self.frange_slider_min = dsr.at["min",sel_x]
                width = self.frange_slider_max - self.frange_slider_min
                min = self.frange_slider_min

                if smin ==None:
                    window["frange_min"].update(value=0)
                    window["rep_frange_min"].update(value=self.frange_slider_min)
                else:
                    window["frange_min"].update(value=smin)
                    window["rep_frange_min"].update(value=str(min + smin*width))
                if smax == None:
                    window["frange_max"].update(value=1)
                    window["rep_frange_max"].update(value=self.frange_slider_max)
                else:
                    window["frange_max"].update(value=smax)
                    window["rep_frange_max"].update(value=str(min + smax*width))

    def frange_slider_moved(self,window,frange_max,frange_min,event):
        width = self.frange_slider_max - self.frange_slider_min
        min = self.frange_slider_min
        sl_val_M = frange_max*width + min
        sl_val_m = frange_min*width + min
        if sl_val_M<sl_val_m:
            sl_val_m=sl_val_M
            if event == "frange_max":
                window["frange_min"].update(value=frange_max)
            else:
                window["frange_max"].update(value=frange_min)
        window["rep_frange_max"].update(value=str(sl_val_M))
        window["rep_frange_min"].update(value=str(sl_val_m))

    def prange_overwritten(self,window,event):
        try:
            if window[event].get()=="":
                self.pre_bounds[event]=0
            else:
                self.pre_bounds[event]=float(window[event].get())
            sep=event.split("_")
            if sep[1] == "Max":
                new_max=self.pre_bounds[event]
                old_min=float(window[sep[0]+"_Min"].get())
                if new_max<old_min:
                    window[sep[0]+"_Min"].update(value=new_max)
                    window[sep[0]].update(range=(new_max,new_max))
                else:
                    window[sep[0]].update(range=(old_min,new_max))
            elif sep[1] == "Min":
                new_min=self.pre_bounds[event]
                old_max=float(window[sep[0]+"_Max"].get())
                if new_min>old_max:
                    window[sep[0]+"_Max"].update(value=new_min)
                    window[sep[0]].update(range=(new_min,new_min))
                else:
                    window[sep[0]].update(range=(new_min,old_max))
        except ValueError:
            window[event].update(value=self.pre_bounds[event])



"""
#template for setting

Fset=[
    "func" :
    "name" :
    "par_info" : [
        ["p0", p0_init, p0_min, p0_max],
        ["p1", p1_init, p1_min, p1_max],
        ...
    ]
]

関数の定義では変数名に必ず x を使うこと（lmfit）のフィットの際に用いるため
→func.__code__.co_varnames で取得するようにする？
"""

def dynes(x, delta, gamma, amp=1, offset=0):
    T = type(np.zeros(1))
    if type(x) != T and type(gamma) != T:
        A = complex(x, -gamma)
    else:
        A = np.empty_like(x).astype(np.complex)
        A.real = x
        A.imag = -gamma
        return amp * np.abs(np.real(A / (A**2 - delta**2)**0.5)) + offset

def sqrt(x, p0, p1, offset):
    return np.sqrt(p0+p1*x) + offset

def poly(x,p0,p1):
    return p0 + p1*x



dynes_set={
    "func" : dynes,
    "name" : "Dynes",
    "par_info" : [
        ["delta", 1000, 0, 5000],
        ["gamma", 1000, 0, 10000],
        ["amp", 1, 0.5, 1.5],
        ["offset", 0, -0.5, 0.5],
    ]
}

sqrt_set = {
    "func" : sqrt,
    "name" : "Sqrt",
    "par_info" : [
        ["p0", 0, -1, 1],
        ["p1", 1, -5, 5],
        ["offset", 0, -0.5, 0.5]
    ]
}

poly_set = {
    "func" : poly,
    "name" : "poly",
    "par_info" : [
        ["p0", 0, -1, 1],
        ["p1", 1, -5, 5],
    ]
}

FIT_FUNCS=[Fit(**poly_set),Fit(**sqrt_set),Fit(**dynes_set)]