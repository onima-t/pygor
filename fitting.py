import numpy as np
import PySimpleGUI as sg
import pandas as pd
import os
import tkinter
from tkinter import filedialog
import lmfit as lf
import matplotlib.patches as patches
from matplotlib import pyplot as plt

SLIDER_DEVIDE = 1000 #sliderを刻む数
PLOT_RESOLUTION = 1000
PLOT_INTERVAL = 1/30
YLIM_EXPAND = 0.05

class Fit:
    l,m,r=None,None,None
    def __init__(self, func, name, par_info):
        global ax
        self.func = func#関数
        self.name = name#表示名
        self.Pnames=[]#names of the parameters ["p0", "p1", ...]
        self.init_params={}#fittingの初期条件["p0":p0, p1":p1, ...]
        self.slider_params={}#sliderの最大最小の初期値["p0":[p0min,p0max], p1":[p1min,p1max], ...]
        self.vary = {}#the parameter is variable or bounded ["p0":True, "p1":False, ...]
        for i in par_info:
            self.Pnames.append(i[0])
            self.init_params[i[0]] = i[1]
            self.slider_params[i[0]] = i[2:]
            self.vary[i[0]] = True
        self.columns=self.Pnames + [i+"_stderr" for i in self.Pnames]
        self.columns_init=[i+"_init_val" for i in self.Pnames] + [i+"_fix" for i in self.Pnames] \
            + [i+"_bound_min" for i in self.Pnames] + [i+"_bound_max" for i in self.Pnames]
        self.columns_data = [self.name+"_xdata",self.name+"_ydata",self.name+"_Data_type",self.name+"_fit_range_min",self.name+"_fit_range_max"]
        ax=None
        self.FLAG_AX_EXISTS=False
        self.FLAG_SHOW_FITTING_RESULT=False
        self.FLAG_SHOW_PLOT_INIT_PARAMS=False

    def fit(self, x_data, y_data, values, mode=None):
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
        if mode==None:
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
        elif mode=="Active":
            best_vals={}
            for i in model.param_names:
                best_vals[i] = res[i].value
            return best_vals

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

        l=ax.plot(x_l,y_l, color="black")[0]
        m=ax.plot(x_m,y_m, color="r", label ="Fitting result")[0]
        r=ax.plot(x_r,y_r, color="black")[0]
        return l,m,r

    def save_params(self,values):
        for i in self.Pnames:
            self.init_params[i] = float(values[i])
            self.slider_params[i] = [float(values[i+"_Min"]), float(values[i+"_Max"])]
            self.vary[i] = True if values[i+"_Fix"]=="Bound" else False

    def plot_frange(self,values):
        global pl_vline,pl_rect,fmin,fmax
        fmin,fmax = float(values["rep_frange_min"]),float(values["rep_frange_max"])
        plt_ymin,plt_ymax = ax.get_ylim()
        plt_height = plt_ymax - plt_ymin
        pl_vline = ax.vlines([fmin,fmax],plt_ymin-1000*plt_height,plt_ymax+1000*plt_height,color="b",linewidth=1,alpha=0.3)
        pl_rect = ax.add_patch(patches.Rectangle(xy=(fmin,1000*plt_ymin), width=fmax-fmin, height =2000*plt_height, fc="b",fill=True,alpha=0.25))

    def VF_init(self,axes,xd,yd,values):
        global ymin,ymax,pl,ax,xdata,ydata
        self.FLAG_AX_EXISTS=True
        ax,xdata,ydata = axes,xd,yd
        ax.scatter(xdata,ydata,c="black", alpha=0.40)
        #self.xmin, self.xmax = np.min(xdata), np.max(xdata)
        self.x=np.linspace(np.min(xdata),np.max(xdata),PLOT_RESOLUTION)
        par={}
        for i in self.Pnames:
            par[i] = float(values[i])
        y_width=np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-y_width*YLIM_EXPAND
        ymax = np.max(ydata)+y_width*YLIM_EXPAND
        ax.set_ylim(ymin,ymax)
        self.plot_frange(values)
        if values["Active"]:
            self.VF_active_fit(values)
        else:
            pl = ax.plot(self.x, self.func(self.x,**par), color="b")[0]
            self.FLAG_SHOW_PLOT_INIT_PARAMS=True
        plt.pause(PLOT_INTERVAL)


    def VF_par_slider_move(self,values):
        global pl
        if self.FLAG_AX_EXISTS:
            if self.FLAG_SHOW_FITTING_RESULT:
                l.remove()
                m.remove()
                r.remove()
                self.FLAG_SHOW_FITTING_RESULT=False
            if self.FLAG_SHOW_PLOT_INIT_PARAMS:
                pl.remove()
            par={}
            for i in self.Pnames:
                par[i] = float(values[i])

            pl = ax.plot(self.x, self.func(self.x,**par), color="b")[0]
            self.FLAG_SHOW_PLOT_INIT_PARAMS=True
            plt.pause(PLOT_INTERVAL)

    def VF_frange_slider_move(self,values):
        if self.FLAG_AX_EXISTS:
            pl_vline.remove()
            pl_rect.remove()
            self.plot_frange(values)
            plt.pause(PLOT_INTERVAL)

    def VF_frange_slider_move_(self,values):
        if self.FLAG_AX_EXISTS:
            pl_vline.remove()
            pl_rect.remove()
            self.plot_frange(values)

    def VF_active_fit(self,values):
        global l,m,r
        if self.FLAG_AX_EXISTS:
            if self.FLAG_SHOW_PLOT_INIT_PARAMS:
                pl.remove()
                self.FLAG_SHOW_PLOT_INIT_PARAMS=False
            Y=ydata[xdata<fmax]
            X=xdata[xdata<fmax]
            Y=Y[X>fmin]
            X=X[X>fmin]
            try:
                fit_res_params = self.fit(X,Y,values, mode="Active")
            except (TypeError,ValueError) as e:
                print(e)
                return None
            if self.FLAG_SHOW_FITTING_RESULT:
                l.remove()
                m.remove()
                r.remove()
            l,m,r = self.plot_result(ax,fit_res_params,np.min(xdata),np.max(xdata),fmin,fmax)
            self.FLAG_SHOW_FITTING_RESULT=True
            plt.pause(PLOT_INTERVAL)
            self.test=l





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
        self.par_bounds={}
        def param_setting(param_name):
            sp=F.slider_params[param_name]
            ip=F.init_params[param_name]
            slider_val = (ip-sp[0])/(sp[1] - sp[0])
            T_s=(7, 1)
            self.par_bounds_names.append(param_name + "_Min")
            self.par_bounds_names.append(param_name + "_Max")
            self.par_bounds[param_name+"_Min"]=sp[0]
            self.par_bounds[param_name+"_Max"]=sp[1]
            return sg.Frame(param_name, layout=[
                    [
                        sg.Text("Init param", size=T_s),
                        sg.Slider([0,1], slider_val, 1/SLIDER_DEVIDE, orientation="h", disable_number_display=True,
                        size=(20,15), key=param_name+"_slider", enable_events=True),
                        sg.Input(ip, size=(5,1), key=param_name, enable_events=True)
                    ],
                    [
                        sg.Text("Min/Max", size=T_s),
                        sg.InputText(default_text=sp[0], size=(5,15), enable_events=True, key=param_name + "_Min", pad=((0,10),(0,0))),
                        sg.InputText(default_text=sp[1], size=(5,15), enable_events=True, key=param_name + "_Max"),
                        sg.OptionMenu(["Bound","Fix"], default_value="Bound" if F.vary[param_name] else "Fix", key=param_name + "_Fix")
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
                sg.Slider([0,1], 1, 1/SLIDER_DEVIDE, size=(30,15), disable_number_display=True,
                    key="frange_max", orientation="h", enable_events=True),
                sg.Input("1", size=(5,1), key="rep_frange_max", )
            ],
            [
                sg.Text("Min", size=(5,1)),
                sg.Slider([0,1], 0, 1/SLIDER_DEVIDE, size=(30,15), disable_number_display=True,
                    key="frange_min", orientation="h", enable_events=True),
                sg.Input("0", size=(5,1), key="rep_frange_min", )
            ]
        ]

        S = (8, 1)
        output=[
            [
                sg.Button("Visualize", key="Visual fit", size=S, disabled=True),
                sg.CBox("Active fit", key="Active", enable_events=True)
            ],
            [sg.Button("Fit",size=S)]
        ]

        return [
            [
                sg.Listbox(values=self.func_list, default_values=self.func_list[0], size=(10,6), key="func_list",
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True),
                sg.Frame("", element_justification="left", border_width=0, layout=[
                        [sg.Frame("", controler, element_justification="left", border_width=0)],
                        [
                            sg.Frame("Fit range", set_f_range, element_justification="left"),
                            sg.Frame("", output, element_justification="left",border_width=0)
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

    def frange_slider_move(self,window,frange_max,frange_min,event):
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

    def par_slider_move(self,window,values,event):
        pname=event.split("_")[0]
        B=self.par_bounds
        val = float(values[pname+"_slider"])*(B[pname+"_Max"]-B[pname+"_Min"]) + B[pname+"_Min"]
        window[pname].update(value=val)
        #window[pname].update(value="{:.3e}".format(val))
        self.func.init_params[pname] = val

    def par_range_overwritten(self,window,values,event):
        B=self.par_bounds.copy()
        self.test = window[event].get()
        self.test1 = window[event].get() in ["","-"]
        try:
            if not self.test1:
                print("aaaa")
                self.par_bounds[event]=float(window[event].get())

            sep=event.split("_")
            val = values[sep[0]+"_slider"]*(B[sep[0]+"_Max"]-B[sep[0]+"_Min"]) + B[sep[0]+"_Min"]
            try:
                slider_val = (val - self.par_bounds[sep[0]+"_Min"])/(self.par_bounds[sep[0]+"_Max"] - self.par_bounds[sep[0]+"_Min"])
            except ZeroDivisionError:
                slider_val = 0

            if sep[1] == "Max":
                new_max=self.par_bounds[event]
                old_min=float(window[sep[0]+"_Min"].get())
                if new_max<old_min:
                    self.par_bounds[sep[0]+"_Min"]=new_max
                    window[sep[0]+"_Min"].update(value=new_max)
                    window[sep[0]+"_slider"].update(value=slider_val)
                else:
                    window[sep[0]+"_slider"].update(value=slider_val)
                if new_max < self.func.init_params[sep[0]]:
                    self.func.init_params[sep[0]]=new_max
                    window[sep[0]].update(value=new_max)

            elif sep[1] == "Min":
                new_min=self.par_bounds[event]
                old_max=float(window[sep[0]+"_Max"].get())
                if new_min>old_max:
                    self.par_bounds[sep[0]+"_Max"]=new_min
                    window[sep[0]+"_Max"].update(value=new_min)
                    window[sep[0]+"_slider"].update(value=slider_val)
                else:
                    window[sep[0]+"_slider"].update(value=slider_val)
                if new_min > self.func.init_params[sep[0]]:
                    self.func.init_params[sep[0]]=new_min
                    window[sep[0]].update(value=new_min)

            self.par_slider_move(window,values,event)
        except ValueError as e:
            window[event].update(value=self.par_bounds[event])
            print(e)

    def init_par_overwritten(self,window,values,pname):
        #入力値がいい感じになってるか判定
        if values[pname] in ["","-"]:
            input = self.func.init_params[pname]
        else:
            try:
                input = float(values[pname])
            except ValueError:
                input = self.func.init_params[pname]

        if input<float(values[pname + "_Min"]):
            self.par_bounds[pname+"_Min"]=input
            window[pname+"_Min"].update(value=input)
        if input>float(values[pname + "_Max"]):
            self.par_bounds[pname+"_Max"]=input
            window[pname+"_Max"].update(value=input)
        try:
            slider_val = (input - self.par_bounds[pname+"_Min"])/(self.par_bounds[pname+"_Max"] - self.par_bounds[pname+"_Min"])
        except ZeroDivisionError as e:
            print(e)
            slider_val = 0

        self.func.init_params[pname] = input
        window[pname+"_slider"].update(value=slider_val)


"""
#template for fitting function settings

hoge_set={
    "func" : function,
    "name" : func_name,
    "par_info" : [
        ["p0", p0_init, p0_min, p0_max],
        ["p1", p1_init, p1_min, p1_max],
        ...
    ]
}

[[注意]]パラメータ名には _ を使わないこと

関数の定義では変数名に必ず x を使うこと(lmfitのフィットの際に用いるため)
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

def sqrt(x, p0, p1, offset):
    return np.sqrt(p0+p1*x) + offset
sqrt_set = {
    "func" : sqrt,
    "name" : "Sqrt",
    "par_info" : [
        ["p0", 0, -1, 1],
        ["p1", 1, -5, 5],
        ["offset", 0, -0.5, 0.5]
    ]
}

def poly(x,p0,p1):
    return p0 + p1*x
poly_set = {
    "func" : poly,
    "name" : "Linear",
    "par_info" : [
        ["p0", 0, -1, 1],
        ["p1", 1, -5, 5],
    ]
}

def gaussian(x,mean,sigma,gamp):
    if sigma == 0:
        return 0
    else:
        return gamp*np.exp(-np.square(x-mean)/(2*np.square(sigma)))
gaussian_set = {
    "func" : gaussian,
    "name" : "Gauss",
    "par_info" : [
        ["mean", 0, -1, 1],
        ["sigma", 1, 0.1, 10],
        ["gamp", 1, 0.1, 10]
    ]
}

def norm_dist(x,mean,sigma):
    if sigma == 0:
        return 0
    else:
        amplitude = 1/(np.sqrt(2*np.pi*np.square(sigma)))
        return amplitude*np.exp(-np.square(x-mean)/(2*np.square(sigma)))
norm_dist_set = {
    "func" : norm_dist,
    "name" : "Norm_dist",
    "par_info" : [
        ["mean", 0, -1, 1],
        ["sigma", 1, 0.1, 10],
    ]
}

def lorentzian(x,x0,d,l_amp):
    if d == 0:
        return 0
    else:
        return amp*np.square(d)/(np.square(x-x0) + np.square(d))
lorentzian_set = {
    "func" : lorentzian,
    "name" : "Lorentzian",
    "par_info" : [
        ["x0", 0, -1, 1],
        ["d", 1, 0.1, 10],
        ["l_amp", 1, 0.1, 10]
    ]
}



FIT_FUNCS=[
Fit(**poly_set),
Fit(**sqrt_set),
Fit(**gaussian_set),
Fit(**norm_dist_set),
Fit(**lorentzian_set),
Fit(**dynes_set)
]
