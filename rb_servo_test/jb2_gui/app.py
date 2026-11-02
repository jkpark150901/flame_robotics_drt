#!/usr/bin/env python3
"""Standalone jb2 experiment GUI, launch with run.sh."""
import csv
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from model import path_plan, path_profile, servo_plan, analyze, ideal_arrival_times, waypoint_proximity
from controller import execute, read_current
import timing_plot


class App(tk.Tk):
    def __init__(self):
        super().__init__();self.title('동시 관절 이동 · servo / jb2');self.geometry('1380x940')
        self.events=queue.Queue();self.stop=threading.Event();self.busy=False
        self.results=[];self.config=None;self.out=None
        top=ttk.Frame(self,padding=8);top.pack(fill='x');self.vars={}
        fields=[('ip','IP','10.0.2.7'),('base','시작 자세 J1..J6 [deg]','0 0 0 0 0 0'),
                ('end','목표 자세 J1..J6 [deg]','30 15 -10 20 5 -15'),
                ('segments','jb2 전체 분할 수','100'),('servo_period_ms','Servo 주기 [ms]','10'),('speed','속도 [deg/s]','10'),('accel','가속도 [deg/s²]','20'),
                ('blend','Blending (jb2)','50'),('gain','Gain (servo)','0.1'),('alpha','Alpha (servo)','0.5'),('servo_t2','Servo 유지시간 t2 [s]','0.1'),('hz','기록 요청 [Hz]','100'),
                ('window','속도 차분 창 [s]','0.1'),('timeout','단계 제한시간 [s]','120')]
        for i,(key,label,value) in enumerate(fields):
            r,c=divmod(i,3);self.vars[key]=tk.StringVar(value=value)
            ttk.Label(top,text=label).grid(row=r,column=2*c,sticky='e',padx=5,pady=3)
            ttk.Entry(top,textvariable=self.vars[key],width=26).grid(row=r,column=2*c+1,sticky='ew')
        route=ttk.LabelFrame(self,text='경유 자세 [deg] — 시작 → 아래 순서 → 목표',padding=5);route.pack(fill='x',padx=8)
        self.via_tree=ttk.Treeview(route,columns=[f'J{j+1}' for j in range(6)],show='headings',height=3,selectmode='browse')
        for j in range(6):
            self.via_tree.heading(f'J{j+1}',text=f'J{j+1}');self.via_tree.column(f'J{j+1}',width=95,anchor='center')
        self.via_tree.pack(side='left',fill='x',expand=True)
        scroll=ttk.Scrollbar(route,orient='vertical',command=self.via_tree.yview);scroll.pack(side='left',fill='y')
        self.via_tree.configure(yscrollcommand=scroll.set)
        edit=ttk.Frame(route);edit.pack(side='left',padx=8)
        self.via_input=tk.StringVar(value='30 15 -10 20 5 -15')
        ttk.Entry(edit,textvariable=self.via_input,width=38).pack(fill='x')
        row=ttk.Frame(edit);row.pack()
        for label,fn in [('추가',self.add_via),('선택 수정',self.edit_via),('선택 삭제',self.delete_via),('↑',lambda:self.move_via(-1)),('↓',lambda:self.move_via(1))]:
            ttk.Button(row,text=label,command=fn,width=9).pack(side='left')
        self.via_tree.bind('<<TreeviewSelect>>',self.choose_via)
        opts=ttk.Frame(self,padding=(8,0));opts.pack(fill='x')
        for key,label,values in [('mode','실행 모드',['simulation','real']),('method','명령 모드',['jb2','servo']),
                                 ('basis','이론 속도 기준',['max_joint','l2'])]:
            self.vars[key]=tk.StringVar(value=values[0]);ttk.Label(opts,text=label).pack(side='left',padx=5)
            box=ttk.Combobox(opts,textvariable=self.vars[key],values=values,state='readonly',width=12);box.pack(side='left')
        self.ack=tk.BooleanVar();ttk.Checkbutton(opts,text='실기 모션 확인',variable=self.ack).pack(side='left',padx=10)
        joints=ttk.Frame(self,padding=(8,4));joints.pack(fill='x')
        ttk.Label(joints,text='표시 관절').pack(side='left',padx=5)
        self.joint_vars=[tk.BooleanVar(value=True) for _ in range(6)]
        for j,var in enumerate(self.joint_vars):
            ttk.Checkbutton(joints,text=f'J{j+1}',variable=var,command=self.draw).pack(side='left',padx=5)
        ttk.Button(joints,text='전체 선택',command=lambda:self.select_joints(True)).pack(side='left',padx=5)
        ttk.Button(joints,text='전체 해제',command=lambda:self.select_joints(False)).pack(side='left')
        actions=ttk.Frame(self,padding=8);actions.pack(fill='x');self.buttons=[]
        for label,fn in [('현재 자세 읽기',self.current),('궤적·이론 미리보기',self.preview),('모션 실행',self.start),('저장 결과 열기',self.load),('전송·모션 타이밍',self.timing_view)]:
            b=ttk.Button(actions,text=label,command=fn);b.pack(side='left',padx=3);self.buttons.append(b)
        ttk.Button(actions,text='중지 (task_stop)',command=self.stop.set).pack(side='left',padx=10)
        self.status=tk.StringVar(value='연결 없이 미리보기 가능 · 실행 시 speed bar=1.0')
        ttk.Label(actions,textvariable=self.status).pack(side='left',padx=8)
        ttk.Label(self,text='점선: 속도·가속도로 계산한 이상적인 삼각/사다리꼴 프로파일. 제어기 blending 예측값이 아닙니다. 실측은 시작 시각만 맞추고 시간 축을 늘이거나 줄이지 않습니다. ○ = 시작·경유·목표 자세(이상 도달 시각), × = 실측이 그 자세에 가장 가까이 간 지점, 작은 점 = jb2로 보낸 분할점.').pack(anchor='w',padx=12)
        pane=ttk.Panedwindow(self,orient='vertical');pane.pack(fill='both',expand=True)
        chart=ttk.Frame(pane);pane.add(chart,weight=4)
        self.fig=Figure(figsize=(11,6));self.ax=self.fig.subplots(2,1,sharex=True)
        self.canvas=FigureCanvasTkAgg(self.fig,master=chart);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        NavigationToolbar2Tk(self.canvas,chart)
        tab=ttk.Frame(pane);pane.add(tab,weight=1)
        ttk.Label(tab,text='웨이포인트 도달 오차 (6관절 공간 최단 거리 / 관절별 오차 = 측정 − 목표, deg)').pack(anchor='w')
        cols=['result','name','ideal','closest','dist','maxj']+[f'e{j+1}' for j in range(6)]
        self.err_tree=ttk.Treeview(tab,columns=cols,show='headings',height=5)
        for col,text,width in [('result','결과',150),('name','웨이포인트',80),('ideal','이상 도달 [s]',90),('closest','최근접 시각 [s]',100),('dist','L2 거리',80),('maxj','최대 관절',80)]+[(f'e{j+1}',f'ΔJ{j+1}',70) for j in range(6)]:
            self.err_tree.heading(col,text=text);self.err_tree.column(col,width=width,anchor='center')
        self.err_tree.pack(fill='both',expand=True)
        self.log=tk.Text(pane,height=7,wrap='word');pane.add(self.log,weight=1)
        self.after(100,self.poll);self.protocol('WM_DELETE_WINDOW',self.close)
        self.preview()

    def write(self,text):self.log.insert('end',str(text)+'\n');self.log.see('end')
    def emit(self,kind,data):self.events.put((kind,data))
    def via_values(self):
        return [[float(x) for x in self.via_tree.item(item,'values')] for item in self.via_tree.get_children()]

    def parse_via(self):
        values=[float(x) for x in self.via_input.get().replace(',',' ').split()]
        if len(values)!=6 or not np.isfinite(values).all():raise ValueError('경유 자세 J1~J6의 각도 6개를 입력하세요')
        return values

    def choose_via(self,event=None):
        selected=self.via_tree.selection()
        if selected:self.via_input.set(' '.join(self.via_tree.item(selected[0],'values')))

    def add_via(self):
        if self.busy:return
        try:
            values=self.parse_via();selected=self.via_tree.selection()
            index=self.via_tree.index(selected[0])+1 if selected else 'end'
            item=self.via_tree.insert('',index,values=values);self.via_tree.selection_set(item);self.via_tree.see(item)
        except ValueError as e:messagebox.showerror('경유 자세 오류',str(e))

    def edit_via(self):
        if self.busy:return
        try:
            selected=self.via_tree.selection()
            if selected:self.via_tree.item(selected[0],values=self.parse_via())
        except ValueError as e:messagebox.showerror('경유 자세 오류',str(e))

    def delete_via(self):
        if not self.busy:
            for item in self.via_tree.selection():self.via_tree.delete(item)

    def move_via(self,offset):
        if self.busy:return
        selected=self.via_tree.selection()
        if selected:
            item=selected[0];index=self.via_tree.index(item)
            self.via_tree.move(item,'',max(0,min(len(self.via_tree.get_children())-1,index+offset)))

    def get_config(self):
        v={k:x.get() for k,x in self.vars.items()}
        def numbers(s):return [float(x) for x in s.replace(',',' ').split()]
        c=dict(ip=v['ip'].strip(),base=numbers(v['base']),end=numbers(v['end']),
               segments=int(v['segments']),mode=v['mode'],method=v['method'],basis=v['basis'],blend=float(v['blend']),gain=float(v['gain']),alpha=float(v['alpha']))
        for k in ('speed','accel','hz','window','timeout','servo_period_ms','servo_t2'):
            c[k]=float(v[k])
            if not np.isfinite(c[k]) or c[k]<=0:raise ValueError(f'{k}: 유한한 양수 필요')
        if not .02<=c['servo_t2']<=.2:raise ValueError('servo t2는 0.02~0.2초')
        if c['segments']>2000:raise ValueError('분할 수는 2000 이하로 설정하세요')
        if not np.isfinite([c['blend'],c['gain'],c['alpha']]).all() or not 0<=c['blend']<=100 or not 0<=c['gain']<=1 or not 0<=c['alpha']<=1:
            raise ValueError('blend: 0..100, gain/alpha: 0..1')
        c['via']=self.via_values()
        points,times,anchors=path_plan(c)
        c.update(speed_bar=1.)
        intervals=np.diff(times)
        c['via_sample_indices']=anchors[:-1].tolist()
        if c['method']=='servo':
            points,times=servo_plan(c)
        return c,points

    def preview(self):
        if self.busy:return
        try:
            self.config,self.points=self.get_config();self.results=[];self.out=None
            self.write(f"시작: {self.config['base']} → 경유 {len(self.config['via'])}개 → 목표: {self.config['end']} / 전체 분할 {len(self.points)}개")
            self.draw()
        except Exception as e:messagebox.showerror('설정 오류',str(e))

    @staticmethod
    def pose_list(c):
        return np.asarray([c['base'],*c.get('via',[]),c['end']],float)

    @staticmethod
    def pose_names(c):
        return ['start',*[f'via{i+1}' for i in range(len(c.get('via',[])))],'end']

    def fill_error_table(self):
        for item in self.err_tree.get_children():self.err_tree.delete(item)
        c=self.config
        if c is None or not self.results:return
        poses=self.pose_list(c);names=self.pose_names(c);arrival=ideal_arrival_times(c)
        for label,rows in self.results:
            t,q,_,_=analyze(rows,c);pr=waypoint_proximity(t,q,poses)
            for k,name in enumerate(names):
                self.err_tree.insert('','end',values=[label,name,f'{arrival[k]:.2f}',f'{pr["time"][k]:.2f}',f'{pr["dist"][k]:.3f}',f'{np.abs(pr["error"][k]).max():.3f}']
                                     +[f'{x:+.3f}' for x in pr['error'][k]])

    def select_joints(self,selected):
        for var in self.joint_vars:var.set(selected)
        self.draw()

    def joint_tag(self):
        return '_'.join(f'J{j+1}' for j,var in enumerate(self.joint_vars) if var.get()) or 'none'

    def draw(self):
        if self.config is None:return
        c=self.config
        selected=[j for j,var in enumerate(self.joint_vars) if var.get()]
        for ax in self.ax:ax.clear()
        if selected:
            ti,qi,vi=path_profile(c)
            _,planned_times,anchors=path_plan(c)
            for anchor in anchors[:-1]:
                for ax in self.ax:ax.axvline(planned_times[anchor],color='gray',ls='--',lw=.6,alpha=.5)
            measured=[(label,*analyze(rows,c)[:3]) for label,rows in self.results]
            poses=self.pose_list(c);arrival=ideal_arrival_times(c)
            prox=[waypoint_proximity(t,q,poses) for _,t,q,_ in measured]
            for j in selected:
                color=f'C{j}'
                self.ax[0].plot(ti,qi[:,j],':',color=color,lw=1.6,label=f'J{j+1} ideal')
                self.ax[1].plot(ti,vi[:,j],':',color=color,lw=1.6,label=f'J{j+1} ideal')
                if c['method']=='jb2':   # 실제로 보낸 jb2 분할점 (이상 도달 시각)
                    self.ax[0].plot(planned_times[1:],self.points[:,j],'.',color=color,ms=3,alpha=.45,zorder=3)
                self.ax[0].plot(arrival,poses[:,j],'o',mfc='white',mec=color,mew=1.6,ms=9,zorder=5,label=f'J{j+1} waypoint' if j==selected[0] else None)
                for (label,t,q,v),pr in zip(measured,prox):
                    self.ax[0].plot(t,q[:,j],color=color,label=f'J{j+1} {label}')
                    self.ax[1].plot(t,v[:,j],color=color,label=f'J{j+1} {label}')
                    self.ax[0].plot(pr['time'],pr['pose'][:,j],'x',color=color,ms=8,mew=2,zorder=6,label=f'{label} closest' if j==selected[0] else None)
                    if j==selected[0]:   # 첫 표시 관절 × 옆에 오차(6관절 L2 거리)를 적는다
                        for k,name in enumerate(self.pose_names(c)):
                            self.ax[0].annotate(f"{name}\nΔ={pr['dist'][k]:.2f}°",(pr['time'][k],pr['pose'][k,j]),textcoords='offset points',xytext=(7,7),fontsize=8,
                                                bbox=dict(boxstyle='round,pad=.2',fc='white',ec='gray',alpha=.8),zorder=7)
                self.ax[0].axhline(c['end'][j],color=color,ls='--',lw=.6,alpha=.5)
        else:
            for ax in self.ax:ax.text(.5,.5,'Select joints to display',ha='center',va='center',transform=ax.transAxes)
        self.fill_error_table()
        self.ax[1].axhline(0,color='gray',lw=.6)
        self.ax[0].set_ylabel('Joint position [deg]');self.ax[1].set_ylabel('Signed joint speed [deg/s]')
        self.ax[1].set_xlabel('Time since motion onset [s]')
        self.ax[0].set_title(f"{c['method']} / speed={c['speed']:g}, accel={c['accel']:g}, {c['basis']} / {c['mode']}")
        for ax in self.ax:
            ax.grid(alpha=.25)
            if selected:ax.legend(fontsize=8,loc='best',ncol=2)
        self.fig.tight_layout();self.canvas.draw_idle()

    def work(self,fn):
        if self.busy:return
        self.busy=True;self.stop.clear()
        for b in self.buttons:b.configure(state='disabled')
        def job():
            try:fn()
            except Exception as e:self.emit('error',str(e))
            finally:self.emit('done',None)
        threading.Thread(target=job,daemon=True).start()

    def current(self):
        ip=self.vars['ip'].get().strip()
        self.work(lambda:self.emit('current',read_current(ip)))

    def start(self):
        if self.busy:return
        try:
            c,points=self.get_config()
            if c['mode']=='real' and not self.ack.get():raise ValueError('시작/경유/목표 자세 확인 후 실기 모션 확인을 선택하세요')
            self.config=c;self.points=points;self.results=[]
            self.out=Path(__file__).resolve().parent/'out'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');self.out.mkdir(parents=True)
            (self.out/'config.json').write_text(json.dumps(c,indent=2))
            np.savetxt(self.out/'waypoints.csv',points,delimiter=',',header='J1,J2,J3,J4,J5,J6',comments='')
            self.draw();self.write(f'결과 저장: {self.out}')
            out=self.out
            self.work(lambda:execute(c,points,out,self.stop,self.emit))
        except Exception as e:messagebox.showerror('실행 오류',str(e))

    def load(self):
        path=filedialog.askdirectory(title='config.json, manifest.json이 있는 실험 폴더')
        if not path:return
        try:
            out=Path(path);c=json.loads((out/'config.json').read_text());results=[]
            points,_,_=path_plan(c)
            if c['method']=='servo' and 'servo_period_ms' in c:points,_=servo_plan(c)
            for entry in json.loads((out/'manifest.json').read_text()):
                if entry['completed']:
                    rows=np.genfromtxt(out/entry['file'],delimiter=',',skip_header=1,ndmin=2);analyze(rows,c)
                    results.append((entry['label'],rows))
            self.out=out;self.config=c;self.points=points;self.results=results
            for key,var in self.vars.items():
                if key in c:var.set(' '.join(map(str,c[key])) if isinstance(c[key],list) else str(c[key]))
            for item in self.via_tree.get_children():self.via_tree.delete(item)
            for pose in c.get('via',[]):self.via_tree.insert('','end',values=pose)
            self.draw();self.report()
        except Exception as e:messagebox.showerror('불러오기 오류',str(e))

    def save_proximity(self,index,label,t,q):
        """웨이포인트(시작·경유·목표)와 jb2 분할점에 대한 근접도 CSV 저장. 반환: 사용자 웨이포인트 근접도."""
        c=self.config;poses=self.pose_list(c);names=self.pose_names(c);arrival=ideal_arrival_times(c)
        prox=waypoint_proximity(t,q,poses)
        self.write(f"{label}: 웨이포인트 최근접 거리(6관절 L2) "+', '.join(f'{n} {d:.3f}' for n,d in zip(names,prox['dist']))+' deg')
        if not self.out:return prox
        def table(path,rows):
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['label','index','name','ideal_time_s','closest_time_s','time_offset_s','dist_L2_deg','max_joint_error_deg']
                    +[f'target_J{j+1}' for j in range(6)]+[f'closest_J{j+1}' for j in range(6)]+[f'error_J{j+1}' for j in range(6)]);w.writerows(rows)
        def rows(pr,names,targets,ideal):
            return [[label,k,names[k],f'{ideal[k]:.4f}',f'{pr["time"][k]:.4f}',f'{pr["time"][k]-ideal[k]:.4f}',f'{pr["dist"][k]:.5f}',f'{np.abs(pr["error"][k]).max():.5f}']
                    +[f'{x:.5f}' for x in targets[k]]+[f'{x:.5f}' for x in pr['pose'][k]]+[f'{x:.5f}' for x in pr['error'][k]] for k in range(len(names))]
        table(self.out/f'waypoint_proximity_{index}.csv',rows(prox,names,poses,arrival))
        if c['method']=='jb2' and getattr(self,'points',None) is not None and len(self.points)<=1000:
            _,planned_times,_=path_plan(c);pts=np.asarray(self.points)
            pp=waypoint_proximity(t,q,pts)
            table(self.out/f'command_points_proximity_{index}.csv',rows(pp,[f'p{k+1}' for k in range(len(pts))],pts,planned_times[1:]))
            self.write(f"{label}: jb2 분할점 {len(pts)}개 최근접 거리 평균 {pp['dist'].mean():.3f}, 최대 {pp['dist'].max():.3f} deg")
        return prox

    def timing_view(self):
        """명령 전송(TCP)·제어기 응답·실제 모션을 한 시간축에 겹친 그래프를 새 창에 띄우고, 결과 폴더에 PNG·CSV 로 저장한다."""
        if not self.out:
            messagebox.showinfo('전송·모션 타이밍','실행했거나 불러온 결과 폴더가 없습니다.');return
        try:
            for line in timing_plot.describe(timing_plot.save(self.out)):self.write('타이밍: '+line)
            fig=timing_plot.figure(timing_plot.build_data(self.out))
        except Exception as e:messagebox.showerror('타이밍 그래프 오류',str(e));return
        win=tk.Toplevel(self);win.title(f'전송·모션 타이밍 — {self.out.name}');win.geometry('1250x850')
        canvas=FigureCanvasTkAgg(fig,master=win);canvas.get_tk_widget().pack(fill='both',expand=True);NavigationToolbar2Tk(canvas,win);canvas.draw()
        self.write(f'타이밍 저장: {self.out}/timing_compare.png, timing_zoom.png, timing_compare.csv')

    def report(self):
        summary=[]
        for index,(label,rows) in enumerate(self.results):
            t,q,v,_=analyze(rows,self.config)
            end_error=q[-1]-np.asarray(self.config['end'])
            self.write(f"{label}: 관절별 종료 오차 {np.round(end_error,4)} deg")
            prox=self.save_proximity(index,label,t,q)
            row=dict(method=self.config['method'],gain=self.config['gain'],blend=self.config['blend'],
                     linf_peak=float(np.abs(v).max()),l2_peak=float(np.linalg.norm(v,axis=1).max()))
            row.update({f'J{j+1}_peak':float(x) for j,x in enumerate(np.abs(v).max(axis=0))})
            row.update({f'J{j+1}_end_error':float(x) for j,x in enumerate(end_error)})
            row.update(waypoint_max_dist=float(prox['dist'].max()),waypoint_max_joint_error=float(np.abs(prox['error']).max()),
                       via_max_dist=float(prox['dist'][1:-1].max()) if len(prox['dist'])>2 else '');summary.append(row)
            if self.out:
                try:self.write('타이밍: '+' / '.join(timing_plot.describe(timing_plot.save(self.out))))
                except Exception as e:self.write(f'타이밍 그래프 생략: {e}')
                np.savetxt(self.out/f'velocity_{index}.csv',np.c_[t,q,v],delimiter=',',header=','.join(['t']+[f'q{j+1}' for j in range(6)]+[f'v{j+1}' for j in range(6)]),comments='')
        if self.out and summary:
            with (self.out/'summary.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
            self.fig.savefig(self.out/f"comparison_{self.joint_tag()}.png",dpi=140)

    def poll(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='log':self.write(data);self.status.set(data)
                elif kind=='current':
                    q=data['ref'] if data['sim'] else data['ang']
                    self.vars['base'].set(' '.join(f'{v:.4f}' for v in q));self.write(f'현재 자세: {q}, state={data["state"]}')
                elif kind=='result':
                    self.results.append(data);self.draw();self.report()
                elif kind=='error':self.write('오류: '+data);self.status.set('오류 / 기록 확인');messagebox.showerror('실험 중단',data)
                elif kind=='done':
                    self.busy=False
                    for b in self.buttons:b.configure(state='normal')
                    self.write('작업 종료')
        except queue.Empty:pass
        except Exception as e:self.write(f'표시/분석 오류: {e}')
        self.after(100,self.poll)

    def close(self):
        if self.busy:
            self.stop.set();self.write('중지 처리 중입니다. 작업 종료 후 창을 닫아 주세요.');return
        self.destroy()


if __name__=='__main__':App().mainloop()
