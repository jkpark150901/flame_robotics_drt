"""Tkinter GUI for sending the generated motion TXT recipes."""
import csv
from datetime import datetime
import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from sender import Sender, parse_recipe


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Rainbow · Motion TXT Sender')
        self.geometry('1180x820')
        self.events = queue.Queue()
        self.worker = None
        self.sender = None
        self.path = None
        self.protocol('WM_DELETE_WINDOW', self.close)
        bar = ttk.Frame(self,padding=10)
        bar.pack(fill='x')
        self.host = tk.StringVar(value='192.168.0.10')
        self.port = tk.StringVar(value='5000')
        self.repeats = tk.StringVar(value='1')
        self.timeout = tk.StringVar(value='120')
        self.lag = tk.StringVar(value='10')
        self.ending = tk.StringVar(value='LF')
        for label,var,width in [('IP / Host',self.host,20),('Port',self.port,6),
                                ('반복 (0=무한)',self.repeats,5),('응답 제한 [s]',self.timeout,6),
                                ('허용 지연 [ms]',self.lag,5)]:
            ttk.Label(bar,text=label).pack(side='left',padx=3)
            ttk.Entry(bar,textvariable=var,width=width).pack(side='left')
        ttk.Combobox(bar,textvariable=self.ending,values=['LF','CRLF'],state='readonly',width=6).pack(side='left',padx=5)
        files = ttk.Frame(self,padding=(10,0))
        files.pack(fill='x')
        ttk.Button(files,text='TXT 열기',command=self.open_file).pack(side='left')
        self.run_button = ttk.Button(files,text='연결 후 전송',command=self.start)
        self.run_button.pack(side='left',padx=5)
        self.stop_button = ttk.Button(files,text='후속 전송 중단',command=self.stop,state='disabled')
        self.stop_button.pack(side='left')
        ttk.Button(files,text='전송 시간 CSV 저장',command=self.save_timing).pack(side='right')
        self.file_label = tk.StringVar(value='TXT를 선택하세요. 실행 직전에 편집 내용을 다시 검사합니다.')
        ttk.Label(self,textvariable=self.file_label,padding=(10,8)).pack(anchor='w')
        self.info = tk.StringVar()
        ttk.Label(self,textvariable=self.info,padding=(10,0)).pack(anchor='w')
        panes=ttk.Panedwindow(self,orient='vertical')
        panes.pack(fill='both',expand=True,padx=10,pady=8)
        source=ttk.LabelFrame(panes,text='전문 미리보기 / 편집')
        logbox=ttk.LabelFrame(panes,text='전송 / 응답 로그')
        panes.add(source,weight=3);panes.add(logbox,weight=2)
        self.editor=scrolledtext.ScrolledText(source,font='TkFixedFont',wrap='none',height=20)
        self.editor.pack(fill='both',expand=True)
        self.log=scrolledtext.ScrolledText(logbox,font='TkFixedFont',wrap='word',height=12,state='disabled')
        self.log.pack(fill='both',expand=True)
        self.status=tk.StringVar(value='대기')
        ttk.Label(self,textvariable=self.status,padding=(10,4)).pack(anchor='w')
        ttk.Label(self,text='Servo 주기는 명령의 t1에서 자동 적용합니다. 주석은 전송하지 않고 # WAIT n s는 로컬 대기로 실행합니다.\n'
                  '전송 버튼을 누르면 지정 서버로 모션 명령을 보냅니다. 후속 전송 중단은 이미 실행 중인 로봇 모션을 정지하지 않습니다.',
                  padding=(10,6)).pack(anchor='w')
        self.after(50,self.poll)

    def open_file(self):
        if self.worker and self.worker.is_alive():
            return
        path=filedialog.askopenfilename(initialdir=str(Path(__file__).resolve().parents[1]/'joint_only_motion/motion_txt'),
                                        filetypes=[('Motion TXT','*.txt'),('All','*')])
        if not path:
            return
        try:
            text=Path(path).read_text(encoding='utf-8-sig')
            recipe=parse_recipe(text)
        except (ValueError,OSError) as exc:
            messagebox.showerror('파일 오류',str(exc));return
        self.path=Path(path)
        self.editor.delete('1.0','end');self.editor.insert('1.0',text)
        self.file_label.set(str(self.path))
        self.describe(recipe)

    def describe(self,recipe):
        periods=', '.join(f'{p*1000:g}ms' for p in recipe.periods) or '없음 (JB2 / Move J)'
        commands=sum(bool(s.command) for s in recipe.steps)
        waits=sum(s.delay for s in recipe.steps)
        self.info.set(f'명령 {commands}개 | Servo 주기: {periods} | 별도 대기 {waits:g}s | 반복 시 setup 제외')

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            recipe=parse_recipe(self.editor.get('1.0','end-1c'))
            host=self.host.get().strip();port=int(self.port.get());repeats=int(self.repeats.get())
            timeout=float(self.timeout.get());lag=float(self.lag.get())
            if not host or not 1<=port<=65535 or repeats<0:
                raise ValueError('주소·포트·반복 횟수를 확인하세요.')
            if not all(math.isfinite(v) and v>0 for v in [timeout,lag]):
                raise ValueError('응답 제한과 허용 지연은 유한한 양수여야 합니다.')
        except ValueError as exc:
            messagebox.showerror('입력 오류',str(exc));return
        self.describe(recipe)
        self.sender=Sender(emit=lambda k,d:self.events.put((k,d)),timeout=timeout,
                           ending=self.ending.get(),max_lag_ms=lag)
        sender=self.sender
        def execute():
            try:
                sender.run(host,port,recipe,repeats)
            except Exception as exc:
                self.events.put(('ERROR',str(exc)))
            finally:
                self.events.put(('DONE','연결 종료'))
        self.run_button.configure(state='disabled');self.stop_button.configure(state='normal')
        self.editor.configure(state='disabled')
        self.status.set('연결 중…')
        self.worker=threading.Thread(target=execute,daemon=True)
        self.worker.start()

    def stop(self):
        if self.sender:
            self.sender.cancel.set()
            self.status.set('후속 전송 중단 요청 중…')

    def poll(self):
        for _ in range(250):
            try:
                kind,data=self.events.get_nowait()
            except queue.Empty:
                break
            if kind in ('STATUS','ERROR'):
                self.status.set(data)
            elif kind=='DONE':
                self.run_button.configure(state='normal');self.stop_button.configure(state='disabled')
                self.editor.configure(state='normal')
            self.log.configure(state='normal')
            self.log.insert('end',f'{datetime.now().astimezone().isoformat(timespec="milliseconds")} {kind} {data.rstrip()}\n')
            if int(self.log.index('end-1c').split('.')[0])>6000:
                self.log.delete('1.0','1001.0')
            self.log.see('end');self.log.configure(state='disabled')
        self.after(50,self.poll)

    def save_timing(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('실행 중','전송이 끝난 뒤 저장하세요.');return
        if not self.sender or not self.sender.records:
            return
        path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='transmission_timing.csv')
        if path:
            try:
                with open(path,'w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(self.sender.records[0]))
                    writer.writeheader();writer.writerows(self.sender.records)
            except OSError as exc:
                messagebox.showerror('저장 오류',str(exc))

    def close(self):
        self.stop()
        self.destroy()


if __name__=='__main__':
    App().mainloop()
