import socket
import threading
import time
import unittest
from pathlib import Path

from sender import ACK, Sender, parse_recipe


MOVE='move_j(jnt[0,0,0,0,0,0], 10, 20, 1)'
SERVO='move_servo_j(jnt[0,0,0,0,0,0], 0.03, 0.04, 0.1, 0.5)'


class SenderTests(unittest.TestCase):
    def run_local(self,text,responder,repeats=1,timeout=.5,cancel=None):
        received=[]
        failures=[]
        server=socket.socket()
        server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(2)
        def serve():
            try:
                with server.accept()[0] as peer:
                    peer.settimeout(2)
                    pending=b''
                    while True:
                        chunk=peer.recv(4096)
                        if not chunk:
                            break
                        pending+=chunk
                        while b'\n' in pending:
                            line,pending=pending.split(b'\n',1)
                            received.append((time.perf_counter(),line.decode()))
                            responder(peer,line,len(received))
            except (BrokenPipeError,ConnectionResetError):
                pass
            except Exception as exc:
                failures.append(exc)
        worker=threading.Thread(target=serve,daemon=True);worker.start()
        sender=Sender(timeout=timeout,max_lag_ms=15,cancel=cancel)
        error=None
        try:
            sender.run('127.0.0.1',server.getsockname()[1],parse_recipe(text),repeats)
        except Exception as exc:
            error=exc
        finally:
            worker.join(3);server.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures,[])
        return received,sender,error

    def test_servo_waits_for_setup_then_streams_without_ack_per_tick(self):
        # Delayed servo ACKs arrive together at the end of the stream. Stale
        # servo ACKs must not satisfy the final completion-wait command.
        buffered=[]
        def respond(peer,line,count):
            if count==1:
                time.sleep(.06)
            if line.startswith(b'move_servo_j'):
                buffered.append(ACK)
                if len(buffered)==5:
                    peer.sendall(b'\n'.join(buffered)+b'\n')
                return
            if line==b'move_finish_wait()' and count>2:
                time.sleep(.03)
            peer.sendall(ACK[:9]);peer.sendall(ACK[9:]+b'\n')
        text=MOVE+'\nmove_finish_wait()\n# LOOP BODY STARTS BELOW.\n'+(SERVO+'\n')*5+'move_finish_wait()'
        rows,sender,error=self.run_local(text,respond)
        self.assertIsNone(error)
        self.assertGreaterEqual(rows[2][0]-rows[0][0],.055)
        ticks=[t for t,line in rows if line.startswith('move_servo_j')]
        self.assertAlmostEqual(ticks[-1]-ticks[0],.12,delta=.025)
        self.assertGreater(rows[-1][0]-ticks[-1],.02)
        self.assertEqual(len(sender.pending),0)

    def test_jb2_comments_wait_and_setup_once(self):
        text=MOVE+'\nmove_finish_wait()\n# LOOP FOREVER FROM HERE.\nmove_jb2_run(1)\nmove_finish_wait()\n# WAIT 0.05 s after completion.\nmove_jb2_clear()'
        rows,_,error=self.run_local(text,lambda p,l,n:p.sendall(ACK+b'\n'),repeats=2)
        self.assertIsNone(error)
        self.assertEqual(sum(line.startswith('move_j(') for _,line in rows),1)
        self.assertEqual(sum(line=='move_jb2_run(1)' for _,line in rows),2)
        for i,(_,line) in enumerate(rows):
            if line=='move_jb2_clear()':
                self.assertGreaterEqual(rows[i][0]-rows[i-1][0],.045)

    def test_timeout_denial_disconnect_cancel(self):
        text=MOVE+'\n'+SERVO
        for respond,expected in [(lambda p,l,n:None,TimeoutError),
                                 (lambda p,l,n:p.sendall(b'The command is not allowed'),ValueError),
                                 (lambda p,l,n:p.shutdown(socket.SHUT_RDWR),EOFError)]:
            rows,_,error=self.run_local(text,respond,timeout=.04)
            self.assertIsInstance(error,expected)
            self.assertEqual(len(rows),1)
        cancel=threading.Event()
        def stop(peer,line,count):
            cancel.set()
            peer.sendall(ACK+b'\n')
        rows,_,error=self.run_local(text,stop,cancel=cancel)
        self.assertIsInstance(error,InterruptedError)
        self.assertEqual(len(rows),1)

    def test_excess_lag_aborts_instead_of_burst(self):
        class LateSender(Sender):
            def wait_until(self,deadline):
                time.sleep(.025)
                super().wait_until(deadline)
        a,b=socket.socketpair()
        try:
            sender=LateSender(max_lag_ms=10)
            sender.sock=a
            with self.assertRaises(TimeoutError):
                sender.execute(parse_recipe(SERVO+'\n'+SERVO),1)
            self.assertEqual(sender.records,[])
        finally:
            a.close();b.close()

    def test_all_twelve_actual_txt_files_and_invalid_input(self):
        root=Path(__file__).resolve().parents[1]/'joint_only_motion/motion_txt'
        files=list(root.rglob('*.txt'))
        self.assertEqual(len(files),12)
        for p in files:
            recipe=parse_recipe(p.read_text())
            self.assertGreater(recipe.loop_start,0)
            self.assertTrue(recipe.steps[0].command.startswith('move_j('))
            if 'servo' in p.name:
                self.assertEqual(recipe.periods,[int(p.stem.split('_')[-1][:-2])/1000])
        for text in ['# empty','bad_command()',SERVO.replace('0.03','0.0001'),'wait (1e999 sec)']:
            with self.assertRaises(ValueError):
                parse_recipe(text)


if __name__=='__main__':
    unittest.main()
