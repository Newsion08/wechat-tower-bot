"""Windows capture and input. No input is generated at import time."""
import ctypes as C
from ctypes import wintypes as W
import threading
import time


class ExclusiveWindow:
    """Prevent two player processes from issuing competing clicks."""
    def __init__(self,hwnd):
        self.k=C.WinDLL('kernel32',use_last_error=True)
        self.k.CreateMutexW.argtypes=[C.c_void_p,W.BOOL,W.LPCWSTR]
        self.k.CreateMutexW.restype=W.HANDLE
        self.k.CloseHandle.argtypes=[W.HANDLE]
        self.k.ReleaseMutex.argtypes=[W.HANDLE]
        self.handle=self.k.CreateMutexW(None,True,f'Local\\TowerBot_{hwnd}')
        if not self.handle:raise C.WinError(C.get_last_error())
        if C.get_last_error()==183:
            self.k.CloseHandle(self.handle)
            raise RuntimeError('Another Tower Bot is already controlling this window')

    def __enter__(self):return self

    def __exit__(self,*args):
        self.k.ReleaseMutex(self.handle);self.k.CloseHandle(self.handle)


class Hotkeys:
    """Queue key presses so a brief tap during vision processing is not lost."""
    def __init__(self):
        self.u=api();self.registered=[]
        self.ids={0x5B1:'F8',0x5B2:'ESC'}
        self.u.RegisterHotKey.argtypes=[W.HWND,C.c_int,W.UINT,W.UINT]
        self.u.UnregisterHotKey.argtypes=[W.HWND,C.c_int]
        self.u.PeekMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT,W.UINT]
        try:
            for key,vk in ((0x5B1,0x77),(0x5B2,0x1B)):
                if not self.u.RegisterHotKey(None,key,0x4000,vk):
                    raise RuntimeError('F8/Esc hotkey registration failed; close other hotkey utilities')
                self.registered.append(key)
        except Exception:
            self.close();raise

    def poll(self):
        result=[];msg=W.MSG()
        while self.u.PeekMessageW(C.byref(msg),None,0x312,0x312,1):
            if msg.wParam in self.ids:result.append(self.ids[msg.wParam])
        return result

    def close(self):
        for key in self.registered:self.u.UnregisterHotKey(None,key)
        self.registered=[]


def api():
    u=C.WinDLL('user32',use_last_error=True)
    u.GetForegroundWindow.restype=W.HWND
    u.GetAncestor.argtypes=[W.HWND,W.UINT];u.GetAncestor.restype=W.HWND
    u.GetWindowRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
    u.GetWindowTextLengthW.argtypes=[W.HWND]
    u.GetWindowTextW.argtypes=[W.HWND,W.LPWSTR,C.c_int]
    u.IsWindow.argtypes=[W.HWND];u.IsIconic.argtypes=[W.HWND]
    u.WindowFromPoint.argtypes=[W.POINT];u.WindowFromPoint.restype=W.HWND
    u.SetProcessDpiAwarenessContext.argtypes=[W.HANDLE]
    u.SetProcessDpiAwarenessContext(C.c_void_p(-4))
    return u


class Window:
    def __init__(self,hwnd):
        self.u=api();self.hwnd=int(hwnd)
        if not self.u.IsWindow(self.hwnd): raise RuntimeError('Game window is closed')
        self.original=self.rect()
        self.title=self.text()

    def rect(self):
        r=W.RECT()
        if not self.u.GetWindowRect(self.hwnd,C.byref(r)):
            raise C.WinError(C.get_last_error())
        return (r.left,r.top,r.right,r.bottom)

    def text(self):
        s=C.create_unicode_buffer(self.u.GetWindowTextLengthW(self.hwnd)+1)
        self.u.GetWindowTextW(self.hwnd,s,len(s))
        return s.value

    def check(self,point):
        if not self.u.IsWindow(self.hwnd): return 'window_closed'
        if self.u.IsIconic(self.hwnd): return 'window_minimized'
        if self.rect()!=self.original: return 'window_moved: rerun setup'
        if self.text()!=self.title: return 'window_changed'
        foreground=self.u.GetAncestor(self.u.GetForegroundWindow(),2)
        if foreground!=self.hwnd: return 'window_not_focused'
        under=self.u.GetAncestor(self.u.WindowFromPoint(W.POINT(*point)),2)
        if under!=self.hwnd: return 'click_point_obscured'
        return None

    def click(self,point):
        reason=self.check(point)
        if reason: raise RuntimeError(reason)
        if self.u.GetAsyncKeyState(0x1B)&0x8000: raise RuntimeError('Escape pressed')
        class MOUSEINPUT(C.Structure):
            _fields_=[('dx',W.LONG),('dy',W.LONG),('mouseData',W.DWORD),
                      ('dwFlags',W.DWORD),('time',W.DWORD),('dwExtraInfo',C.c_size_t)]
        class KEYBDINPUT(C.Structure):
            _fields_=[('wVk',W.WORD),('wScan',W.WORD),('dwFlags',W.DWORD),
                      ('time',W.DWORD),('dwExtraInfo',C.c_size_t)]
        class HARDWAREINPUT(C.Structure):
            _fields_=[('uMsg',W.DWORD),('wParamL',W.WORD),('wParamH',W.WORD)]
        class UNION(C.Union):
            _fields_=[('mi',MOUSEINPUT),('ki',KEYBDINPUT),('hi',HARDWAREINPUT)]
        class INPUT(C.Structure):
            _anonymous_=('data',)
            _fields_=[('type',W.DWORD),('data',UNION)]
        self.u.SendInput.argtypes=[W.UINT,C.POINTER(INPUT),C.c_int]
        self.u.SendInput.restype=W.UINT
        left,top,width,height=[self.u.GetSystemMetrics(i) for i in (76,77,78,79)]
        x=round((point[0]-left)*65535/(width-1))
        y=round((point[1]-top)*65535/(height-1))
        items=(INPUT*3)()
        items[0].mi=MOUSEINPUT(x,y,0,0x8000|0x4000|0x0001,0,0)
        items[1].mi=MOUSEINPUT(0,0,0,0x0002,0,0)
        items[2].mi=MOUSEINPUT(0,0,0,0x0004,0,0)
        if self.u.SendInput(3,items,C.sizeof(INPUT))!=3:
            release=INPUT();release.mi=MOUSEINPUT(0,0,0,0x0004,0,0)
            self.u.SendInput(1,C.byref(release),C.sizeof(INPUT))
            raise RuntimeError('SendInput failed; check matching privilege levels')

    def park(self,point):
        reason=self.check(point)
        if reason:raise RuntimeError(reason)
        if not self.u.SetCursorPos(*point):raise C.WinError(C.get_last_error())


def find_window(title):
    u=api();found=[]
    callback=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
    def visit(hwnd,lparam):
        if u.IsWindowVisible(hwnd):
            buf=C.create_unicode_buffer(512);u.GetWindowTextW(hwnd,buf,512)
            if buf.value==title: found.append(int(hwnd))
        return True
    u.EnumWindows(callback(visit),0)
    if len(found)!=1:
        raise RuntimeError(f'Expected one window named {title!r}, found {len(found)}. Open the game first.')
    return Window(found[0])


class Capture:
    def __init__(self,region,output=0,fps=120):
        import dxcam
        self.camera=dxcam.create(output_idx=output,output_color='BGR',max_buffer_len=4)
        self.camera.start(region=tuple(region),target_fps=fps,video_mode=False)
        self.latest=None;self.error=None;self.closed=False
        self.thread=threading.Thread(target=self.read,daemon=True)
        self.thread.start()

    def read(self):
        try:
            while not self.closed:
                value=self.camera.get_latest_frame(with_timestamp=True)
                if value is None: break
                frame,t=value
                # DXGI LastPresentTime and perf_counter share the QPC clock.
                if abs(time.perf_counter()-t)>5:
                    raise RuntimeError('Capture timestamp and perf_counter clocks differ')
                self.latest=(frame,t)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'

    def close(self):
        self.closed=True
        self.camera.stop()
        self.thread.join(timeout=1)
        self.camera.release()


class Scheduler:
    """One cancellable deadline. A fired command cannot be replaced/repeated."""
    def __init__(self,send,guard,clock=time.perf_counter):
        self.send=send;self.guard=guard;self.clock=clock
        self.lock=threading.Lock();self.pending=None;self.result=None;self.closed=False
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def replace(self,proposal):
        with self.lock:
            if self.result is None: self.pending=proposal

    def cancel(self):
        with self.lock: self.pending=None

    def take(self):
        with self.lock:
            result=self.result;self.result=None
            return result

    def run(self):
        while not self.closed:
            with self.lock:
                p=self.pending
                if p and self.clock()>=p['deadline']:
                    self.pending=None
                    now=self.clock()
                    reason=self.guard()
                    if now-p['deadline']>.008: reason='missed_deadline'
                    if now-p['observed']>.12: reason='stale_frame'
                    if reason:
                        self.result={'skipped':reason,'proposal':p}
                    else:
                        try:
                            self.send()
                            self.result={'fired':now,'proposal':p}
                        except Exception as exc:
                            self.result={'error':str(exc),'proposal':p}
            time.sleep(.001)

    def close(self):
        self.cancel();self.closed=True;self.thread.join(timeout=1)
