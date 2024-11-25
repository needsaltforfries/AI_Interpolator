#import libraries
import cv2
import tkinter as tk
from PIL import Image, ImageTk

#create window
root = tk.Tk()
root.geometry("900x550")
root.title("FrameInterpolator")

canv1 = tk.Canvas(root, width=540,height=420)
canv1.pack()

#Declare globals/useful variables/load video
#make video capture object
cap = cv2.VideoCapture('marthclip.mp4')
if not cap.isOpened():
    print("Error opening video file")

videoPaused = False
#check length of video in frames
curFrame = 0
max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

startFrame: int = 0
endFrame: int = max_frames

#Declare GUI labels
lbl_start = tk.Label(canv1, text=startFrame)
lbl_end = tk.Label(canv1, text=endFrame)

btn_slctStart = tk.Button(canv1, text="Select Start Frame")
btn_slctEnd = tk.Button(canv1, text="Select End Frame")

#set slider to value between 0 and max frames
sld = tk.Scale(canv1, variable=curFrame, from_=0, to=max_frames, orient="horizontal")

img = None
frLabel = tk.Label(root)
frLabel = tk.Label(root, image=img)

def updateFrame(val: int):
    global curFrame
    curFrame = val

#display image
def readVideo(frame: int):
    global img
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ret, fr = cap.read()
        if not ret:
            return
        fr = cv2.resize(fr, (640, 360))
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = Image.fromarray(fr)
        img = ImageTk.PhotoImage(fr)
        frLabel.config(image=img)
        frLabel.image = img
        root.update()

#pause unpause functions
def pauseVideo():
    global videoPaused
    videoPaused = True

def unpauseVideo():
    global videoPaused
    videoPaused = False

#plays the video until end or paused
def playVideo():
    global img
    global curFrame
    if cap.isOpened():
        framesPassed = 0
        while int(curFrame) < int(max_frames):
            if cv2.waitKey(1) == ord('p') or videoPaused == True:
                break
            ret, fr = cap.read()
            framesPassed += 1
            if not ret:
                return
            fr = cv2.resize(fr, (640, 360))
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = Image.fromarray(fr)
            img = ImageTk.PhotoImage(fr)
            frLabel.config(image=img)
            frLabel.image = img
            root.update()
        sld.set(framesPassed + int(curFrame))

#update frame labels
def selectStartFrame(val: int):
    global startFrame
    startFrame = val
    lbl_start.config(text=f"Start: {startFrame}")
    root.update()
    
def selectEndFrame(val: int):
    global endFrame
    endFrame = val
    lbl_end.config(text=f"End: {endFrame}")
    root.update()

#display window
sld.configure(command=lambda val:[updateFrame(val), readVideo(val)])
sld.grid(row=0, column= 0)

lbl_start.grid(row=0, column=1)

lbl_end.grid(row=0, column=3)

btn_slctStart.config(command= lambda: selectStartFrame(sld.get()))
btn_slctStart.grid(row=0, column=2)

btn_slctEnd.config(command= lambda: selectEndFrame(sld.get()))
btn_slctEnd.grid(row=0, column=4)

btn_openVid = tk.Button(canv1, text="Open Vid", command=lambda:[updateFrame(0), readVideo(0), sld.set(0)])
btn_openVid.grid(row=1, column=0)

icn_play = tk.PhotoImage(file='playbtn.png')
btn_play = tk.Button(canv1, image=icn_play, command=lambda:[unpauseVideo(), playVideo()])
btn_play.grid(row=1, column=1)

icn_pause = tk.PhotoImage(file='pausebtn.png')
btn_pause = tk.Button(canv1, image=icn_pause, command=pauseVideo)
btn_pause.grid(row=1, column=2)

#place image in window
frLabel.pack()

root.mainloop()
cap.release()
cv2.destroyAllWindows()