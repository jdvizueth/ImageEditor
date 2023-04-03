import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
import tkinter.ttk as ttk
import os
import math
import json
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from rembg import remove

BUTTON_WIDTH = 14
SLIDER_LENGTH = 250

LEFT_IMAGE = 0
RIGHT_IMAGE = 1

GREEN = (0, 255, 0)
RED = (0, 0, 255)
AQUAMARINE = (212, 255, 127)

image_chosen = ''

supportedFiletypes = [('JPEG Image', '*.jpg'), ('PNG Image', '*.png'),
                      ('PPM Image', '*.ppm')]


def error(msg):
    tkMessageBox.showerror("Error", msg)


class CustomJSONEncoder(json.JSONEncoder):
    '''This class supports the serialization of JSON files containing
       OpenCV's feature and match objects as well as Numpy arrays.'''

    def __init__(self):
        super(CustomJSONEncoder, self).__init__(indent=True)

    def default(self, o):
        if hasattr(o, 'pt') and hasattr(o, 'size') and hasattr(o, 'angle') and \
           hasattr(o, 'response') and hasattr(o, 'octave') and \
           hasattr(o, 'class_id'):
            return {'__type__': 'cv2.KeyPoint',
                    'point': o.pt,
                    'size': o.size,
                    'angle': o.angle,
                    'response': o.response,
                    'octave': o.octave,
                    'class_id': o.class_id}

        elif hasattr(o, 'distance') and hasattr(o, 'trainIdx') and \
                hasattr(o, 'queryIdx') and hasattr(o, 'imgIdx'):
            return {'__type__': 'cv2.DMatch',
                    'distance': o.distance,
                    'trainIdx': o.trainIdx,
                    'queryIdx': o.queryIdx,
                    'imgIdx': o.imgIdx}

        elif isinstance(o, np.ndarray):
            return {'__type__': 'numpy.ndarray',
                    '__shape__': o.shape,
                    '__array__': list(o.ravel())}
        else:
            json.JSONEncoder.default(self, o)


def customLoader(d):
    '''This function supports the deserialization of the custom types defined
       above.'''
    if '__type__' in d:
        if d['__type__'] == 'cv2.KeyPoint':
            k = cv2.KeyPoint()
            k.pt = (float(d['point'][0]), float(d['point'][1]))
            k.size = float(d['size'])
            k.angle = float(d['angle'])
            k.response = float(d['response'])
            k.octave = int(d['octave'])
            k.class_id = int(d['class_id'])
            return k
        elif d['__type__'] == 'cv2.DMatch':
            dm = cv2.DMatch()
            dm.distance = float(d['distance'])
            dm.trainIdx = int(d['trainIdx'])
            dm.queryIdx = int(d['queryIdx'])
            dm.imgIdx = int(d['imgIdx'])
            return dm
        elif d['__type__'] == 'numpy.ndarray':
            arr = np.array([float(x) for x in d['__array__']])
            arr.reshape(tuple([int(x) for x in d['__shape__']]))
            return arr
        else:
            return d
    else:
        return d


def load(filepath):
    return json.load(open(filepath, 'r'), object_hook=customLoader)


def dump(filepath, obj):
    with open(filepath, 'w') as f:
        f.write(CustomJSONEncoder().encode(obj))


class ImageWidget(tk.Canvas):
    '''This class represents a Canvas on which OpenCV images can be drawn.
       The canvas handles shrinking of the image if the image is too big,
       as well as writing of the image to files. '''

    def __init__(self, parent):
        self.imageCanvas = tk.Canvas.__init__(self, parent)
        self.originalImage = None
        self.bind("<Configure>", self.redraw)

    def convertCVToTk(self, cvImage):
        height, width, _ = cvImage.shape
        if height == 0 or width == 0:
            return 0, 0, None
        img = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
        return height, width, ImageTk.PhotoImage(img)

    def fitImageToCanvas(self, cvImage):
        height, width, _ = cvImage.shape
        if height == 0 or width == 0:
            return cvImage
        ratio = width / float(height)
        if self.winfo_height() < height:
            height = self.winfo_height()
            width = int(ratio * height)
        if self.winfo_width() < width:
            width = self.winfo_width()
            height = int(width / ratio)
        dest = cv2.resize(cvImage, (width, height),
                          interpolation=cv2.INTER_LANCZOS4)
        return dest

    def drawCVImage(self, cvImage):
        self.originalImage = cvImage
        height, width, img = self.convertCVToTk(self.fitImageToCanvas(cvImage))
        if height == 0 or width == 0:
            return
        self.tkImage = img  # prevent the image from being garbage collected
        self.delete("all")
        x = (self.winfo_width() - width) / 2.0
        y = (self.winfo_height() - height) / 2.0
        self.create_image(x, y, anchor=tk.NW, image=self.tkImage)

    def redraw(self, _):
        if self.originalImage is not None:
            self.drawCVImage(self.originalImage)

    def writeToFile(self, filename):
        if self.originalImage is not None:
            cv2.imwrite(filename, self.originalImage)


class BaseFrame(tk.Frame):
    '''The base frame inherited by all the tabs in the UI.'''

    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.root = root

        # UI elements which appear in all frames
        # We don't specify the position here, because it varies among
        # different frames
        self.status = tk.Label(self, text='Load an image to remove Background')

        self.thresholdLabel = tk.Label(self, text='Threshold (10^x):')
        self.thresholdSlider = tk.Scale(self, from_=-4, to=0, resolution=0.1,
                                        orient=tk.HORIZONTAL, length=SLIDER_LENGTH)
        self.thresholdSlider.set(-2)

        self.imageCanvas = ImageWidget(self)

        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        self.grid_rowconfigure(3, weight=1)

    def setStatus(self, text):
        self.status.configure(text=text)


class BackgroundRemoveFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        self.loadImageButton = tk.Button(self, text='Load Image',
                                         command=self.loadImage, width=BUTTON_WIDTH)
        self.loadImageButton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.clearKeypointsButton = tk.Button(self, text='Clear Keypoints',
                                              command=self.reloadImage, width=BUTTON_WIDTH)
        self.clearKeypointsButton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Screenshot',
                                          command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.computeRemoverButton = tk.Button(
            self, text='Compute', command=self.computeRemove, width=BUTTON_WIDTH)

        self.computeRemoverButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.thresholdLabel.grid(row=1, column=3, sticky=tk.W)
        self.thresholdSlider.grid(row=1, column=4, columnspan=2,
                                  sticky=tk.W+tk.E)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

        self.image = None

        self.keypoints = None

    def loadImage(self):
        filename = tkFileDialog.askopenfilename(parent=self.root,
                                                filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
            self.image = cv2.imread(filename)
            self.imageCanvas.drawCVImage(self.image)
            self.setStatus('Loaded ' + filename)

    def reloadImage(self, image):
        if self.image is not None:
            self.keypoints = None
            self.image = image
            print('has been switched')
            self.imageCanvas.drawCVImage(self.image)

    def screenshot(self):
        if self.image is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                                                      filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved screenshot to ' + filename)
        else:
            error('Load image before taking a screenshot!')

    def computeRemove(self, *args):
        if self.image is not None:
            self.setStatus('Clearing Background')

            output = remove(self.image)

            self.reloadImage(output)

            self.setStatus('Cleared Background')


class BackgroundUIFrame(tk.Frame):
    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.root = root
        self.notebook = ttk.Notebook(self.parent)
        # Add Sections and Buttons here that specifies what to do
        self.backgroundRemoverFrame = BackgroundRemoveFrame(
            self.notebook, root)

        self.notebook.add(self.backgroundRemoverFrame, text='RemoveBackground')

        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)

    def CloseWindow(self):
        self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = BackgroundUIFrame(root, root)
    root.title('Image Editor')
    # Put the window on top of the other windows
    # root.attributes('-fullscreen', True)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry("%dx%d+0+0" % (w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    # root.wm_attributes('-topmost', 1)
    root.mainloop()
