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
        self.status = tk.Label(self, text='Load an image to remove background')

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


class CropImage(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        # Global variables to store the coordinates
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1
        self.cropping = False
        self.image = None
        self.image2 = None

        self.loadImageButton = tk.Button(self, text='Load Image',
                                         command=self.loadImage, width=BUTTON_WIDTH)
        self.loadImageButton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Save Image',
                                          command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.cropButton = tk.Button(
            self, text='Crop Image', command=self.croppingImage, width=BUTTON_WIDTH)
        self.cropButton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

    def crop_image(self, image, start_x, start_y, end_x, end_y):
        cropped_image = image[start_y:end_y, start_x:end_x]
        return cropped_image

    def mouse_callback(self, event, x, y, flags, param):
        # Access the global variables

        if event == cv2.EVENT_LBUTTONDOWN:
            # Initialize the starting coordinates
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
            self.image = np.copy(self.image2)
            cv2.imshow('Press q to quit, c to crop', self.image)
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            # Update the ending coordinates and indicate that cropping is finished
            self.end_x, self.end_y = x, y
            self.cropping = False

            # Ensure the coordinates are valid (start < end)
            self.start_x, self.start_y = min(
                self.start_x, self.end_x), min(self.start_y, self.end_y)
            self.end_x, self.end_y = max(
                self.start_x, self.end_x), max(self.start_y, self.end_y)

            self.image = np.copy(self.image2)

            cv2.rectangle(self.image, (self.start_x, self.start_y),
                          (self.end_x, self.end_y), (0, 255, 0), 2)

            cv2.imshow('Press q to quit, c to crop', self.image)
            # # Crop the image
            # cropped = self.crop_image(
            #     self.image, self.start_x, self.start_y, self.end_x, self.end_y)

            # # Show the cropped image
            # cv2.imshow('Cropped Image', cropped)
            # cv2.waitKey(0)

    def croppingImage(self):
        if self.image is not None:
            self.image2 = np.copy(self.image)
            cv2.namedWindow('Press q to quit, c to crop')
            cv2.setMouseCallback(
                'Press q to quit, c to crop', self.mouse_callback)

            while True:
                cv2.imshow('Press q to quit, c to crop', self.image)

                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    self.image = np.copy(self.image2)
                    break

                if key == ord('c') and not self.cropping:
                    cropped_image = self.crop_image(
                        self.image2, self.start_x, self.start_y, self.end_x, self.end_y)
                    # self.image = np.copy(self.image2)
                    # cv2.imshow('Cropped Image', cropped_image)
                    self.image = np.copy(cropped_image)
                    self.imageCanvas.drawCVImage(cropped_image)
                    # cv2.waitKey(0)
                    break

            cv2.destroyAllWindows()
        else:
            error('Load image before cropping!')

    def loadImage(self):
        filename = tkFileDialog.askopenfilename(
            parent=self.root, filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
            self.image = cv2.imread(filename)
            self.imageCanvas.drawCVImage(self.image)

            self.setStatus('Loaded ' + filename)

    def reloadImage(self, image):
        if self.image is not None:
            self.keypoints = None
            self.image = image
            self.imageCanvas.drawCVImage(self.image)

    def screenshot(self):
        if self.image is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                                                      filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved image to ' + filename)
        else:
            error('Load image before taking a screenshot!')


class BackgroundRemoveFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        self.loadImageButton = tk.Button(self, text='Load Image',
                                         command=self.loadImage, width=BUTTON_WIDTH)
        self.loadImageButton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Save Image',
                                          command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.computeRemoverButton = tk.Button(
            self, text='Remove Background', command=self.computeRemove, width=BUTTON_WIDTH)

        self.computeRemoverButton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

        self.image = None

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
            self.imageCanvas.drawCVImage(self.image)

    def screenshot(self):
        if self.image is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                                                      filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved image to ' + filename)
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

        self.notebook.add(self.backgroundRemoverFrame,
                          text='Background Removal Tab')

        self.cropFrame = CropImage(self.notebook, root)

        self.notebook.add(self.cropFrame, text='Cropping Frame')

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
