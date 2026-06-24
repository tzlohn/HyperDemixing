import numpy as np
import tifffile as TFF
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
#from PyQt5.QtCore import *
from PyQt5.Qt import Qt
import os,sys,time,csv
#import matplotlib.pyplot as p
import pyqtgraph as pg
import multiprocessing as mp
from multiprocessing import get_context

#from SpotAnalyzer import MainWin

class MainWin(QWidget):
    def __init__(self,filename, isAuto = False):
        super().__init__()
        """
        Count,done = QInputDialog.getInt(self,"Input dialoge","input total number of channels to exclude:",3,0,32,1)
        ExChan = list()
        ExcludeList = [6,12,21]
        if Count !=0 :
            for idx in range(Count):
                Num,done = QInputDialog.getInt(self,"Input dialoge","select the channel to exclude:",ExcludeList[idx],1,32,1)
                ExChan.append(Num-1)
        
        self.setWindowTitle("Main Image Window")
        """
        img = TFF.imread(filename)
        self.img = img
        """
        if len(ExChan) != 0:
            self.img = self.removeExcludeChan(img,ExChan)
        else:
            self.img = img
        """
        self.filename = filename

        self.ImageWindow = pg.ImageView()
        self.cursor = QCursor(Qt.CrossCursor)
        self.ImageWindow.setCursor(self.cursor)
        self.PenColor = [Qt.yellow,Qt.red,Qt.blue,Qt.green,Qt.magenta,QColor(153,76,0), Qt.black]
        self.FalseColor = {"-1":Qt.black,"0":Qt.white,"2":Qt.cyan}

        self.FilePath = QLineEdit()
        self.FilePath.setText(filename)

        self.CursorInfo = QLineEdit()
        Font_Cursor = QFont("Helvetica",pointSize = 9)
        Font_Cursor.setBold(True)
        self.CursorInfo.setFont(Font_Cursor)

        self.ImageWindow.setImage(self.img)

        self.Layout = QGridLayout()
        self.Layout.addWidget(self.ImageWindow,1,0,10,10)
        self.Layout.addWidget(self.FilePath,11,0,1,6)
        self.Layout.addWidget(self.CursorInfo,11,6,1,2)

        self.setLayout(self.Layout)

        self.cursorEvent = pg.SignalProxy(self.ImageWindow.scene.sigMouseMoved, delay = 0, slot = self.showValue)

        self.resize(1000,1000)

    def removeExcludeChan(self,img,ExChan):
        NewImage = list()
        for idx in range(img.shape[0]):
            if not idx in ExChan:
                NewImage.append(img[idx,:,:])
        
        return np.asarray(NewImage)

    def showValue(self,CursorInPlot):
        x,y,z = self.getXY(CursorInPlot)
        
        try:
            if len(self.ImageWindow.image.shape) == 3:
                self.CursorInfo.setText("(x,y,z) = (%d,%d,%d,%d)"%(x,y,z+1,self.img[z,x,y]))
            elif len(self.ImageWindow.image.shape) == 2:
                self.CursorInfo.setText("(x,y) = (%d,%d,%d)"%(x,y,self.ImageWindow.image[x,y]))
        except:
            pass
    
    def getXY(self,CursorInPlot):
        cursor = self.ImageWindow.view.mapSceneToView(CursorInPlot[0])
        x = int(cursor.x())
        y = int(cursor.y())
        z = self.ImageWindow.currentIndex
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        return x,y,z     
    
    def createCSV(self,filename,data):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in zip(*data):
                csvwriter.writerow(row)
        print("%s is saved in %s"%(filename,os.getcwd()))

class PCAImageWin(MainWin):
    def __init__(self,FileName):
        super().__init__(FileName)

        self.PCAButton = QPushButton(self)
        self.PCAButton.setText("Thresholding with PCA")
        self.PCAButton.clicked.connect(self.doPCA)
        self.UnmixingButton = QPushButton(self)
        self.UnmixingButton.setText("Open demixing wizard")
        self.UnmixingButton.clicked.connect(self.openUnmixWizard)
        self.Layout.addWidget(self.PCAButton,0,0,1,2)
        self.Layout.addWidget(self.UnmixingButton,0,2,1,2)
        self.setLayout(self.Layout)

    def doPCA(self):
        [v1,v2,self.ImgData] = self.ImgPCA(self.img)    
        self.Plot = Plot2D(self,v1,v2,self.ImgData)
        self.Plot.show()
        self.clickEvent = pg.SignalProxy(self.ImageWindow.scene.sigMouseClicked, delay = 0, slot = self.sendPixelPos)

    def ImgPCA(self,Img:np.ndarray):
        self.OriShape = Img.shape
        ImgData = Img.reshape(self.OriShape[0],self.OriShape[1]*self.OriShape[2])
        NormImgData = ImgData/np.sum(ImgData,axis = 0)
        CovMx = np.dot(NormImgData,np.transpose(NormImgData))
        [Eig , EigV] = np.linalg.eig(CovMx)
        #print(Eig[0],EigV[0])
        #print(Eig[1],EigV[1])
        V1 = np.dot(np.transpose(NormImgData),EigV[0])
        V2 = np.dot(np.transpose(NormImgData),EigV[1])
        EigImg = np.dot(np.transpose(NormImgData),EigV)
        EigImg = np.transpose(EigImg)
        EigImg = EigImg.reshape(self.OriShape[0],self.OriShape[1],self.OriShape[2])
        #TFF.imwrite("EigImg.tiff",EigImg)
        #ImgPath = QFileDialog.getOpenFileName()
        #TestImg = TFF.imread(ImgPath[0])
        #EigImg2 = self.getEigImage(EigV,TestImg)
        #TFF.imwrite("EigImgTest.tiff",EigImg2)
        #print(EigV[0])
        #print(EigV[1])
        return(V1,V2,ImgData)
    
    def getEigImage(self,EigV,Img):
        ImgShape = Img.shape
        ImgData = Img.reshape(ImgShape[0],ImgShape[1]*ImgShape[2])
        NormImgData = ImgData/np.sum(ImgData,axis = 0)
        EigImg = np.dot(np.transpose(NormImgData),EigV)
        EigImg = np.transpose(EigImg)
        EigImg = EigImg.reshape(ImgShape[0],ImgShape[1],ImgShape[2])
        return EigImg        
    
    def labelPixels(self,index):
        self.Pixels = list()
        for idx in index:

            x = idx//self.OriShape[2]
            y = idx%self.OriShape[2]

            GraphicItems = QGraphicsEllipseItem(x,y,2,2)
            GraphicItems.setPen(QPen(Qt.yellow,0.3,Qt.SolidLine))
            self.ImageWindow.view.addItem(GraphicItems)
            self.Pixels.append(GraphicItems)
    
    def openUnmixWizard(self):
        self.UnmixingWiz = DemixingWizard(self)
    
    def sendPixelPos(self,event):
        cursor = self.ImageWindow.view.mapSceneToView(event[0].pos())
        x = int(cursor.x())
        y = int(cursor.y())
        z = self.ImageWindow.currentIndex
        PixPos = x*self.OriShape[1]+y
        #print(y,self.OriShape[1],x,PixPos)
        #print(self.ImgData[z,PixPos])
        self.Plot.showClickedPixel(PixPos)

class DemixingWizard(QWidget):
    def __init__(self,parent):
        super().__init__()

        self.MainWin = parent
        self.setWindowTitle("Demixing wizard")
        self.ColorList = ["red","blue","green","white","purple","cyan","orange","yellow"]
        self.NoText = QLabel(self)
        self.NoText.setText("No of channels :")
        self.NoSpinBox = QSpinBox(self)
        self.NextButton = QPushButton(self)
        self.NextButton.setText("Next")
        self.NextButton.clicked.connect(self.showChannelFormula)

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(self.NoText,0,0,1,2)
        self.Layout.addWidget(self.NoSpinBox,0,2,1,2)
        self.Layout.addWidget(self.NextButton,1,2,1,2)
        self.setLayout(self.Layout)

        self.show()

    def ColorSelectComboBox(self,ColorList):
        ColorBox =QComboBox(self)
        for color in ColorList:
            ColorBox.addItem(convertQColorToQIcon(QColor(color)),"")    
        return ColorBox   

    def showChannelFormula(self):
        self.WidgetList = list()
        self.Name = QLabel(self)
        self.Name.setText("Channel name")
        self.Path = QLabel(self)
        self.Path.setText("File path")
        self.Empty = QLabel(self)
        self.Deduct = QLabel(self)
        self.Deduct.setText("Deduct from:")
        self.Layout.addWidget(self.Name,1,0,1,2)
        self.Layout.addWidget(self.Path,1,2,1,6)
        self.Layout.addWidget(self.Empty,1,8,1,2)
        self.Layout.addWidget(self.Deduct,1,10,1,2)
        for count in range(self.NoSpinBox.value()):
            widget = self.createWidgets(count)
            self.WidgetList.append(widget)
            self.Layout.addWidget(self.WidgetList[count][0],count+2,0,1,2)
            self.Layout.addWidget(self.WidgetList[count][1],count+2,2,1,6)
            self.Layout.addWidget(self.WidgetList[count][2],count+2,8,1,2)
            self.Layout.addWidget(self.WidgetList[count][3],count+2,10,1,2)
            self.Layout.addWidget(self.WidgetList[count][4],count+2,12,1,1)
        self.Layout.removeWidget(self.NextButton)
        self.NextButton.disconnect()
        self.NextButton.clicked.connect(self.unmix)
        self.ShowSpectraButton = QPushButton(self)
        self.ShowSpectraButton.setText("Plot their spectra")
        self.ShowSpectraButton.clicked.connect(self.showSpectra)            
        self.Layout.addWidget(self.NextButton,self.NoSpinBox.value()+2,9,1,4)
        self.Layout.addWidget(self.ShowSpectraButton,self.NoSpinBox.value()+2,4,1,5)

    def showSpectra(self):    
        RefSpecta = self.loadSpectra()
        SpectraPlot = pg.plot()
        for idx,Spectrum in enumerate(RefSpecta):
            Color = self.ColorList[self.WidgetList[idx][4].currentIndex()]
            SpectrumPlot = pg.PlotCurveItem(pen = pg.mkPen(color = QColor(Color), width = 3))
            SpectrumPlot.setData(Spectrum)
            SpectraPlot.addItem(SpectrumPlot)
    
    def createWidgets(self,idx):
        ChannelName = QLineEdit(self)
        ChannelPath = QLineEdit(self)
        ChannelPath.setDisabled(True)
        ChannelSubtract = QSpinBox(self)
        ChannelSubtract.setValue(0)
        ChannelSubtract.setMaximum(self.NoSpinBox.value())
        ChannelPathButton = QPushButton(self)
        ChannelPathButton.setText("Browse...")
        ChannelPathButton.clicked.connect(self.browse)
        ChannelPathButton.setProperty("ChannelIdx",idx)
        ChannelColor = self.ColorSelectComboBox(self.ColorList)
        return [ChannelName,ChannelPath,ChannelPathButton,ChannelSubtract,ChannelColor]
    
    def browse(self):
        parent = self.sender()
        idx = parent.property("ChannelIdx")
        Path = QFileDialog.getOpenFileName()
        self.WidgetList[idx][1].setText(Path[0])
        if self.WidgetList[idx][0].text() == "":
            self.WidgetList[idx][0].setText(self.getChannelName(Path[0]))
    
    def getChannelName(self,PathName):
        for idx in range(1,len(PathName)):
            if PathName[-idx] == ".":
                EndIdx = len(PathName)-idx
            if PathName[-idx] == "/":
                StartIdx = len(PathName)-idx+1
                break

        return PathName[StartIdx:EndIdx]

    def readCSV(self,path):
        ThisData = list()
        with open(path) as csvFile:
            data = csv.reader(csvFile)
            for idx,row in enumerate(data):
                if idx != 0:
                    ThisData.append([float(n) for n in row])
        
        ThisData = np.asarray(ThisData)
        MeanData = np.mean(ThisData,axis = 0)
        return MeanData
    
    def loadSpectra(self):
        Count,done = QInputDialog.getInt(self,"Input dialoge","input total number of channels to exclude:",0,0,32,1)
        ExChan = list()
        if Count !=0 :
            for idx in range(Count):
                Num,done = QInputDialog.getInt(self,"Input dialoge","select the channel to exclude:",0,1,32,1)
                ExChan.append(Num)

        RefSpectaDict = list()
        TempData = list()
        for aWidget in self.WidgetList:
            #self.RefSpectaDict[aWidget[0].text()] = self.readCSV(aWidget[1].text())
            ThisData = self.readCSV(aWidget[1].text())
            self.mask = np.ones(len(ThisData),dtype = bool)
            self.mask[ExChan] = False
            ThisData = ThisData[self.mask,...]
            TempData.append(ThisData)

        for idx,Datum in enumerate(TempData):
            SubtractChannel = self.WidgetList[idx][3].value()
            if SubtractChannel == 0:
                pass
            else:
                Datum = Datum - TempData[SubtractChannel-1]
            RefSpectaDict.append(Datum)
        
        return RefSpectaDict

    def unmix(self):
        ChanNameList = dict()
        for idx,aWidget in enumerate(self.WidgetList):
            ChanNameList["Channel "+str(idx+1)] = aWidget[0].text()
        self.RefSpectaDict = self.loadSpectra()
        Output = self.solveLSQ(np.transpose(self.RefSpectaDict))
        TFF.imwrite("Demixing.tif",Output,metadata={"Info":"\n".join(ChanNameList)+"\n"})
        MsgBox = QMessageBox.information(self,"Info","The image is unmixed and saved")

    def solveLSQ(self,FeatureMx):
        CovMx = np.dot(np.transpose(FeatureMx),FeatureMx)
        Kernel = np.dot(np.linalg.inv(CovMx),np.transpose(FeatureMx))
        Image = self.MainWin.img

        PoolInput = [(Image[:,x,y][self.mask,...],Kernel) for x in range(Image.shape[2]) for y in range(Image.shape[1])]

        #PoolInput = [(idx,idx+1) for idx in range(self.img.shape[0])]
        t_start = time.time()

        with get_context("spawn").Pool(processes=2) as pool:
            Result = pool.starmap(unitSolve,PoolInput)
            pool.close()
            pool.join()

        Output = np.zeros(shape = (len(self.WidgetList),Image.shape[1],Image.shape[2]))
        for idx,aPixel in enumerate(Result):
            x = idx//Image.shape[1]
            y = idx%Image.shape[1]
            if aPixel < 0:
                aPixel = 0

            Output[:,x,y] = aPixel
        
        return Output

def unitSolve(aPixel,Kernel):
    return np.dot(Kernel,aPixel) 

def convertQColorToQIcon(color):
    # color: QColor 
    # Create a QPixmap object with the desired size and color
    sizeV = 15  # desired size of the icon in pixels
    sizeH = 20
    pixmap = QPixmap(sizeH, sizeV)
    pixmap.fill(color)

    # Create a QIcon object from the pixmap
    icon = QIcon(pixmap)

    return icon

class Plot2D(QWidget):
    def __init__(self,parent, XData,YData,ImgData):
        super().__init__()

        self.MainWin = parent
        self.OriImage = ImgData
        self.XData = XData
        self.YData = YData
        ImgData = np.max(self.OriImage , axis = 0)
        self.OriData = [[x,y,I] for x,y,I in zip(XData,YData,ImgData)]
        Intensity = np.asarray([v[2] for v in self.OriData])
        self.Data = [[v[0],v[1],v[2],idx] for idx,v in enumerate(self.OriData)]

        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter.setData(XData,YData)
        self.isKeyPressedEnabled = False

        self.ThresholdSlider = QSlider(self)
        self.ThresholdSlider.setOrientation(Qt.Horizontal)
        self.ThresholdSlider.setMaximum(np.max(Intensity))
        self.ThresholdSlider.valueChanged.connect(self.resetData)
        self.CursorButton = QPushButton(self)
        self.CursorButton.setText("EnableCursor")
        self.CursorButton.clicked.connect(self.enableCursor)
        self.CursorInfo = QLineEdit(self)
        self.CursorInfo.setReadOnly(True)
        self.DrawROI = QPushButton(self)
        self.DrawROI.setText("Draw a ROI")
        self.DrawROI.clicked.connect(self.startDraw)
        self.DrawROI.setProperty("isContinouous",True)
        self.LabelButton = QPushButton(self)
        self.LabelButton.setText("Label ROI")
        self.LabelButton.setDisabled(True)
        self.LabelButton.clicked.connect(self.LabelorSave)
        self.LabelButton.setProperty("LabelorSave",True)
        self.ThresholdValue = QSpinBox(self)
        self.ThresholdValue.setMaximum(65535)
        self.ThresholdValue.valueChanged.connect(self.sendValueToSlider)
        self.ChannelSpinBox = QComboBox(self)
        ChannelList = [str(n) for n in range(self.OriImage.shape[0])]    
        self.ChannelSpinBox.addItems(["All"]+ChannelList)
        self.ChannelSpinBox.currentIndexChanged.connect(self.updateOriData)  

        self.plot = pg.plot()
        self.plot.addItem(self.scatter)
        self.AxisRange = self.plot.getPlotItem().getViewBox().getState()["viewRange"]

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(self.plot,0,0,4,4)
        self.Layout.addWidget(self.ThresholdSlider,5,1,1,2)
        self.Layout.addWidget(self.CursorButton,4,0,1,1)
        self.Layout.addWidget(self.CursorInfo,4,1,1,1)
        self.Layout.addWidget(self.DrawROI,4,2,1,1)
        self.Layout.addWidget(self.LabelButton,4,3,1,1)
        self.Layout.addWidget(self.ThresholdValue,5,3,1,1)
        self.Layout.addWidget(self.ChannelSpinBox,5,0,1,1)
        self.setLayout(self.Layout)
    
    def updateOriData(self):
        layer = self.ChannelSpinBox.currentIndex()

        if layer == 0:
            ImgData = np.max(self.OriImage , axis = 0)
            self.OriData = [[x,y,I] for x,y,I in zip(self.XData,self.YData,ImgData)]
        else:
            ImgData = self.OriImage[layer-1,:]
            self.OriData = [[x,y,I] for x,y,I in zip(self.XData,self.YData,ImgData)]
    
    def sendValueToSlider(self):
        value = self.ThresholdValue.value()
        self.ThresholdSlider.setValue(value)
        self.updatePCA(value)

    def resetData(self):
        value = self.ThresholdSlider.value()
        self.ThresholdValue.setValue(value)
    
    def updatePCA(self,value):
        self.plot.removeItem(self.scatter)
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))        
        self.plot.setXRange(self.AxisRange[0][0],self.AxisRange[0][1])
        self.plot.setYRange(self.AxisRange[1][0],self.AxisRange[1][1])
        self.Data = [[v[0],v[1],v[2],idx] if v[2] > value else -1 for idx,v in enumerate(self.OriData)]
        self.Data = [d for d in self.Data if d != -1]
        self.scatter.setData([x for x,_,_,_ in self.Data],[y for _,y,_,_ in self.Data])
        self.plot.addItem(self.scatter)
        if not self.LabelButton.property("LabelorSave"):
            self.findROIindex()    
        
    def enableCursor(self):
        self.Cursor = Qt.CrossCursor
        self.scatter.setCursor(self.Cursor)
        self.CursorEvent = pg.SignalProxy(self.plot.sceneObj.sigMouseMoved, delay = 0, slot = self.showValue)
        self.ClickEvent = pg.SignalProxy(self.plot.sceneObj.sigMouseClicked, delay = 0, slot = self.displayCursor)

    def showValue(self,Cursor):
        cursor = self.plot.plotItem.vb.mapSceneToView(Cursor[0])
        self.CursorPos = [cursor.x(),cursor.y()]       

    def displayCursor(self):
        self.isKeyPressedEnabled = True
        self.DataIdx = self.findLocalIndex(self.CursorPos)
        self.displayValue()
    
    def findLocalIndex(self,CursorPos):
        d_Data = np.asarray(self.Data)[:,0:2]-np.asarray(CursorPos)
        Norm = np.linalg.norm(d_Data,axis = 1)
        idx = np.where(Norm == np.min(Norm))
        return idx[0][0]
    
    def displayValue(self):
        x = self.Data[self.DataIdx][0]
        y = self.Data[self.DataIdx][1]
        m = self.Data[self.DataIdx][2]
        if hasattr(self,"CursorPlot"):
            self.plot.removeItem(self.CursorPlot)
        self.CursorPlot = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0, 255))
        self.CursorPlot.addPoints([x],[y])
        self.plot.addItem(self.CursorPlot)
        self.CursorInfo.setText("%d : (x,y,Intensity) = (%.2f,%.2f,%d)"%(self.DataIdx,x,y,m))

    def keyPressEvent(self, event):
        if self.isKeyPressedEnabled:
            if event.key() == Qt.Key_Right:
                self.DataIdx = self.DataIdx+1
                self.plot.removeItem(self.CursorPlot)
                self.displayValue()

            if event.key() == Qt.Key_Left:
                if self.DataIdx > 0:
                    self.DataIdx = self.DataIdx-1
                    self.plot.removeItem(self.CursorPlot)
                    self.displayValue()
    
    def startDraw(self):
        self.isContinuous = self.DrawROI.property("isContinouous")
        #print(self.plot.sceneObj.mousePressEvent.)
        if self.isContinuous:
            if hasattr(self,"CurveLines"):
                for aLine in self.CurveLines:
                    self.plot.removeItem(aLine)
            else:
                self.CurveLines = list()
            self.CurvePoints = list()
            self.DrawROI.setText("Finish Drawing")
            self.DrawStartEvent = pg.SignalProxy(self.plot.sceneObj.sigMouseClicked, delay = 0, slot = self.draw)
            self.DrawROI.setProperty("isContinouous",False)
        if not self.isContinuous:
            self.getDrawEnd()
            self.DrawROI.setText("Draw ROI")
            self.DrawROI.setProperty("isContinouous",True)
            self.DrawStartEvent.disconnect()
            self.drawEvent.disconnect()
            self.LabelButton.setEnabled(True)

    def draw(self,Cursor):
        if not self.isContinuous:
            self.LineStartPos = [x,y]
            self.drawEvent = pg.SignalProxy(self.plot.sceneObj.sigMouseMoved, delay = 0, slot = self.drawALine)
        else:
            if not hasattr(self,"LineStartPos"):
                Cursor = self.plot.plotItem.vb.mapSceneToView(Cursor[0].scenePos())
                x = Cursor.x()
                y = Cursor.y()
                self.LineStartPos = [x,y]
                self.drawEvent = pg.SignalProxy(self.plot.sceneObj.sigMouseMoved, delay = 0, slot = self.drawALine)   
            else:
                x = self.cursorX
                y = self.cursorY
                self.LineEndPos = [x,y]
                self.CurveLines.append(self.drawTheLine(self.LineStartPos,self.LineEndPos,isEnd=True))
                self.LineStartPos = self.LineEndPos
            self.CurvePoints.append([x,y]) 
    
    def getDrawEnd(self):
        LineEndPos = self.CurvePoints[0]
        self.drawTheLine(self.LineStartPos,LineEndPos,isEnd = True)

    def drawALine(self,Cursor):
        Cursor = Cursor[0]
        cursor = self.plot.plotItem.vb.mapSceneToView(Cursor)
        self.cursorX = cursor.x()
        self.cursorY = cursor.y()

        self.drawTheLine(self.LineStartPos,[self.cursorX,self.cursorY])        

    def drawTheLine(self,pos1,pos2,isEnd = False):
        if not isEnd or not self.isContinuous:
            if hasattr(self,"TempLine"):
                self.plot.removeItem(self.TempLine)

        if not self.isContinuous:
            self.Line = QGraphicsLineItem()
            self.Line.setPen(QPen(Qt.yellow, 0.001, Qt.SolidLine))
            self.Line.setLine(pos1[0],pos1[1],pos2[0],pos2[1])
            self.plot.addItem(self.Line)
        else:
            TempLine = QGraphicsLineItem()
            TempLine.setPen(QPen(Qt.yellow, 0.001, Qt.SolidLine))
            TempLine.setLine(pos1[0],pos1[1],pos2[0],pos2[1])
            if not isEnd:
                self.TempLine = TempLine
                self.plot.addItem(self.TempLine)
            else:
                self.CurveLines.append(TempLine)
                self.plot.addItem(TempLine)
                return TempLine
    
    def LabelorSave(self):
        LoS = self.LabelButton.property("LabelorSave")
        if LoS:
            self.findROIindex()
            self.LabelButton.setProperty("LabelorSave",False)
            self.LabelButton.setText("Save Result")
        else:
            self.saveROI()
            self.LabelButton.setProperty("LabelorSave",True)
            self.LabelButton.setText("Label ROI")
    
    def saveROI(self):
        OutputData = list()
        for idx in self.index:
            OutputData.append(self.MainWin.ImgData[:,idx])

        Name = QFileDialog.getSaveFileName()  
        with open(Name[0]+".csv","w+",newline="") as csvfile:
            wr = csv.writer(csvfile)
            wr.writerows(OutputData)
        OutputDataMean = np.mean(np.asarray(OutputData),axis = 0)
        pg.plot(OutputDataMean)
        OutputDataMean = OutputDataMean.tolist()
        MeanRowList = list()
        for v in OutputDataMean:
            MeanRowList.append([v])
        NameStat = Name[0]+"_Mean_I.csv"
        with open(NameStat,"w+",newline="") as csvfile:
            wr = csv.writer(csvfile)
            wr.writerows(MeanRowList)        

    def findROIindex(self):
        # given a contour defined by a set of vectors (CurvePoint and ROIvec) which form a closed loop, 
        # all the points inside the loop will have the same sign of their cross product with all the contour vectors 
        CurvePointShift = self.CurvePoints.copy()
        CurvePointShift[:-1] = self.CurvePoints[1:]
        CurvePointShift[-1] = self.CurvePoints[0]
        ROIvec = [np.asarray(a)-np.asarray(b) for a,b in zip(CurvePointShift,self.CurvePoints)] 
        idx = 0
        Data = [[x,y] for x,y,_,_ in self.Data]
        for roiVec,onePoint in zip(ROIvec,self.CurvePoints):
            ThisData = np.cross((np.asarray(Data)-np.asarray(onePoint)),roiVec)
            ThisData = ThisData//np.abs(ThisData)
            if idx == 0:
                d_Data = ThisData
                idx = 1
            else:
                d_Data = d_Data + ThisData

        idx = np.where(abs(d_Data) == len(self.CurvePoints))[0]
        XData = [self.Data[n][0] for n in idx]
        YData = [self.Data[n][1] for n in idx]
        self.index = [self.Data[n][3] for n in idx]
        self.CursorPlot = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 0, 0, 255))
        self.CursorPlot.addPoints(XData,YData)
        self.plot.addItem(self.CursorPlot)
        for aLine in self.CurveLines:
            self.plot.removeItem(aLine)
        
        self.MainWin.labelPixels(self.index)
    
    def closeEvent(self, a0):
        if hasattr(self.MainWin,"Pixels"):
            for aGraphicItem in self.MainWin.Pixels:
                self.MainWin.ImageWindow.view.removeItem(aGraphicItem)
        return super().closeEvent(a0)
    
    def showClickedPixel(self,PixPos):
        TheData = self.OriData[PixPos]
        x = TheData[0]
        y = TheData[1]
        if hasattr(self,"PixelPoint"):
            self.plot.removeItem(self.PixelPoint)
        self.PixelPoint = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0, 255))
        self.PixelPoint.addPoints([x],[y])
        self.plot.addItem(self.PixelPoint)

def findDir(DirStr):
    for idx in range(len(DirStr)):
        if DirStr[-idx-1] == "/":
            break
    return DirStr[:-idx-1]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ImgPath = QFileDialog.getOpenFileName()
    ImgPath =ImgPath[0]
    os.chdir(findDir(ImgPath))
    Win = PCAImageWin(ImgPath)
    Win.show()
    sys.exit(app.exec())