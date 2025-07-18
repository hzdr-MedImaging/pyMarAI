import sys
import os
import subprocess
import multiprocessing
import logging
import login
import predict

from PyQt5 import QtCore
from multiprocessing import Pipe

from config import AppConfig

from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar, QWidget, QTabWidget,
                             QPushButton, QSizePolicy, QPlainTextEdit, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QMessageBox)

from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StdIORedirector:
  def __init__(self, pipe_connection):
    self.pipe = pipe_connection

  def write(self, string):
    if string and self.pipe and not self.pipe.closed:
      try:
        self.pipe.send(string)
      except Exception as e:
        logger.error(f"[ERROR] Error writing to pipe in StdIORedirector: {e}")

  def flush(self):
    # No-op for pipes, as send is usually immediate
    pass

class PyMarAiGuiApp(QDialog):
    # Signals to update GUI from worker threads/processes
    update_progress_text_signal = pyqtSignal(str)
    update_progress_bar_signal = pyqtSignal(int, int, str)
    processing_finished_signal = pyqtSignal()
    processing_started_signal = pyqtSignal()

    # we use constructor to create the GUI layout
    def __init__(self, config: AppConfig, parent=None):
        super(PyMarAiGuiApp, self).__init__(parent)

        self.config = config

        self.microscopes = config.get_microscopes()
        self.defaultMicroscopeType = config.get_default_microscope()

        self.settings = QSettings("PyMarAi", "PyMarAiGuiApp")
        self.selectedInputDirectory = self.settings.value("lastInputDir", os.getcwd())
        self.lastOutputDirectory = self.settings.value("lastOutputDir", os.getcwd())
        self.selectedRetrainInputDirectory = self.settings.value("lastRetrainInputDir", os.getcwd())
        self.lastRetrainOutputDirectory = self.settings.value("lastRetrainOutputDir", os.getcwd())
        self.currentImage = None
        self.currentImageIsPillow = False

        self.previewList = []
        self.previewIndex = 0
        self.predictionThread = None
        self.retrainThread = None
        self.fileLoader = None
        self.processingRunning = False
        self.outputBasenames = set()

        self.update_progress_text_signal.connect(self.showProgressMessage)
        self.update_progress_bar_signal.connect(self.updateProgressBarDetailed)
        self.processing_finished_signal.connect(self.processingFinished)
        self.processing_started_signal.connect(self.processingStarted)

        self.tab_widget = QTabWidget()
        self.prediction_tab = QWidget()
        self.retrain_tab = QWidget() # New: Retrain tab

        self.tab_widget.addTab(self.prediction_tab, "Prediction")
        self.tab_widget.addTab(self.retrain_tab, "Re-training") # New: Add re-training tab

        self.setupPredictionTab()
        self.setupRetrainTab() # New: Setup re-training tab

        main_layout = QVBoxLayout(self) # Changed to QVBoxLayout for the tab widget
        main_layout.addWidget(self.tab_widget)

        self.setWindowTitle("Spheroids-DNN: Auto Inference Tool")
        self.resize(800, 800)

        self.initElements()

    def setupPredictionTab(self):
        # setup for the prediction tab
        inputFileLabel = self.createLabel("Input folder:")
        inputFileLabel.setStyleSheet("color: #333; font-weight: bold;")
        self.inputDirButton = self.createButton("Browse", self.loadInputDirectory)
        self.selectAllButton = self.createButton("Select All", self.selectAllFiles)
        self.deselectAllButton = self.createButton("Deselect All", self.deselectAllFiles)
        self.inputFileListWidget = QListWidget()
        self.inputFileListWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.inputFileListWidget.setFixedHeight(375)
        self.inputFileListWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.inputFileListWidget.itemSelectionChanged.connect(self.updatePreviewList)
        self.inputFileListWidget.itemDoubleClicked.connect(self.openAnalyzedFile)

        self.imagePreviewLabel = self.createLabel("Image Preview")
        self.imagePreviewLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.imagePreviewLabel.setFixedHeight(375)
        self.imagePreviewLabel.setMinimumWidth(375)
        self.imagePreviewLabel.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.imagePreviewLabel.setAlignment(Qt.AlignCenter)

        self.imageFilenameLabel = self.createLabel("No file selected")
        self.imageFilenameLabel.setAlignment(Qt.AlignCenter)
        self.imageFilenameLabel.setStyleSheet("color: #333; font-weight: bold;")

        self.prevButton = self.createButton("Previous", self.showPreviousImage)
        self.nextButton = self.createButton("Next", self.showNextImage)

        navLayout = QHBoxLayout()
        navLayout.addWidget(self.imageFilenameLabel)
        navLayout.addWidget(self.prevButton)
        navLayout.addWidget(self.nextButton)

        inputFilePathButtonsLayout = QHBoxLayout()
        inputFilePathButtonsLayout.addWidget(self.inputDirButton)
        inputFilePathButtonsLayout.addWidget(self.selectAllButton)
        inputFilePathButtonsLayout.addWidget(self.deselectAllButton)
        inputFilePathButtonsLayout.addStretch()

        inputFileShowLayout = QHBoxLayout()
        inputFileShowLayout.addWidget(self.inputFileListWidget)
        inputFileShowLayout.addWidget(self.imagePreviewLabel)

        self.outputFileLabel = self.createLabel("Output Folder:")
        self.outputFileLabel.setStyleSheet("color: #333; font-weight: bold;")
        self.outputFilePathTextEdit = self.createTextEdit()

        self.outputFilePathSelectButton = self.createButton("Browse", self.outputDirSelect)
        outputFilePathLayout = QHBoxLayout()
        outputFilePathLayout.addWidget(self.outputFileLabel)
        outputFilePathLayout.addWidget(self.outputFilePathTextEdit)
        outputFilePathLayout.addWidget(self.outputFilePathSelectButton)

        self.microscopeLabel = self.createLabel("Microscope:")
        self.microscopeLabel.setStyleSheet("color: #333; font-weight: bold;")
        self.microscopeComboBox = self.createComboBox()
        self.microscopeComboBox.addItems(self.microscopes)
        index = self.microscopeComboBox.findText(self.defaultMicroscopeType)
        if index != -1:
            self.microscopeComboBox.setCurrentIndex(index)

        microscopeLayout = QHBoxLayout()
        microscopeLayout.addWidget(self.microscopeLabel)
        microscopeLayout.setSpacing(23)
        microscopeLayout.addWidget(self.microscopeComboBox)

        predictionButtonLayout = QHBoxLayout()
        self.predictionButton = self.createButton("Run Prediction", self.predictionButtonPressed)
        predictionButtonLayout.addWidget(self.predictionButton)
        predictionButtonLayout.addStretch()
        self.progressBarLabel = QLabel()
        self.progressBarLabel.hide()
        self.progressBar = QProgressBar()
        self.progressBar.setStyleSheet("""
        QProgressBar {
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #0078d7;
            border-radius: 4px;
        }
        """)
        self.progressBar.setFixedHeight(8)
        self.progressBar.setFixedWidth(300)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.hide()
        predictionButtonLayout.addWidget(self.progressBarLabel)
        predictionButtonLayout.addWidget(self.progressBar)

        progressLabel = self.createLabel("Progress Output:")
        self.progressPlainTextEdit = self.createPlainTextEdit()

        prediction_tab_layout = QGridLayout(self.prediction_tab) # Apply layout to the tab widget
        prediction_tab_layout.setColumnMinimumWidth(0, 75)
        prediction_tab_layout.setColumnStretch(1, 1)
        prediction_tab_layout.setColumnStretch(3, 1)

        row = 0
        prediction_tab_layout.addWidget(inputFileLabel, row, 0)

        row += 1
        prediction_tab_layout.addLayout(inputFileShowLayout, row, 0, 1, 4)

        row += 1
        prediction_tab_layout.addLayout(inputFilePathButtonsLayout, row, 0)
        prediction_tab_layout.addLayout(navLayout, row, 3)

        row += 1
        prediction_tab_layout.addLayout(outputFilePathLayout, row, 0, 1, 4)

        row += 1
        prediction_tab_layout.addLayout(microscopeLayout, row, 0)

        row += 1
        prediction_tab_layout.addLayout(predictionButtonLayout, row, 0, 1, 4)

        row += 1
        prediction_tab_layout.addWidget(progressLabel, row, 0, 1, 4)

        row += 1
        prediction_tab_layout.addWidget(self.progressPlainTextEdit, row, 0, 1, 4)

    def setupRetrainTab(self):
        # setup for the re-training tab
        retrainInputFileLabel = self.createLabel("Input folder:")
        retrainInputFileLabel.setStyleSheet("color: #333; font-weight: bold;")
        self.retrainInputDirButton = self.createButton("Browse", self.loadRetrainInputDirectory)
        self.retrainSelectAllButton = self.createButton("Select All", self.selectAllRetrainFiles)
        self.retrainDeselectAllButton = self.createButton("Deselect All", self.deselectAllRetrainFiles)
        self.retrainInputFileListWidget = QListWidget()
        self.retrainInputFileListWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.retrainInputFileListWidget.setFixedHeight(375)
        self.retrainInputFileListWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.retrainInputFileListWidget.itemSelectionChanged.connect(self.updateRetrainPreviewList)
        # self.retrainInputFileListWidget.itemDoubleClicked.connect(self.openRetrainFile) # Future visualization

        self.retrainImagePreviewLabel = self.createLabel("Image Preview")
        self.retrainImagePreviewLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.retrainImagePreviewLabel.setFixedHeight(375)
        self.retrainImagePreviewLabel.setMinimumWidth(375)
        self.retrainImagePreviewLabel.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.retrainImagePreviewLabel.setAlignment(Qt.AlignCenter)

        self.retrainImageFilenameLabel = self.createLabel("No file selected")
        self.retrainImageFilenameLabel.setAlignment(Qt.AlignCenter)
        self.retrainImageFilenameLabel.setStyleSheet("color: #333; font-weight: bold;")

        self.retrainPrevButton = self.createButton("Previous", self.showPreviousRetrainImage)
        self.retrainNextButton = self.createButton("Next", self.showNextRetrainImage)

        retrainNavLayout = QHBoxLayout()
        retrainNavLayout.addWidget(self.retrainImageFilenameLabel)
        retrainNavLayout.addWidget(self.retrainPrevButton)
        retrainNavLayout.addWidget(self.retrainNextButton)

        retrainInputFilePathButtonsLayout = QHBoxLayout()
        retrainInputFilePathButtonsLayout.addWidget(self.retrainInputDirButton)
        retrainInputFilePathButtonsLayout.addWidget(self.retrainSelectAllButton)
        retrainInputFilePathButtonsLayout.addWidget(self.retrainDeselectAllButton)
        retrainInputFilePathButtonsLayout.addStretch()

        retrainInputFileShowLayout = QHBoxLayout()
        retrainInputFileShowLayout.addWidget(self.retrainInputFileListWidget)
        retrainInputFileShowLayout.addWidget(self.retrainImagePreviewLabel)

        self.retrainOutputFileLabel = self.createLabel("Output Folder:")
        self.retrainOutputFileLabel.setStyleSheet("color: #333; font-weight: bold;")
        self.retrainOutputFilePathTextEdit = self.createTextEdit()

        self.retrainOutputFilePathSelectButton = self.createButton("Browse", self.retrainOutputDirSelect)
        retrainOutputFilePathLayout = QHBoxLayout()
        retrainOutputFilePathLayout.addWidget(self.retrainOutputFileLabel)
        retrainOutputFilePathLayout.addWidget(self.retrainOutputFilePathTextEdit)
        retrainOutputFilePathLayout.addWidget(self.retrainOutputFilePathSelectButton)

        retrainProgressLabel = self.createLabel("Progress Output:")
        self.retrainProgressPlainTextEdit = self.createPlainTextEdit()

        retrain_tab_layout = QGridLayout(self.retrain_tab)
        retrain_tab_layout.setColumnMinimumWidth(0, 75)
        retrain_tab_layout.setColumnStretch(1, 1)
        retrain_tab_layout.setColumnStretch(3, 1)

        row = 0
        retrain_tab_layout.addWidget(retrainInputFileLabel, row, 0)

        row += 1
        retrain_tab_layout.addLayout(retrainInputFileShowLayout, row, 0, 1, 4)

        row += 1
        retrain_tab_layout.addLayout(retrainInputFilePathButtonsLayout, row, 0)
        retrain_tab_layout.addLayout(retrainNavLayout, row, 3)

        row += 1
        retrain_tab_layout.addLayout(retrainOutputFilePathLayout, row, 0, 1, 4)

        #row += 1
        #retrain_tab_layout.addLayout(retrainButtonLayout, row, 0, 1, 4)

        row += 1
        retrain_tab_layout.addWidget(retrainProgressLabel, row, 0, 1, 4)

        row += 1
        retrain_tab_layout.addWidget(self.retrainProgressPlainTextEdit, row, 0, 1, 4)


    # initial state of elements
    def initElements(self):
        self.enableWidgets(True)

        if os.path.isdir(self.lastOutputDirectory):
            self.outputFilePathTextEdit.insert(self.lastOutputDirectory)

        if os.path.isdir(self.lastRetrainOutputDirectory):
            self.retrainOutputFilePathTextEdit.insert(self.lastRetrainOutputDirectory)

        if os.path.isdir(self.selectedInputDirectory):
            self.loadFilesFromDirectory(self.selectedInputDirectory)

        if os.path.isdir(self.selectedRetrainInputDirectory):
            self.loadRetrainFilesFromDirectory(self.selectedRetrainInputDirectory)

    def enableWidgets(self, enable):
        # prediction tab widgets
        self.inputFileListWidget.setEnabled(enable)
        self.outputFilePathTextEdit.setEnabled(enable)
        self.inputDirButton.setEnabled(enable)
        self.outputFilePathSelectButton.setEnabled(enable)
        self.microscopeComboBox.setEnabled(enable)
        self.selectAllButton.setEnabled(enable)
        self.deselectAllButton.setEnabled(enable)
        self.prevButton.setEnabled(enable)
        self.nextButton.setEnabled(enable)

        # re-training tab widgets
        self.retrainInputFileListWidget.setEnabled(enable)
        self.retrainOutputFilePathTextEdit.setEnabled(enable)
        self.retrainInputDirButton.setEnabled(enable)
        self.retrainOutputFilePathSelectButton.setEnabled(enable)
        self.retrainSelectAllButton.setEnabled(enable)
        self.retrainDeselectAllButton.setEnabled(enable)
        self.retrainPrevButton.setEnabled(enable)
        self.retrainNextButton.setEnabled(enable)

    # function to create GUI elements
    def createButton(self, text, member):
        button = QPushButton(text)
        button.setMaximumWidth(button.fontMetrics().boundingRect(text).width() + 7)
        button.clicked.connect(member)
        return button

    def createLabel(self, text):
        label = QLabel(text)
        return label

    def createComboBox(self):
        comboBox = QComboBox()
        return comboBox

    def createPlainTextEdit(self):
        textEdit = QPlainTextEdit()
        textEdit.setReadOnly(True)
        return textEdit

    def createTextEdit(self):
        lineEdit = QLineEdit()
        lineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return lineEdit

    def outputDirSelect(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select a prediction output folder:')
        if dir != "":
            self.outputFilePathTextEdit.clear()
            self.outputFilePathTextEdit.insert(dir)
            self.settings.setValue("lastOutputDir", dir)

            self.updateOutputBasenames()

            if self.inputFileListWidget.count() > 0:
                self.markAnalyzedFiles()

    def retrainOutputDirSelect(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select a re-training output folder:')
        if dir != "":
            self.retrainOutputFilePathTextEdit.clear()
            self.retrainOutputFilePathTextEdit.insert(dir)
            self.settings.setValue("lastRetrainOutputDir", dir)

    def updateOutputBasenames(self):
        self.outputBasenames = set()
        output_dir = self.outputFilePathTextEdit.text().strip()
        if os.path.isdir(output_dir):
            self.outputBasenames = {
                os.path.splitext(f)[0]
                for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
            }

    def markAnalyzedFiles(self):
        output_dir = self.outputFilePathTextEdit.text().strip()

        for i in range(self.inputFileListWidget.count()):
            item = self.inputFileListWidget.item(i)
            original_text = item.text().split(" [")[0].strip()
            base = os.path.splitext(original_text)[0]

            # in MarAi, .v and .rdf are produced in the output directory
            v_path = os.path.join(output_dir, base + ".v")
            rdf_path = os.path.join(output_dir, base + ".rdf")

            if os.path.exists(v_path) and os.path.exists(rdf_path):
                item.setText(original_text + " [✓]")
                item.setForeground(Qt.darkGreen)
            else:
                item.setText(original_text)
                item.setForeground(Qt.black)

    def cleanFilename(self, text):
        return text.split(" [")[0].strip()

    def loadInputDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Prediction Input Folder")
        if dir_path:
            self.settings.setValue("lastInputDir", dir_path)
            self.loadFilesFromDirectory(dir_path)

    def loadRetrainInputDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Re-training Input Folder")
        if dir_path:
            self.settings.setValue("lastRetrainInputDir", dir_path)
            self.loadRetrainFilesFromDirectory(dir_path)

    def loadFilesFromDirectory(self, dir_path):
        self.update_progress_text_signal.emit(f"Scanning directory for prediction: {dir_path}.\n")
        self.inputFileListWidget.clear()
        self.setProgressBarText("Loading files...")
        self.previewList = []

        self.processingRunning = True  
        self.enableWidgets(False)  
        self.predictionButton.setEnabled(False) 

        self.fileLoader = FileLoaderWorker(dir_path, (".tif", ".png"))
        self.fileLoader.filesLoaded.connect(self.onPredictionFilesLoaded)
        self.fileLoader.errorOccurred.connect(self.onFileLoadError)
        self.fileLoader.finished.connect(self.fileLoadingFinished)
        self.fileLoader.start()

    def loadRetrainFilesFromDirectory(self, dir_path):
        self.update_progress_text_signal.emit(f"Scanning directory for re-training: {dir_path}.\n")
        self.retrainInputFileListWidget.clear()
        #self.setRetrainProgressBarText("Loading files...")
        self.retrainPreviewList = []

        self.processingRunning = True
        self.enableWidgets(False)
        self.predictionButton.setEnabled(False)
        #self.retrainButton.setEnabled(False)

        self.retrainFileLoader = FileLoaderWorker(dir_path, (".v", ".rdf"))
        self.retrainFileLoader.filesLoaded.connect(self.onRetrainFilesLoaded)
        self.retrainFileLoader.errorOccurred.connect(self.onFileLoadError)
        self.retrainFileLoader.finished.connect(self.fileLoadingFinished)
        self.retrainFileLoader.start()

    def onPredictionFilesLoaded(self, file_list, dir_path):
        self.selectedInputDirectory = dir_path
        self.inputFileListWidget.clear()
        self.inputFileListWidget.clearSelection()

        for file in file_list:
            item = QListWidgetItem(file)
            self.inputFileListWidget.addItem(item)

        self.markAnalyzedFiles() 

        if file_list:
            self.previewList = []
            self.previewIndex = 0
            self.imagePreviewLabel.clear()
            self.imagePreviewLabel.setText("Image Preview")
            self.imageFilenameLabel.setText("No file selected")
        else:
            self.imagePreviewLabel.clear()
            self.imagePreviewLabel.setText("No image files found")
            self.imageFilenameLabel.setText("")

        self.update_progress_text_signal.emit(f"Found {len(file_list)} compatible files for prediction.\n")

    def onRetrainFilesLoaded(self, file_list, dir_path):
        self.selectedRetrainInputDirectory = dir_path
        self.retrainInputFileListWidget.clear()
        self.retrainInputFileListWidget.clearSelection()

        for file in file_list:
            item = QListWidgetItem(file)
            self.retrainInputFileListWidget.addItem(item)

        if file_list:
            self.retrainPreviewList = []
            self.retrainPreviewIndex = 0
            self.retrainImagePreviewLabel.clear()
            self.retrainImagePreviewLabel.setText("Image Preview")
            self.retrainImageFilenameLabel.setText("No file selected")
        else:
            self.retrainImagePreviewLabel.clear()
            self.retrainImagePreviewLabel.setText("No .v or .rdf files found")
            self.retrainImageFilenameLabel.setText("")

        self.update_progress_text_signal.emit(f"Found {len(file_list)} compatible files for re-training.\n")

    def onFileLoadError(self, error_message):
        self.update_progress_text_signal.emit(f"[ERROR] Error loading files: {error_message}.\n")
        self.imagePreviewLabel.setText("Failed to load images")

    def fileLoadingFinished(self):
        self.processingRunning = False
        self.setProgressBarText()  
        self.enableWidgets(True)  
        self.predictionButton.setEnabled(True)  

    def updatePreviewList(self):
        items = self.inputFileListWidget.selectedItems()

        if items:
            self.previewList = [self.cleanFilename(item.text()) for item in items]
            self.previewIndex = 0
            self.showImageAtIndex(self.previewIndex)
        else:
            self.previewList = []
            self.imagePreviewLabel.clear()
            self.imagePreviewLabel.setText("Image Preview")
            self.imageFilenameLabel.setText("No file selected")

    def updateRetrainPreviewList(self):
        items = self.retrainInputFileListWidget.selectedItems()

        if items:
            self.retrainPreviewList = [self.cleanFilename(item.text()) for item in items]
            self.retrainPreviewIndex = 0
            self.showRetrainImageAtIndex(self.retrainPreviewIndex)
        else:
            self.retrainPreviewList = []
            self.retrainImagePreviewLabel.clear()
            self.retrainImagePreviewLabel.setText("Image Preview")
            self.retrainImageFilenameLabel.setText("No file selected")

    def showImageAtIndex(self, index):
        if not self.previewList:
            self.imagePreviewLabel.setText("No files selected")
            return

        filename = self.previewList[index]
        self.imageFilenameLabel.setText(filename)
        full_path = os.path.join(self.selectedInputDirectory, filename)
        self.imagePreviewLabel.setText("Loading...")

        try:
            if full_path.lower().endswith(".tif"):
                image = Image.open(full_path)
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                data = image.tobytes("raw", "RGB")
                width, height = image.size
                qimg = QImage(data, width, height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
            else:
                pixmap = QPixmap(full_path)
                if pixmap.isNull():
                    raise ValueError("[ERROR] QPixmap could not load the image.\n")
                width, height = pixmap.width(), pixmap.height()

            # Fixed height for preview
            fixed_height = 375
            aspect_ratio = width / height
            new_height = fixed_height
            new_width = int(fixed_height * aspect_ratio)

            # Resize the label and scale the pixmap accordingly
            self.imagePreviewLabel.setFixedSize(new_width, new_height)
            scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imagePreviewLabel.setPixmap(scaled_pixmap)
            self.imagePreviewLabel.setText("")

        except Exception as e:
            self.imagePreviewLabel.setText("Failed to load image.\n")
            self.update_progress_text_signal.emit(f"[ERROR] Error loading image '{full_path}': {e}.\n")
            # print(f"Error loading image '{full_path}': {e}.\n") # Original print, now use signal

    def showNextImage(self):
        if self.previewList:
            self.previewIndex = (self.previewIndex + 1) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)

    def showPreviousImage(self):
        if self.previewList:
            self.previewIndex = (self.previewIndex - 1 + len(self.previewList)) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)

    def showNextRetrainImage(self):
        if self.retrainPreviewList:
            self.retrainPreviewIndex = (self.retrainPreviewIndex + 1) % len(self.retrainPreviewList)
            self.showRetrainImageAtIndex(self.retrainPreviewIndex)

    def showPreviousRetrainImage(self):
        if self.retrainPreviewList:
            self.retrainPreviewIndex = (self.retrainPreviewIndex - 1 + len(self.retrainPreviewList)) % len(
                self.retrainPreviewList)
            self.showRetrainImageAtIndex(self.retrainPreviewIndex)

    def selectAllFiles(self):
        self.inputFileListWidget.selectAll()
        self.updatePreviewList()

    def deselectAllFiles(self):
        self.inputFileListWidget.clearSelection()
        self.updatePreviewList()

    def selectAllRetrainFiles(self):
        self.retrainInputFileListWidget.selectAll()
        self.updateRetrainPreviewList()

    def deselectAllRetrainFiles(self):
        self.retrainInputFileListWidget.clearSelection()
        self.updateRetrainPreviewList()

    # --- Progress Bar Handling ---
    def setProgressBarText(self, text=None):
        if text is None:
            self.progressBarLabel.setText("")
            self.progressBar.setValue(0)
            self.progressBar.setMaximum(100) 
            self.progressBar.hide()
            self.progressBarLabel.hide()
        else:
            self.progressBarLabel.setText(text)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)  
            self.progressBar.show()
            self.progressBarLabel.show()

    def updateProgressBarDetailed(self, current_count, total_count, filename):
        if total_count > 0:
            percentage = int((current_count / total_count) * 100)
            self.progressBar.setMaximum(total_count)  
            self.progressBar.setValue(current_count)  
            self.progressBarLabel.setText(f"Processing: {current_count}/{total_count} ({percentage}%)")
            self.progressBar.show()
            self.progressBarLabel.show()
            self.update_progress_text_signal.emit(f"Completed {filename} ({current_count}/{total_count}).\n")
        else:
            self.setProgressBarText("Starting prediction...") 

    # switching between two states of elements
    # for prediction and user interaction mode

    def switchElementsToPrediction(self, isPrediction):
        if isPrediction:
            self.predictionButton.setText("Stop")
            self.predictionButton.setEnabled(True)  # Keep enabled to allow stopping
        else:
            self.predictionButton.setText("Run Prediction")
            self.predictionButton.setEnabled(True)  # Re-enable after prediction finishes or stops

    def openAnalyzedFile(self, item):
        text = item.text()
        if "[✓]" not in text:
            self.update_progress_text_signal.emit(
                f"'[ERROR] {self.cleanFilename(text)}' is not marked as analyzed. Skipping open with ROVER.\n")
            return

        filename = self.cleanFilename(text)
        base = os.path.splitext(filename)[0]

        output_dir = self.outputFilePathTextEdit.text().strip()
        if not os.path.isdir(output_dir):
            self.update_progress_text_signal.emit("[ERROR] Output directory is not valid.\n")
            return

        # find all .v files matching the base filename in output_dir
        vFiles = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f)) and f.startswith(base) and f.endswith('.v')
        ]

        # find all .rdf files matching the base filename in output_dir
        rdfFiles = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f)) and f.startswith(base) and f.endswith('.rdf')
        ]

        if not vFiles:
            self.update_progress_text_signal.emit(f"[ERROR] No .v file found for '{filename}'.\n")
            return

        if not rdfFiles:
            self.update_progress_text_signal.emit(f"[ERROR] No .rdf file found for '{filename}'.\n")
            return

        # show which files you will open
        self.update_progress_text_signal.emit(
            f"Opening ROVER for '{filename}':\n"
            + "\n".join(vFiles + rdfFiles) + "\n"
        )

        # open each .v file with rover, with -R 1 option
        for vFile in vFiles:
            try:
                # Correct way to pass command + arguments:
                subprocess.Popen(["rover", "-R", "1", vFile])
            except Exception as e:
                self.update_progress_text_signal.emit(f"[ERROR] Failed to open {vFile}: {e}\n")
                QMessageBox.warning(self, "Error Opening ROVER",
                                    f"Could not open ROVER for {os.path.basename(vFile)}: {e}\nPlease ensure ROVER is installed and in your system PATH.")

    #############################################
    # handle the event of Run Prediction button pressing

    def predictionButtonPressed(self):
        if not self.processingRunning:
            # show login dialog for SSH credentials
            login_dialog = login.LoginDialog(self)
            if login_dialog.exec_() == QMessageBox.Accepted:  # Use QMessageBox.Accepted
                username, password = login_dialog.get_credentials()

                if not username or not password:
                    self.update_progress_text_signal.emit("Username or password cannot be empty.\n")
                    QMessageBox.warning(self, "Login Error", "Username or password cannot be empty.")
                    return

                # clear the progress output and prepare for job processing
                self.progressPlainTextEdit.clear()

                # build the prediction job parameters
                prediction_params = self.getPredictionParams()
                if prediction_params is None:
                    return

                # create and start the prediction thread
                self.predictionThread = PyMarAiThread(
                    parent=self,
                    params=prediction_params,
                    username=username,
                    password=password
                )
                self.predictionThread.started.connect(self.processing_started_signal.emit)
                self.predictionThread.finished.connect(self.processing_finished_signal.emit)
                self.predictionThread.progress_update.connect(self.update_progress_bar_signal)
                self.predictionThread.text_update.connect(self.update_progress_text_signal)
                self.predictionThread.error_message.connect(
                    lambda msg: QMessageBox.critical(self, "Prediction Error", msg))

                self.predictionThread.start()
                self.switchElementsToPrediction(True)

            else:
                self.update_progress_text_signal.emit("[ERROR] Login cancelled.\n")

        else:
            # we are running, so the button acts as "Stop"
            if self.predictionThread and self.predictionThread.isRunning():
                self.update_progress_text_signal.emit("\n*** Aborting prediction ***\n")
                # request termination of the process
                self.predictionThread.stop_prediction_process()
                # the finished signal will handle GUI cleanup

    # function to combine the selected parameters and actually
    # start the prediction by passing these parameters
    # to runPrediction.py main function (now getPredictionParams)

    def getPredictionParams(self):
        # collect input files
        selected_items = self.inputFileListWidget.selectedItems()
        if not selected_items:
            self.update_progress_text_signal.emit("[ERROR] No input files selected.\n")
            QMessageBox.warning(self, "Input Error", "Please select input files.")
            return None

        input_files = [
            os.path.join(self.selectedInputDirectory, self.cleanFilename(item.text()))
            for item in selected_items
        ]

        # output directory
        output_dir = self.outputFilePathTextEdit.text().strip()
        if not os.path.isdir(output_dir):
            self.update_progress_text_signal.emit("[ERROR] Invalid output directory.\n")
            QMessageBox.warning(self, "Input Error", "Please select a valid output directory.")
            return None

        # microscope selection
        microscope_text = self.microscopeComboBox.currentText()
        if microscope_text.strip() == "-":
            self.update_progress_text_signal.emit("No microscope selected.\n")
            QMessageBox.warning(self, "Input Error", "Please select a microscope.")
            return None

        # extract microscope code
        microscope_code = microscope_text.split(":")[0].strip()
        try:
            microscope_number = int(microscope_code)
        except ValueError:
            self.update_progress_text_signal.emit(f"[ERROR] Invalid microscope code: {microscope_code}.\n")
            QMessageBox.warning(self, "Input Error", "Invalid microscope selection.")
            return None

        self.update_progress_text_signal.emit(f"Running prediction with:\n"
                                              f"Input files: {input_files}\n"
                                              f"Output dir: {output_dir}\n"
                                              f"Microscope: {microscope_number}\n")

        # return a dictionary of parameters for clarity
        return {
            "input_files": input_files,
            "output_dir": output_dir,
            "microscope_number": microscope_number
        }

    # we use this function to show console-like messages during reconstruction
    def showProgressMessage(self, message):
        if not message.endswith('\n'):
            message += '\n'
        cursor = self.progressPlainTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(message)
        self.progressPlainTextEdit.ensureCursorVisible()

    # call when processing thread started
    def processingStarted(self):
        self.processingRunning = True
        self.enableWidgets(False)
        self.switchElementsToPrediction(True)

    # call when processing thread finished (or stopped)
    def processingFinished(self):
        self.processingRunning = False
        self.enableWidgets(True)
        self.progressBar.hide()
        self.progressBarLabel.hide()
        self.updateOutputBasenames()
        self.markAnalyzedFiles()
        self.switchElementsToPrediction(False)


class PyMarAiThread(QtCore.QThread):
    progress_update = pyqtSignal(int, int, str)
    text_update = pyqtSignal(str)
    error_message = pyqtSignal(str)

    def __init__(self, parent, params, username, password):
        super().__init__()
        self.parent = parent
        self.params = params
        self.username = username
        self.password = password
        self._process = None

        # Create pipes
        self.stdout_pipe_parent, self.stdout_pipe_child = Pipe(duplex=False)
        self.progress_pipe_parent, self.progress_pipe_child = Pipe(duplex=False)

        # Emitters
        self.stdout_emitter = self.Emitter(self.stdout_pipe_parent)
        self.stdout_emitter.message.connect(self.text_update)
        self.stdout_emitter.start()

        self.progress_emitter = self.ProgressEmitter(self.progress_pipe_parent)
        self.progress_emitter.progress.connect(self.progress_update)
        self.progress_emitter.start()

    def run(self):
        self.text_update.emit("Starting prediction process...\n")
        try:
            self._process = self.PredictionProcess(
                target=predict.gui_entry_point,
                params=self.params,
                username=self.username,
                password=self.password,
                stdout_pipe_child=self.stdout_pipe_child, 
                progress_pipe_child=self.progress_pipe_child, 
                hostname=self.parent.config.get_default_remote_hostname()
            )

            self._process.start()
            self._process.join() 

            if self._process.exitcode != 0:
                self.error_message.emit(f"[ERROR] Prediction process exited with error code: {self._process.exitcode}")
                self.text_update.emit(f"[ERROR] Prediction process failed with exit code: {self._process.exitcode}\n")
            else:
                self.text_update.emit("[INFO] Prediction process completed.\n")

        except Exception as e:
            self.error_message.emit(f"[ERROR] An unexpected error occurred: {e}")
            self.text_update.emit(f"[ERROR] An unexpected error occurred during prediction: {e}\n")
        finally:
            self.stdout_emitter.stop_and_wait()
            self.progress_emitter.stop_and_wait()

    def stop_prediction_process(self):
        if self._process and self._process.is_alive():
            self.text_update.emit("[INFO] Terminating prediction process...\n")
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self.text_update.emit("Warning: Process did not terminate gracefully.\n")
            else:
                self.text_update.emit("[INFO] Prediction process terminated.\n")

            # signal emitters to stop and wait for them to finish
            self.stdout_emitter.stop_and_wait()
            self.progress_emitter.stop_and_wait()

            self.finished.emit() # emit finished signal manually if stopped by user

    """ Inner classes for inter-process and inter-thread communication """

    class Emitter(QtCore.QThread):
        message = pyqtSignal(str)

        def __init__(self, pipe_connection):
            super().__init__()
            self.pipe = pipe_connection
            self._running = True

        def run(self):
            while self._running:
                try:
                    if self.pipe.poll(0.1):
                        msg = self.pipe.recv()
                        self.message.emit(msg)
                except EOFError:
                    # This happens when the writing end of the pipe is closed
                    logger.info("[INFO] Pipe EOF detected for Emitter.")
                    break
                except Exception as e:
                    logger.error(f"[ERROR] Error in Emitter thread: {e}")
                    break
            self._close_pipe()

        def stop(self):
            # this method just sets a flag. The thread will eventually terminate.
            self._running = False

        def stop_and_wait(self):
            # this method sets a flag and waits for the thread to terminate.
            self._running = False
            self.wait() # Wait for the thread's run method to complete

        def _close_pipe(self):
            # Only close if it's still open
            if self.pipe and not self.pipe.closed:
                try:
                    self.pipe.close()
                    logger.debug("[DEBUG] Emitter pipe closed.")
                except OSError as e:
                    logger.warning(f"[WARNING] Error closing Emitter pipe: {e}")


    class ProgressEmitter(QtCore.QThread):
        progress = pyqtSignal(int, int, str)

        def __init__(self, pipe_connection):
            super().__init__()
            self.pipe = pipe_connection
            self._running = True

        def run(self):
            while self._running:
                try:
                    if self.pipe.poll(0.1): # Poll with a timeout
                        data = self.pipe.recv()
                        if isinstance(data, tuple) and len(data) == 3:
                            current, total, filename = data
                            self.progress.emit(current, total, filename)
                except EOFError:
                    logger.info("[INFO] Pipe EOF detected for ProgressEmitter.")
                    break
                except Exception as e:
                    logger.error(f"[ERROR] Error in ProgressEmitter thread: {e}")
                    break
            self._close_pipe()

        def stop(self):
            self._running = False

        def stop_and_wait(self):
            self._running = False
            self.wait()

        def _close_pipe(self):
            if self.pipe and not self.pipe.closed:
                try:
                    self.pipe.close()
                    logger.debug("[DEBUG] ProgressEmitter pipe closed.")
                except OSError as e:
                    logger.warning(f"[WARNING] Error closing ProgressEmitter pipe: {e}")

    class PredictionProcess(multiprocessing.Process):
        # target, params, username, password, stdout_pipe_child, progress_pipe_child, hostname
        def __init__(self, target, **kwargs):
            super().__init__()
            self._target = target
            self._all_kwargs = kwargs

        def run(self):
            # extract the pipe connections and the stop_event from the kwargs
            stdout_pipe_child = self._all_kwargs.pop('stdout_pipe_child')
            progress_pipe_child = self._all_kwargs.pop('progress_pipe_child')
            stop_event = self._all_kwargs.pop('stop_event', None) 

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = StdIORedirector(stdout_pipe_child)
                sys.stderr = StdIORedirector(stdout_pipe_child)


                self._target(
                    progress_pipe_connection=progress_pipe_child,
                    stdout_pipe_connection=stdout_pipe_child,
                    stop_event=stop_event, 
                    **self._all_kwargs 
                )
            except Exception as e:
                error_msg = f"Unhandled exception in prediction process: {e}\n"
                logger.exception(error_msg)
                if not stdout_pipe_child.closed:
                    try:
                        stdout_pipe_child.send(f"[ERROR] {error_msg}")
                    except Exception as pipe_err:
                        sys.stderr.write(f"Failed to send error message through pipe: {pipe_err}\n")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                if stdout_pipe_child and not stdout_pipe_child.closed:
                    try:
                        stdout_pipe_child.close()
                        logger.debug("Child stdout pipe closed in PredictionProcess.run().")
                    except Exception as e:
                        logger.error(f"Error closing stdout_pipe_child in PredictionProcess: {e}")
                if progress_pipe_child and not progress_pipe_child.closed:
                    try:
                        progress_pipe_child.close()
                        logger.debug("Child progress pipe closed in PredictionProcess.run().")
                    except Exception as e:
                        logger.error(f"Error closing progress_pipe_child in PredictionProcess: {e}")

# thread to load data
class FileLoaderWorker(QThread):
    filesLoaded = pyqtSignal(list, str)
    errorOccurred = pyqtSignal(str)

    def __init__(self, dir_path, extensions):
        super().__init__()
        self.dir_path = dir_path
        self.extensions = extensions

    def run(self):
        if not os.path.isdir(self.dir_path):
            self.errorOccurred.emit(f"[ERROR] Selected directory is invalid: {self.dir_path}.\n")
            return

        file_list = [
            f for f in os.listdir(self.dir_path)
            if os.path.isfile(os.path.join(self.dir_path, f)) and f.lower().endswith(self.extensions)
        ]

        if not file_list:
            ext_str = ", ".join(self.extensions)
            self.errorOccurred.emit(
                f"[ERROR] No compatible files ({ext_str}) found in the selected folder: {self.dir_path}.\n")
            return

        file_list.sort()
        self.filesLoaded.emit(file_list, self.dir_path)


# main function to start GUI
def main():
    app = QApplication(sys.argv)
    config = AppConfig()
    window = PyMarAiGuiApp(config)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()