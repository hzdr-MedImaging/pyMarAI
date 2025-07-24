import sys
import os
import subprocess
import multiprocessing
import logging
import login
import predict
import shutil
import numpy as np
import pmedio
import time

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
    # no-op for pipes, as send is usually immediate
    pass

class PyMarAiGuiApp(QDialog):
    # signals to update GUI from worker threads/processes
    update_progress_text_signal = pyqtSignal(str)
    update_progress_bar_signal = pyqtSignal(int, int, str)
    processing_finished_signal = pyqtSignal()
    processing_started_signal = pyqtSignal()
    update_retrain_progress_text_signal = pyqtSignal(str)

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
        self.retrainPreviewList = []
        self.previewIndex = 0
        self.retrainPreviewIndex = 0
        self.predictionThread = None
        self.retrainThread = None
        self.fileLoader = None
        self.processingRunning = False
        self.outputBasenames = set()

        self.update_progress_text_signal.connect(self.showProgressMessage)
        self.update_progress_bar_signal.connect(self.updateProgressBarDetailed)
        self.processing_finished_signal.connect(self.processingFinished)
        self.processing_started_signal.connect(self.processingStarted)
        self.update_retrain_progress_text_signal.connect(self.showRetrainProgressMessage)

        self.tab_widget = QTabWidget()
        self.prediction_tab = QWidget()
        self.retrain_tab = QWidget()

        self.tab_widget.addTab(self.prediction_tab, "Prediction")
        self.tab_widget.addTab(self.retrain_tab, "Re-training")

        self.setupPredictionTab()
        self.setupRetrainTab()

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        self.setWindowTitle("Spheroids-DNN: Auto Inference Tool")
        self.resize(1200, 1000)

        self.initElements()

    def setupPredictionTab(self):
        # setup for the prediction tab
        inputFileLabel = self.createLabel("Input folder:")
        inputFileLabel.setStyleSheet("color: #333; font-weight: bold;")
        self.inputDirButton = self.createButton("Browse", self.loadInputDirectory)
        self.selectAllButton = self.createButton("Select All", self.selectAllFiles)
        self.deselectAllButton = self.createButton("Deselect All", self.deselectAllFiles)
        self.openRoverButton = self.createButton("Open in ROVER", self.openAllSelectedFilesInRover)

        self.inputFileListWidget = QListWidget()
        self.inputFileListWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.inputFileListWidget.setFixedHeight(600)
        self.inputFileListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.inputFileListWidget.itemSelectionChanged.connect(self.updatePreviewList)
        self.inputFileListWidget.itemClicked.connect(self.showImageOnItemClick)
        self.inputFileListWidget.itemDoubleClicked.connect(self.openAnalyzedFile)

        self.imagePreviewLabel = self.createLabel("Image Preview")
        self.imagePreviewLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.imagePreviewLabel.setFixedHeight(600)
        self.imagePreviewLabel.setMinimumWidth(800)
        self.imagePreviewLabel.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.imagePreviewLabel.setAlignment(Qt.AlignCenter)

        self.imageFilenameLabel = self.createLabel("No file selected")
        self.imageFilenameLabel.setAlignment(Qt.AlignCenter)
        self.imageFilenameLabel.setStyleSheet("color: #333; font-weight: bold;")

        self.prevButton = self.createButton("Previous", self.showPreviousImage)
        self.nextButton = self.createButton("Next", self.showNextImage)

        navLayout = QHBoxLayout()
        navLayout.addStretch()
        navLayout.addWidget(self.imageFilenameLabel)
        navLayout.addWidget(self.prevButton)
        navLayout.addWidget(self.nextButton)

        inputFilePathButtonsLayout = QHBoxLayout()
        inputFilePathButtonsLayout.addWidget(self.inputDirButton)
        inputFilePathButtonsLayout.addWidget(self.selectAllButton)
        inputFilePathButtonsLayout.addWidget(self.deselectAllButton)
        inputFilePathButtonsLayout.addWidget(self.openRoverButton)
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
        self.retrainInputFileListWidget.setFixedHeight(600)
        self.retrainInputFileListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.retrainInputFileListWidget.itemSelectionChanged.connect(self.updateRetrainPreviewList)
        self.retrainInputFileListWidget.itemClicked.connect(self.showRetrainImageOnItemClick)

        self.retrainImagePreviewLabel = self.createLabel("Image Preview")
        self.retrainImagePreviewLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.retrainImagePreviewLabel.setFixedHeight(600)
        self.retrainImagePreviewLabel.setMinimumWidth(600)
        self.retrainImagePreviewLabel.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.retrainImagePreviewLabel.setAlignment(Qt.AlignCenter)

        self.retrainImageFilenameLabel = self.createLabel("No file selected")
        self.retrainImageFilenameLabel.setAlignment(Qt.AlignCenter)
        self.retrainImageFilenameLabel.setStyleSheet("color: #333; font-weight: bold;")

        self.retrainPrevButton = self.createButton("Previous", self.showPreviousRetrainImage)
        self.retrainNextButton = self.createButton("Next", self.showNextRetrainImage)

        retrainNavLayout = QHBoxLayout()
        retrainNavLayout.addStretch()
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
        self.update_progress_text_signal.emit(f"Scanning directory for prediction files: {dir_path}.\n") # Updated message
        self.inputFileListWidget.clear()
        self.setProgressBarText("Loading files...")
        self.previewList = []

        self.processingRunning = True
        self.enableWidgets(False)
        self.predictionButton.setEnabled(False)

        self.fileLoader = FileLoaderWorker(dir_path, (".tif", ".png"))
        self.fileLoader.filesLoaded.connect(self.onPredictionFilesLoaded)
        self.fileLoader.errorOccurred.connect(self.onFileLoadError)
        self.fileLoader.finished.connect(self.predictionFileLoadingFinished)
        self.fileLoader.start()

    def loadRetrainFilesFromDirectory(self, dir_path):
        self.update_retrain_progress_text_signal.emit(f"Scanning directory for re-training files: {dir_path}.\n")
        self.retrainInputFileListWidget.clear()
        self.retrainPreviewList = []

        self.processingRunning = True
        self.enableWidgets(False)
        self.predictionButton.setEnabled(False)
        #self.retrainButton.setEnabled(False)

        self.retrainFileLoader = FileLoaderWorker(dir_path, (".v", ".rdf"))
        self.retrainFileLoader.filesLoaded.connect(self.onRetrainFilesLoaded)
        self.retrainFileLoader.errorOccurred.connect(self.showRetrainProgressMessage)
        self.retrainFileLoader.finished.connect(self.retrainFileLoadingFinished)
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

        self.update_progress_text_signal.emit(f"Found {len(file_list)} compatible prediction files.\n") # Updated message

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

        self.update_retrain_progress_text_signal.emit(f"Found {len(file_list)} compatible re-training files.\n")

    def onFileLoadError(self, error_message):
        self.update_progress_text_signal.emit(f"[ERROR] Error loading prediction files: {error_message}.\n")
        self.imagePreviewLabel.setText("Failed to load images")

    # prediction tab file loading completion
    def predictionFileLoadingFinished(self):
        self.processingRunning = False
        self.setProgressBarText()
        self.enableWidgets(True)
        self.predictionButton.setEnabled(True)
        self.updateOutputBasenames()
        self.markAnalyzedFiles()

    # re-training tab file loading completion
    def retrainFileLoadingFinished(self):
        self.processingRunning = False
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

    def showImageOnItemClick(self, item):
        filename = self.cleanFilename(item.text())
        self.previewList = [filename]
        self.previewIndex = 0
        self.showImageAtIndex(self.previewIndex)

    def showRetrainImageOnItemClick(self, item):
        filename = self.cleanFilename(item.text())
        self.retrainPreviewList = [filename]
        self.retrainPreviewIndex = 0
        self.showRetrainImageAtIndex(self.retrainPreviewIndex)

    def create_gradient_mask_overlay(self, original_image_pil, mask_pil, alpha=100, scale=255.0):
        # blue for darker pixels, green for lighter pixels
        start_color = (0, 0, 255)  # Blue
        end_color = (0, 255, 0)  # Green

        # convert original image to RGBA
        original_image_rgba = original_image_pil.convert('RGBA')

        # create an empty RGBA image for the overlay
        overlay_image = Image.new('RGBA', original_image_rgba.size, (0, 0, 0, 0))

        # get pixel data for both images
        original_pixels = original_image_rgba.load()
        mask_pixels = mask_pil.load()
        overlay_pixels = overlay_image.load()

        # get the size of the image
        width, height = original_image_rgba.size

        for y in range(height):
            for x in range(width):
                # apply overlay only where the mask is active
                if mask_pixels[x, y] > 0:
                    # get the average brightness of the original pixel
                    r, g, b, _ = original_pixels[x, y]
                    brightness = (r + g + b) // 3

                    # normalize brightness to a 0-1 scale
                    t = brightness / scale

                    # linearly interpolate between the start and end colors
                    r_grad = int(start_color[0] * (1 - t) + end_color[0] * t)
                    g_grad = int(start_color[1] * (1 - t) + end_color[1] * t)
                    b_grad = int(start_color[2] * (1 - t) + end_color[2] * t)

                    overlay_pixels[x, y] = (r_grad, g_grad, b_grad, alpha)

        # composite the original image with the new overlay
        return Image.alpha_composite(original_image_rgba, overlay_image)

    def showImageAtIndex(self, index):
        if not self.previewList:
            self.imagePreviewLabel.setText("No files selected")
            return

        filename = self.previewList[index]
        self.imageFilenameLabel.setText(filename)
        full_input_path = os.path.join(self.selectedInputDirectory, filename)
        output_dir = self.outputFilePathTextEdit.text().strip()

        # check if the current item is marked as analyzed
        current_item = None
        for i in range(self.inputFileListWidget.count()):
            item = self.inputFileListWidget.item(i)
            if self.cleanFilename(item.text()) == filename:
                current_item = item
                break

        is_analyzed_file = current_item and "[✓]" in current_item.text()

        self.imagePreviewLabel.setText("Loading...")

        try:
            if is_analyzed_file and os.path.isdir(output_dir):
                base_name, _ = os.path.splitext(filename)
                v_file_path_in_output = os.path.join(output_dir, base_name + ".v")
                rdf_file_path_in_output = os.path.join(output_dir, base_name + ".rdf")
                generated_mask_path = os.path.join(output_dir, base_name + "_cnn.v") # thrass output

                if os.path.exists(v_file_path_in_output) and os.path.exists(rdf_file_path_in_output):
                    # load the original input image (tif/png) as the background
                    if full_input_path.lower().endswith(".tif"):
                        original_image_pil = Image.open(full_input_path)
                        original_image_pil = ImageOps.exif_transpose(original_image_pil)
                        if original_image_pil.mode != "RGB":
                            original_image_pil = original_image_pil.convert("RGB")
                    else: # assuming .png
                        original_image_pil = Image.open(full_input_path)
                        if original_image_pil.mode != "RGB":
                            original_image_pil = original_image_pil.convert("RGB")

                    # use timestamp from .v file in output directory to tag cache
                    try:
                        v_file_mtime = os.path.getmtime(v_file_path_in_output)
                        v_file_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(v_file_mtime))
                    except Exception as e:
                        self.update_progress_text_signal.emit(f"[ERROR] Failed to get timestamp for {v_file_path_in_output}: {e}\n")
                        raise

                    cache_dir = "/tmp/marai_cnn_masks_prediction" # Separate cache for prediction tab
                    os.makedirs(cache_dir, exist_ok=True)
                    cached_mask_name = f"{base_name}_{v_file_timestamp}_cnn.v"
                    cached_mask_path = os.path.join(cache_dir, cached_mask_name)

                    mask_file_to_use = None

                    # reuse only if cache exists and is newer than source file
                    if os.path.exists(cached_mask_path) and os.path.getmtime(cached_mask_path) >= v_file_mtime:
                        self.update_progress_text_signal.emit(f"[INFO] Using cached mask for prediction: {cached_mask_path}\n")
                        mask_file_to_use = cached_mask_path
                    else:
                        self.update_progress_text_signal.emit(f"Running thrass for mask generation on {os.path.basename(v_file_path_in_output)} in output directory...\n")

                        # thrass command executed in the output directory
                        thrass_command = ["thrass", "-t", "cnnPrepare", "-b", os.path.basename(v_file_path_in_output)]
                        env = os.environ.copy()
                        result = subprocess.run(
                            thrass_command,
                            capture_output=True,
                            text=True,
                            cwd=output_dir, # Execute thrass in the output directory
                            env=env
                        )

                        self.update_progress_text_signal.emit(f"Thrass stdout:\n{result.stdout}\n")
                        if result.stderr:
                            self.update_progress_text_signal.emit(f"Thrass stderr:\n{result.stderr}\n")

                        if result.returncode != 0:
                            raise RuntimeError(f"Thrass failed with exit code {result.returncode} for file: {v_file_path_in_output}")

                        # wait briefly for file to appear
                        wait_time = 0
                        max_wait = 5
                        while not os.path.exists(generated_mask_path) and wait_time < max_wait:
                            time.sleep(0.5)
                            wait_time += 0.5

                        if not os.path.exists(generated_mask_path):
                            raise FileNotFoundError(
                                f"Thrass mask file not found at: {generated_mask_path}\n"
                                f"Check if RDF is missing or malformed for {v_file_path_in_output}."
                            )

                        # copy to cache
                        shutil.copy2(str(generated_mask_path), str(cached_mask_path))
                        self.update_progress_text_signal.emit(f"[INFO] Cached mask for prediction: {cached_mask_path}\n")
                        mask_file_to_use = cached_mask_path

                        # cleanup thrass output in the directory where it was generated
                        for fname in os.listdir(output_dir):
                            if fname.startswith(base_name + "_cnn") and fname.endswith(".v"):
                                try:
                                    os.remove(os.path.join(output_dir, fname))
                                    self.update_progress_text_signal.emit(
                                        f"[INFO] Deleted Thrass output file: {fname}\n")
                                except Exception as e:
                                    self.update_progress_text_signal.emit(f"[WARNING] Failed to delete {fname}: {e}\n")

                    if not mask_file_to_use:
                        raise FileNotFoundError("Mask file could not be determined or generated.")

                    # read and normalize mask
                    mask_data_pmedio = pmedio.read(mask_file_to_use)
                    mask_data = mask_data_pmedio.toarray()
                    mask_data_squeezed = np.squeeze(mask_data)

                    if mask_data_squeezed.max() > mask_data_squeezed.min():
                        normalized_mask_data = ((mask_data_squeezed - mask_data_squeezed.min()) /
                                                (mask_data_squeezed.max() - mask_data_squeezed.min()) * 255).astype(np.uint8)
                    else:
                        normalized_mask_data = np.zeros_like(mask_data_squeezed, dtype=np.uint8)

                    mask_pil = Image.fromarray(normalized_mask_data.T) # transpose if necessary based on data orientation
                    if mask_pil.mode != 'L':
                        mask_pil = mask_pil.convert('L')

                    mask_pil = mask_pil.rotate(180, expand=False)
                    mask_pil = mask_pil.resize(original_image_pil.size, Image.Resampling.NEAREST)

                    # create the gradient overlay
                    combined_image_pil = self.create_gradient_mask_overlay(original_image_pil, mask_pil, alpha=150, scale=125.0)

                    # convert to QPixmap for display
                    qimg = QImage(combined_image_pil.tobytes("raw", "RGBA"),
                                  combined_image_pil.width, combined_image_pil.height,
                                  QImage.Format_RGBA8888)
                    pixmap = QPixmap.fromImage(qimg)

                else:
                    # if analyzed files are marked but not found, fall back to original image
                    self.update_progress_text_signal.emit(f"[WARNING] Analyzed files ({v_file_path_in_output}, {rdf_file_path_in_output}) not found for '{filename}'. Displaying original input image.\n")
                    raise FileNotFoundError("Analyzed files not found, displaying original.")
            else:
                # fallback to the original image display logic for non-analyzed files
                if full_input_path.lower().endswith(".tif"):
                    image = Image.open(full_input_path)
                    image = ImageOps.exif_transpose(image)
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    data = image.tobytes("raw", "RGB")
                    width, height = image.size
                    qimg = QImage(data, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                else: # assuming .png
                    pixmap = QPixmap(full_input_path)
                    if pixmap.isNull():
                        raise ValueError("[ERROR] QPixmap could not load the image.\n")

            # fixed height for preview
            fixed_height = 600
            width, height = pixmap.width(), pixmap.height()
            aspect_ratio = width / height
            new_height = fixed_height
            new_width = int(fixed_height * aspect_ratio)

            # resize the label and scale the pixmap accordingly
            self.imagePreviewLabel.setFixedSize(new_width, new_height)
            scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imagePreviewLabel.setPixmap(scaled_pixmap)
            self.imagePreviewLabel.setText("")

        except Exception as e:
            self.imagePreviewLabel.setText("Failed to load image.\n")
            self.update_progress_text_signal.emit(f"[ERROR] Error loading image '{full_input_path}': {e}.\n")
            # print(f"Error loading image '{full_input_path}': {e}.\n") # Original print, now use signal

    # display images in the re-training tab's preview
    def showRetrainImageAtIndex(self, index):
        if not self.retrainPreviewList:
            self.retrainImagePreviewLabel.setText("No files selected")
            return

        filename_with_ext = self.cleanFilename(self.retrainPreviewList[index])
        filename_base, _ = os.path.splitext(filename_with_ext)

        v_filename = filename_base + ".v"
        v_file_path = os.path.join(self.selectedRetrainInputDirectory, v_filename)
        generated_mask_path = os.path.join(self.selectedRetrainInputDirectory, filename_base + "_cnn.v")

        # use timestamp from .v file to tag cache
        try:
            v_file_mtime = os.path.getmtime(v_file_path)
            v_file_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(v_file_mtime))
        except Exception as e:
            self.update_retrain_progress_text_signal.emit(f"[ERROR] Failed to get timestamp: {e}\n")
            return

        cache_dir = "/tmp/marai_cnn_masks"
        os.makedirs(cache_dir, exist_ok=True)
        cached_mask_name = f"{filename_base}_{v_file_timestamp}_cnn.v"
        cached_mask_path = os.path.join(cache_dir, cached_mask_name)

        self.retrainImageFilenameLabel.setText(filename_base)
        self.retrainImagePreviewLabel.setText("Loading, processing with thrass, and overlaying...")

        try:
            if not os.path.exists(v_file_path):
                raise FileNotFoundError(f".v file not found: {v_file_path}")

            v_data_pmedio = pmedio.read(v_file_path)
            original_v_data = v_data_pmedio.toarray()
            original_v_data_squeezed = np.squeeze(original_v_data)

            if original_v_data_squeezed.max() > original_v_data_squeezed.min():
                normalized_original_data = ((original_v_data_squeezed - original_v_data_squeezed.min()) /
                                            (
                                                        original_v_data_squeezed.max() - original_v_data_squeezed.min()) * 255).astype(
                    np.uint8)
            else:
                normalized_original_data = np.zeros_like(original_v_data_squeezed, dtype=np.uint8)

            original_image_pil = Image.fromarray(normalized_original_data.T)
            if original_image_pil.mode != 'RGB':
                original_image_pil = original_image_pil.convert('RGB')

            # reuse only if cache exists and is newer than source file
            if os.path.exists(cached_mask_path) and os.path.getmtime(cached_mask_path) >= v_file_mtime:
                self.update_retrain_progress_text_signal.emit(f"[INFO] Using cached mask: {cached_mask_path}\n")
                mask_file_to_use = cached_mask_path
            else:
                self.update_retrain_progress_text_signal.emit(
                    f"Running thrass for mask generation on {v_filename}...\n")

                thrass_command = ["thrass", "-t", "cnnPrepare", "-b", v_filename]
                env = os.environ.copy()
                result = subprocess.run(
                    thrass_command,
                    capture_output=True,
                    text=True,
                    cwd=self.selectedRetrainInputDirectory,
                    env=env
                )

                self.update_retrain_progress_text_signal.emit(f"Thrass stdout:\n{result.stdout}\n")
                if result.stderr:
                    self.update_retrain_progress_text_signal.emit(f"Thrass stderr:\n{result.stderr}\n")

                if result.returncode != 0:
                    raise RuntimeError(f"Thrass failed with exit code {result.returncode} for file: {v_file_path}")

                # wait briefly for file to appear
                wait_time = 0
                max_wait = 5
                while not os.path.exists(generated_mask_path) and wait_time < max_wait:
                    time.sleep(0.5)
                    wait_time += 0.5

                if not os.path.exists(generated_mask_path):
                    raise FileNotFoundError(
                        f"Thrass mask file not found at: {generated_mask_path}\n"
                        f"Check if RDF is missing or malformed for {v_file_path}."
                    )

                # copy to cache
                shutil.copy2(str(generated_mask_path), str(cached_mask_path))
                self.update_retrain_progress_text_signal.emit(f"[INFO] Cached mask: {cached_mask_path}\n")
                mask_file_to_use = cached_mask_path

                # cleanup thrass output in input directory
                for fname in os.listdir(self.selectedRetrainInputDirectory):
                    if fname.startswith(filename_base + "_cnn") and fname.endswith(".v"):
                        try:
                            os.remove(os.path.join(self.selectedRetrainInputDirectory, fname))
                            self.update_retrain_progress_text_signal.emit(
                                f"[INFO] Deleted Thrass output file: {fname}\n")
                        except Exception as e:
                            self.update_retrain_progress_text_signal.emit(f"[WARNING] Failed to delete {fname}: {e}\n")

            # read and normalize mask
            mask_data_pmedio = pmedio.read(mask_file_to_use)
            mask_data = mask_data_pmedio.toarray()
            mask_data_squeezed = np.squeeze(mask_data)

            if mask_data_squeezed.max() > mask_data_squeezed.min():
                normalized_mask_data = ((mask_data_squeezed - mask_data_squeezed.min()) /
                                        (mask_data_squeezed.max() - mask_data_squeezed.min()) * 255).astype(np.uint8)
            else:
                normalized_mask_data = np.zeros_like(mask_data_squeezed, dtype=np.uint8)

            mask_pil = Image.fromarray(normalized_mask_data.T)
            if mask_pil.mode != 'L':
                mask_pil = mask_pil.convert('L')
            mask_pil = mask_pil.resize(original_image_pil.size, Image.Resampling.NEAREST)

            # overlay
            overlay_color = (4, 95, 208)
            alpha = 100
            red_overlay = Image.new('RGBA', original_image_pil.size, (0, 0, 0, 0))
            pixels = red_overlay.load()
            mask_pixels = mask_pil.load()

            for y in range(original_image_pil.size[1]):
                for x in range(original_image_pil.size[0]):
                    if mask_pixels[x, y] > 0:
                        pixels[x, y] = overlay_color + (alpha,)

            # create the gradient overlay
            combined_image_pil = self.create_gradient_mask_overlay(original_image_pil, mask_pil, alpha=150, scale=255.0)

            # convert to QPixmap for display
            qimg = QImage(combined_image_pil.tobytes("raw", "RGBA"),
                            combined_image_pil.width, combined_image_pil.height,
                            QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)

            fixed_height = 600
            width, height = pixmap.width(), pixmap.height()
            aspect_ratio = width / height
            new_height = fixed_height
            new_width = int(fixed_height * aspect_ratio)

            self.retrainImagePreviewLabel.setFixedSize(new_width, new_height)
            scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.retrainImagePreviewLabel.setPixmap(scaled_pixmap)
            self.retrainImagePreviewLabel.setText("")

        except Exception as e:
            self.retrainImagePreviewLabel.setText(f"Error loading image: {e}")
            self.update_retrain_progress_text_signal.emit(f"[ERROR] {e}\n")
 
    def showNextImage(self):
        if self.previewList:
            self.previewIndex = (self.previewIndex + 1) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)

    def showPreviousImage(self):
        if self.previewList:
            self.previewIndex = (self.previewIndex - 1 + len(self.previewList)) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)

    def updatePreviewImage(self):
        if not self.currentImage:
            self.imagePreviewLabel.setText("No image to display")
            return

        # calculate the scale factor from the zoom step
        scale_factor = 1.0 + self.zoomStep * 0.1
        new_width = int(self.currentImage.width() * scale_factor)
        new_height = int(self.currentImage.height() * scale_factor)

        scaled_pixmap = self.currentImage.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imagePreviewLabel.setPixmap(scaled_pixmap)
        self.imagePreviewLabel.setText("")
        self.imagePreviewLabel.resize(new_width, new_height)

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

    def openAllSelectedFilesInRover(self):
        selected_items = self.inputFileListWidget.selectedItems()
        if not selected_items:
            self.update_progress_text_signal.emit("[ERROR] No files selected to open in ROVER.\n")
            return

        # prepare the list of filenames
        selected_filenames = [self.cleanFilename(item.text()) for item in selected_items if "[✓]" in item.text()]

        if not selected_filenames:
            self.update_progress_text_signal.emit(
                "[ERROR] None of the selected files are marked as analyzed. Skipping.\n")
            return

        # call the new method with the list of files
        self.openMultipleFilesInRover(selected_filenames)

    def selectAllRetrainFiles(self):
        self.retrainInputFileListWidget.selectAll()
        self.updateRetrainPreviewList()

    def deselectAllRetrainFiles(self):
        self.retrainInputFileListWidget.clearSelection()
        self.updateRetrainPreviewList()

    # --- Progress Bar Handling ---
    def setProgressBarText(self, text=None):
        # this is specifically for the prediction tab's progress bar
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
        # this is specifically for the prediction tab's progress bar
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
            self.predictionButton.setEnabled(True)  # keep enabled to allow stopping
        else:
            self.predictionButton.setText("Run Prediction")
            self.predictionButton.setEnabled(True)  # re-enable after prediction finishes or stops

    def openAnalyzedFile(self, item):
        text = item.text()
        filename = self.cleanFilename(text)

        if "[✓]" not in text:
            self.update_progress_text_signal.emit(
                f"[ERROR] '{filename}' is not marked as analyzed. Skipping open with ROVER.\n")
            return

        # Call the new method with a single file
        self.openMultipleFilesInRover([filename])

    def openMultipleFilesInRover(self, selected_filenames):
        output_dir = self.outputFilePathTextEdit.text().strip()
        if not os.path.isdir(output_dir):
            self.update_progress_text_signal.emit("[ERROR] Output directory is not valid.\n")
            QMessageBox.warning(self, "Open in ROVER", "Please select a valid output directory.")
            return

        v_files_to_open = []
        for filename in selected_filenames:
            base, _ = os.path.splitext(filename)

            # Find all .v files matching the base filename
            vFiles = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f)) and f.startswith(base) and f.endswith('.v')
            ]

            if not vFiles:
                self.update_progress_text_signal.emit(f"[ERROR] No .v file found for '{filename}'. Skipping.\n")
            else:
                v_files_to_open.extend(vFiles)

        if not v_files_to_open:
            self.update_progress_text_signal.emit("[ERROR] No valid .v files found to open in ROVER.\n")
            QMessageBox.warning(self, "Open in ROVER", "No valid files were found to open.")
            return

        # Prepare the command to open all .v files in a single ROVER instance
        command = ["rover", "-R", "1"] + v_files_to_open

        # Log the command and files
        self.update_progress_text_signal.emit(
            f"Opening {len(v_files_to_open)} file(s) in ROVER:\n"
            + "\n".join(v_files_to_open) + "\n"
        )

        try:
            subprocess.Popen(command)
        except Exception as e:
            self.update_progress_text_signal.emit(f"[ERROR] Failed to open ROVER: {e}\n")
            QMessageBox.warning(self, "Error Opening ROVER",
                                f"Could not open ROVER for selected files: {e}\nPlease ensure ROVER is installed and in your system PATH.")

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

        # Filter out files that have already been analyzed
        unprocessed_items = [item for item in selected_items if "[✓]" not in item.text()]
        processed_items = [item for item in selected_items if "[✓]" in item.text()]

        if not unprocessed_items:
            self.update_progress_text_signal.emit(
                "[INFO] All selected files are already analyzed. No new predictions will be run.\n")
            QMessageBox.information(self, "Prediction Info", "All selected files are already analyzed.")
            return None

        if processed_items:
            processed_filenames = [self.cleanFilename(item.text()) for item in processed_items]
            self.update_progress_text_signal.emit(
                f"[INFO] Skipping the following already analyzed files: {', '.join(processed_filenames)}\n")

        input_files = [
            os.path.join(self.selectedInputDirectory, self.cleanFilename(item.text()))
            for item in unprocessed_items
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

        self.update_progress_text_signal.emit(f"Running prediction with {len(input_files)} file(s).\n"
                                              f"Input files: {input_files}\n"
                                              f"Output dir: {output_dir}\n"
                                              f"Microscope: {microscope_number}\n")

        # return a dictionary of parameters for clarity
        return {
            "input_files": input_files,
            "output_dir": output_dir,
            "microscope_number": microscope_number
        }

    # show console-like messages during reconstruction
    def showProgressMessage(self, message):

        if not message.endswith('\n'):
            message += '\n'
        cursor = self.progressPlainTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(message)
        self.progressPlainTextEdit.ensureCursorVisible()

    # show progress messages for the re-training tab
    def showRetrainProgressMessage(self, message):
        if not message.endswith('\n'):
            message += '\n'
        cursor = self.retrainProgressPlainTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(message)
        self.retrainProgressPlainTextEdit.ensureCursorVisible()

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