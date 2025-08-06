import sys
import os
import subprocess
import multiprocessing
import logging
import login
import predict
import traceback
import shutil
import pmedio
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PyQt5 import QtCore
from multiprocessing import Pipe

from config import AppConfig

from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QGridLayout, QHBoxLayout, QVBoxLayout,
                             QLabel, QProgressBar, QWidget, QTabWidget, QCheckBox, QPushButton, QSizePolicy, QPlainTextEdit,
                             QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QGroupBox, QColorDialog)

from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush, QPainter, QPainterPath
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal, QThreadPool
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main GUI application class for Spheroids-DNN
class PyMarAiGuiApp(QDialog):

    # signals to update GUI from worker threads/processes
    update_progress_text_signal = pyqtSignal(str)
    update_progress_bar_signal = pyqtSignal(int, int, str, str)
    processing_finished_signal = pyqtSignal()
    processing_started_signal = pyqtSignal()
    update_retrain_progress_text_signal = pyqtSignal(str)

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

        self.previewList = []
        self.retrainPreviewList = []
        self.previewIndex = 0
        self.retrainPreviewIndex = 0
        self.predictionThread = None
        self.retrainThread = None
        self.fileLoader = None
        self.processingRunning = False
        self.outputBasenames = set()

        self.thread_pool = QThreadPool.globalInstance()
        self.current_preview_filename = None

        self.predictionMaskedPixmaps = {}
        self.retrainMaskedPixmaps = {}

        self.originalPredictionImage = None
        self.originalRetrainImage = None

        self.tabType = "prediction"

        # connect signals to slots
        self.update_progress_text_signal.connect(self.showProgressMessage)
        self.update_progress_bar_signal.connect(self.updateProgressBarDetailed)
        self.processing_finished_signal.connect(self.processingFinished)
        self.processing_started_signal.connect(self.processingStarted)
        self.update_retrain_progress_text_signal.connect(self.showRetrainProgressMessage)

        # setup tab widget and tabs
        self.tab_widget = QTabWidget()
        self.prediction_tab = QWidget()
        self.retrain_tab = QWidget()

        self.tab_widget.addTab(self.prediction_tab, "Prediction")
        self.tab_widget.addTab(self.retrain_tab, "Re-training")

        self.setupPredictionTab()
        self.setupRetrainTab()

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        self.setWindowTitle("Spheroids-DNN: Auto Inference Tool")
        self.resize(1600, 1200)

        self.initElements()

    # sets up the layout and widgets for the prediction tab
    def setupPredictionTab(self):
        # --- Input File Section ---
        inputFileLabel = self.createLabel("Input folder:")
        self.inputDirButton = self.createButton("Browse", self.loadInputDirectory)
        self.selectAllButton = self.createButton("Select All", self.selectAllFiles)
        self.deselectAllButton = self.createButton("Deselect All", self.deselectAllFiles)
        self.openRoverButton = self.createButton("Open in ROVER", self.openAllSelectedFilesInRover)

        inputFilePathButtonsLayout = QHBoxLayout()
        inputFilePathButtonsLayout.addWidget(self.inputDirButton)
        inputFilePathButtonsLayout.addWidget(self.selectAllButton)
        inputFilePathButtonsLayout.addWidget(self.deselectAllButton)
        inputFilePathButtonsLayout.addWidget(self.openRoverButton)
        inputFilePathButtonsLayout.addStretch()

        self.inputFileListWidget = QListWidget()
        self.inputFileListWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.inputFileListWidget.setFixedHeight(600)
        self.inputFileListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.inputFileListWidget.itemSelectionChanged.connect(self.updatePreviewList)
        self.inputFileListWidget.itemDoubleClicked.connect(self.openAnalyzedFile)

        inputFileSectionWidget = QWidget()
        inputFileSectionLayout = QVBoxLayout(inputFileSectionWidget)
        inputFileSectionLayout.addWidget(inputFileLabel)
        inputFileSectionLayout.addWidget(self.inputFileListWidget)
        inputFileSectionLayout.addLayout(inputFilePathButtonsLayout)

        # --- Image Preview Section ---
        self.imagePreviewLabel = self.createLabel("Image Preview")
        self.imagePreviewLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.imagePreviewLabel.setFixedHeight(600)
        self.imagePreviewLabel.setMinimumWidth(800)
        self.imagePreviewLabel.setAlignment(Qt.AlignCenter)
        self.imagePreviewLabel.setStyleSheet("""
            border: 1px solid #cccccc;
            background-color: #fafafa;
            border-radius: 6px;
        """)

        self.imageFilenameLabel = self.createLabel("No file selected")
        self.imageFilenameLabel.setAlignment(Qt.AlignCenter)

        self.prevButton = self.createButton("Previous", self.showPreviousImage)
        self.nextButton = self.createButton("Next", self.showNextImage)

        imagePreviewContainerWidget = QWidget()
        imagePreviewContainerLayout = QVBoxLayout(imagePreviewContainerWidget)

        imagePreviewContainerLayout.addWidget(self.imageFilenameLabel, alignment=Qt.AlignCenter)
        imagePreviewContainerLayout.addWidget(self.imagePreviewLabel)

        prevNextButtonsHLayout = QHBoxLayout()
        prevNextButtonsHLayout.addStretch()
        prevNextButtonsHLayout.addWidget(self.prevButton)
        prevNextButtonsHLayout.addWidget(self.nextButton)
        prevNextButtonsHLayout.addStretch()
        imagePreviewContainerLayout.addLayout(prevNextButtonsHLayout)

        # --- Mask Buttons and Mask Display Options ---
        self.applyMaskButton = self.createButton("Apply Mask", self.applyPredictionMask)
        self.removeMaskButton = self.createButton("Remove Mask", self.removePredictionMask)

        maskButtonsHLayout = QHBoxLayout()
        maskButtonsHLayout.addWidget(self.applyMaskButton)
        maskButtonsHLayout.addWidget(self.removeMaskButton)
        maskButtonsHLayout.addStretch()

        mask_options_group_box = self.setupMaskDisplayOptions("prediction")

        maskControlsVWidget = QWidget()
        maskControlsVLayout = QVBoxLayout(maskControlsVWidget)
        maskControlsVLayout.addSpacing(21)
        maskControlsVLayout.addLayout(maskButtonsHLayout)
        maskControlsVLayout.addWidget(mask_options_group_box)

        # --- Output File Section ---
        self.outputFileLabel = self.createLabel("Output Folder:")
        self.outputFilePathTextEdit = self.createTextEdit()
        self.outputFilePathSelectButton = self.createButton("Browse", self.outputDirSelect)

        outputFilePathLayout = QHBoxLayout()
        outputFilePathLayout.addWidget(self.outputFileLabel)
        outputFilePathLayout.addWidget(self.outputFilePathTextEdit)
        outputFilePathLayout.addWidget(self.outputFilePathSelectButton)

        outputFilePathWidget = QWidget()
        outputFilePathWidget.setLayout(outputFilePathLayout)

        # --- Microscope Selection ---
        self.microscopeLabel = self.createLabel("Microscope:")
        self.microscopeComboBox = self.createComboBox()
        self.microscopeComboBox.addItems(self.microscopes)
        index = self.microscopeComboBox.findText(self.defaultMicroscopeType)
        if index != -1:
            self.microscopeComboBox.setCurrentIndex(index)

        microscopeLayout = QHBoxLayout()
        microscopeLayout.addWidget(self.microscopeLabel)
        microscopeLayout.setSpacing(23)
        microscopeLayout.addWidget(self.microscopeComboBox)

        microscopeWidget = QWidget()
        microscopeWidget.setLayout(microscopeLayout)

        # --- Combine Output Folder + Microscope Selection ---
        outputAndMicroscopeLayout = QHBoxLayout()
        outputAndMicroscopeLayout.addWidget(outputFilePathWidget, stretch=3)
        outputAndMicroscopeLayout.addWidget(microscopeWidget, stretch=1)

        # --- Prediction Button, Progress Bar, and Progress Label (Grouped) ---
        self.predictionButton = self.createButton("Run Prediction", self.predictionButtonPressed)
        self.predictionButton.setFixedSize(140, 36)

        self.progressBarLabel = QLabel()
        self.progressBarLabel.hide()
        self.progressBar = QProgressBar()
        self.progressBar.setFixedHeight(8)
        self.progressBar.setFixedWidth(300)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.hide()

        predictionRunWidget = QWidget()
        predictionRunLayout = QHBoxLayout(predictionRunWidget)
        predictionRunLayout.addWidget(self.predictionButton, alignment=Qt.AlignCenter)
        progressBarHLayout = QHBoxLayout()
        predictionRunLayout.addStretch()
        progressBarHLayout.addWidget(self.progressBarLabel)
        progressBarHLayout.addWidget(self.progressBar, alignment=Qt.AlignCenter)
        predictionRunLayout.addLayout(progressBarHLayout)
        predictionRunLayout.addStretch()

        # --- Progress Output Label and Text Area ---
        progressLabel = self.createLabel("Progress Output:")
        self.progressPlainTextEdit = self.createPlainTextEdit()

        predictionLogWidget = QWidget()
        predictionLogLayout = QVBoxLayout(predictionLogWidget)
        predictionLogLayout.addWidget(progressLabel)
        predictionLogLayout.addWidget(self.progressPlainTextEdit)

        # --- Main Prediction Tab Layout ---
        prediction_tab_layout = QGridLayout(self.prediction_tab)

        current_row = 0
        prediction_tab_layout.addWidget(inputFileSectionWidget, current_row, 0, 1, 1)
        prediction_tab_layout.addWidget(imagePreviewContainerWidget, current_row, 1, 1, 2)
        prediction_tab_layout.addWidget(maskControlsVWidget, current_row, 3, 1, 1, Qt.AlignRight | Qt.AlignTop)

        current_row += 1
        prediction_tab_layout.addLayout(outputAndMicroscopeLayout, current_row, 0, 1, 3)

        current_row += 1
        prediction_tab_layout.addWidget(predictionRunWidget, current_row, 0, 1, 2)

        current_row += 1
        prediction_tab_layout.addWidget(predictionLogWidget, current_row, 0, 1, 4)

    # sets up the layout and widgets for the re-training tab
    def setupRetrainTab(self):
        # --- Input File Section ---
        retrainInputFileLabel = self.createLabel("Input folder:")
        self.retrainInputDirButton = self.createButton("Browse", self.loadRetrainInputDirectory)
        self.retrainSelectAllButton = self.createButton("Select All", self.selectAllRetrainFiles)
        self.retrainDeselectAllButton = self.createButton("Deselect All", self.deselectAllRetrainFiles)

        retrainInputFilePathButtonsLayout = QHBoxLayout()
        retrainInputFilePathButtonsLayout.addWidget(self.retrainInputDirButton)
        retrainInputFilePathButtonsLayout.addWidget(self.retrainSelectAllButton)
        retrainInputFilePathButtonsLayout.addWidget(self.retrainDeselectAllButton)
        retrainInputFilePathButtonsLayout.addStretch()

        self.retrainInputFileListWidget = QListWidget()
        self.retrainInputFileListWidget.setFixedHeight(600)
        self.retrainInputFileListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.retrainInputFileListWidget.itemSelectionChanged.connect(self.updateRetrainPreviewList)
        self.retrainInputFileListWidget.itemClicked.connect(self.showRetrainImageOnItemClick)

        retrainInputFileSectionLayout = QVBoxLayout()
        retrainInputFileSectionLayout.addWidget(retrainInputFileLabel)
        retrainInputFileSectionLayout.addWidget(self.retrainInputFileListWidget)
        retrainInputFileSectionLayout.addLayout(retrainInputFilePathButtonsLayout)

        retrainInputFileSectionWidget = QWidget()
        retrainInputFileSectionWidget.setLayout(retrainInputFileSectionLayout)

        # --- Image Preview Section ---
        self.retrainImagePreviewLabel = self.createLabel("Image Preview")
        self.retrainImagePreviewLabel.setFixedHeight(600)
        self.retrainImagePreviewLabel.setMinimumWidth(800)
        self.retrainImagePreviewLabel.setAlignment(Qt.AlignCenter)
        self.retrainImagePreviewLabel.setStyleSheet("""
            border: 1px solid #cccccc;
            background-color: #fafafa;
            border-radius: 6px;
        """)

        self.retrainImageFilenameLabel = self.createLabel("No file selected")
        self.retrainImageFilenameLabel.setAlignment(Qt.AlignCenter)

        self.retrainPrevButton = self.createButton("Previous", self.showPreviousRetrainImage)
        self.retrainNextButton = self.createButton("Next", self.showNextRetrainImage)

        retrainPrevNextButtonsLayout = QHBoxLayout()
        retrainPrevNextButtonsLayout.addStretch()
        retrainPrevNextButtonsLayout.addWidget(self.retrainPrevButton)
        retrainPrevNextButtonsLayout.addWidget(self.retrainNextButton)
        retrainPrevNextButtonsLayout.addStretch()

        retrainImagePreviewLayout = QVBoxLayout()
        retrainImagePreviewLayout.addWidget(self.retrainImageFilenameLabel)
        retrainImagePreviewLayout.addWidget(self.retrainImagePreviewLabel)
        retrainImagePreviewLayout.addLayout(retrainPrevNextButtonsLayout)

        retrainImagePreviewWidget = QWidget()
        retrainImagePreviewWidget.setLayout(retrainImagePreviewLayout)

        # --- Mask Buttons + Placeholder for Display Options ---
        self.retrainApplyMaskButton = self.createButton("Apply Mask", self.applyRetrainMask)
        self.retrainRemoveMaskButton = self.createButton("Remove Mask", self.removeRetrainMask)

        retrainMaskButtonsLayout = QHBoxLayout()
        retrainMaskButtonsLayout.addWidget(self.retrainApplyMaskButton)
        retrainMaskButtonsLayout.addWidget(self.retrainRemoveMaskButton)
        retrainMaskButtonsLayout.addStretch()

        retrainMaskControlsLayout = QVBoxLayout()
        retrainMaskControlsLayout.addSpacing(21)
        retrainMaskControlsLayout.addLayout(retrainMaskButtonsLayout)

        mask_options_group_box = self.setupMaskDisplayOptions("retrain")

        retrainMaskControlsLayout.addWidget(mask_options_group_box)
        retrainMaskControlsWidget = QWidget()
        retrainMaskControlsWidget.setLayout(retrainMaskControlsLayout)

        # --- Output Folder Section ---
        self.retrainOutputFileLabel = self.createLabel("Output folder:")
        self.retrainOutputFilePathTextEdit = self.createTextEdit()
        self.retrainOutputFilePathSelectButton = self.createButton("Browse", self.retrainOutputDirSelect)

        retrainOutputLayout = QHBoxLayout()
        retrainOutputLayout.addWidget(self.retrainOutputFileLabel)
        retrainOutputLayout.addWidget(self.retrainOutputFilePathTextEdit)
        retrainOutputLayout.addWidget(self.retrainOutputFilePathSelectButton)

        retrainOutputWidget = QWidget()
        retrainOutputWidget.setLayout(retrainOutputLayout)

        # --- Re-training Run Button + Progress Bar ---
        self.retrainButton = self.createButton("Run Re-training", self.retrainButtonPressed)
        self.retrainButton.setFixedSize(140, 36)

        self.retrainProgressBarLabel = QLabel()
        self.retrainProgressBarLabel.hide()
        self.retrainProgressBar = QProgressBar()
        self.retrainProgressBar.setFixedHeight(8)
        self.retrainProgressBar.setFixedWidth(300)
        self.retrainProgressBar.setRange(0, 100)
        self.retrainProgressBar.hide()

        retrainRunLayout = QHBoxLayout()
        retrainRunLayout.addWidget(self.retrainButton)
        retrainRunLayout.addWidget(self.retrainProgressBarLabel)
        retrainRunLayout.addWidget(self.retrainProgressBar)
        retrainRunLayout.addStretch()

        retrainRunWidget = QWidget()
        retrainRunWidget.setLayout(retrainRunLayout)

        # --- Progress Log Section ---
        retrainProgressLabel = self.createLabel("Progress Output:")
        self.retrainProgressPlainTextEdit = self.createPlainTextEdit()

        retrainLogLayout = QVBoxLayout()
        retrainLogLayout.addWidget(retrainProgressLabel)
        retrainLogLayout.addWidget(self.retrainProgressPlainTextEdit)

        retrainLogWidget = QWidget()
        retrainLogWidget.setLayout(retrainLogLayout)

        # --- Final Tab Layout ---
        layout = QGridLayout(self.retrain_tab)

        row = 0
        layout.addWidget(retrainInputFileSectionWidget, row, 0)
        layout.addWidget(retrainImagePreviewWidget, row, 1, 1, 2)
        layout.addWidget(retrainMaskControlsWidget, row, 3, Qt.AlignTop | Qt.AlignRight)

        row += 1
        layout.addWidget(retrainOutputWidget, row, 0, 1, 3)

        row += 1
        layout.addWidget(retrainRunWidget, row, 0, 1, 2)

        row += 1
        layout.addWidget(retrainLogWidget, row, 0, 1, 4)

    # creates and configures the QGroupBox for mask display options
    def setupMaskDisplayOptions(self, tab_type):
        group_box = QGroupBox("Mask Display Options:")
        layout = QVBoxLayout()

        # initialize separate state variables
        if tab_type == 'prediction':
            self.prediction_show_gradient = self.settings.value("showGradientPrediction", "false") == "true"
            self.prediction_show_filled = self.settings.value("showFilledPrediction", "false") == "true"
            self.prediction_show_contour = self.settings.value("showContourPrediction", "false") == "true"
            self.prediction_gradient_colormap = self.settings.value("gradientColormapPrediction", "jet")
            self.prediction_filled_color = QColor(self.settings.value("filledColorPrediction", "#0078d7"))
            self.prediction_contour_color = QColor(self.settings.value("contourColorPrediction", "#a6d8fa"))

            gradient_checkbox = self.prediction_gradient_checkbox = QCheckBox("Gradient")
            filled_checkbox = self.prediction_filled_checkbox = QCheckBox("Filled")
            contour_checkbox = self.prediction_contour_checkbox = QCheckBox("Contour")

        elif tab_type == 'retrain':
            self.retrain_show_gradient = self.settings.value("showGradientRetrain", "false") == "true"
            self.retrain_show_filled = self.settings.value("showFilledRetrain", "false") == "true"
            self.retrain_show_contour = self.settings.value("showContourRetrain", "false") == "true"
            self.retrain_gradient_colormap = self.settings.value("gradientColormapRetrain", "jet")
            self.retrain_filled_color = QColor(self.settings.value("filledColorRetrain", "#0078d7"))
            self.retrain_contour_color = QColor(self.settings.value("contourColorRetrain", "#a6d8fa"))

            gradient_checkbox = self.retrain_gradient_checkbox = QCheckBox("Gradient")
            filled_checkbox = self.retrain_filled_checkbox = QCheckBox("Filled")
            contour_checkbox = self.retrain_contour_checkbox = QCheckBox("Contour")

        # Set initial states
        gradient_checkbox.setChecked(getattr(self, f"{tab_type}_show_gradient"))
        filled_checkbox.setChecked(getattr(self, f"{tab_type}_show_filled"))
        contour_checkbox.setChecked(getattr(self, f"{tab_type}_show_contour"))

        # gradient colormap
        colormap_combo = QComboBox()
        colormaps = ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'bone', 'copper', 'binary']
        colormap_combo.addItems(colormaps)
        colormap_combo.setFixedSize(100, 25)
        selected_cmap = getattr(self, f"{tab_type}_gradient_colormap")
        index = colormap_combo.findText(selected_cmap)
        if index != -1:
            colormap_combo.setCurrentIndex(index)

        if tab_type == 'prediction':
            self.prediction_gradient_colormap_combo = colormap_combo
            colormap_combo.currentIndexChanged.connect(lambda: self.setGradientColormap("prediction"))
        else:
            self.retrain_gradient_colormap_combo = colormap_combo
            colormap_combo.currentIndexChanged.connect(lambda: self.setGradientColormap("retrain"))

        gradient_layout = QHBoxLayout()
        gradient_layout.addWidget(gradient_checkbox)
        gradient_layout.addWidget(colormap_combo)
        layout.addLayout(gradient_layout)

        # filled color
        filled_color_button = QPushButton("Select Color")
        filled_color_button.setFixedSize(100, 25)
        color = getattr(self, f"{tab_type}_filled_color")
        filled_color_button.setStyleSheet(f"background-color: {color.name(QColor.HexArgb)};")

        if tab_type == 'prediction':
            self.prediction_filled_color_button = filled_color_button
            filled_color_button.clicked.connect(lambda: self.pickFilledColor("prediction"))
        else:
            self.retrain_filled_color_button = filled_color_button
            filled_color_button.clicked.connect(lambda: self.pickFilledColor("retrain"))

        filled_layout = QHBoxLayout()
        filled_layout.addWidget(filled_checkbox)
        filled_layout.addWidget(filled_color_button)
        layout.addLayout(filled_layout)

        # contour color
        contour_color_button = QPushButton("Select Color")
        contour_color_button.setFixedSize(100, 25)
        color = getattr(self, f"{tab_type}_contour_color")
        contour_color_button.setStyleSheet(f"background-color: {color.name()};")

        if tab_type == 'prediction':
            self.prediction_contour_color_button = contour_color_button
            contour_color_button.clicked.connect(lambda: self.pickContourColor("prediction"))
        else:
            self.retrain_contour_color_button = contour_color_button
            contour_color_button.clicked.connect(lambda: self.pickContourColor("retrain"))

        contour_layout = QHBoxLayout()
        contour_layout.addWidget(contour_checkbox)
        contour_layout.addWidget(contour_color_button)
        layout.addLayout(contour_layout)

        # store layout and state
        gradient_checkbox.stateChanged.connect(lambda: (self.setMaskStyle(tab_type), self.enableWidgets(True)))
        filled_checkbox.stateChanged.connect(lambda: (self.setMaskStyle(tab_type), self.enableWidgets(True)))
        contour_checkbox.stateChanged.connect(lambda: (self.setMaskStyle(tab_type), self.enableWidgets(True)))

        group_box.setLayout(layout)
        return group_box

    # initializes the state of GUI elements
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

        self.applyMaskButton.setEnabled(False)
        self.removeMaskButton.setEnabled(False)
        self.retrainApplyMaskButton.setEnabled(False)
        self.retrainRemoveMaskButton.setEnabled(False)

        # update prediction color buttons
        self.prediction_contour_color_button.setStyleSheet(f"background-color: {self.prediction_contour_color.name()};")
        self.prediction_filled_color_button.setStyleSheet(
            f"background-color: {self.prediction_filled_color.name(QColor.HexArgb)};")

        # update retrain color buttons
        self.retrain_contour_color_button.setStyleSheet(f"background-color: {self.retrain_contour_color.name()};")
        self.retrain_filled_color_button.setStyleSheet(
            f"background-color: {self.retrain_filled_color.name(QColor.HexArgb)};")

        # apply initial mask styles separately
        self.setMaskStyle("prediction")
        self.setMaskStyle("retrain")


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
        self.openRoverButton.setEnabled(enable)

        # prediction tab mask widgets
        self.prediction_gradient_checkbox.setEnabled(enable)
        self.prediction_filled_checkbox.setEnabled(enable)
        self.prediction_contour_checkbox.setEnabled(enable)

        self.prediction_gradient_colormap_combo.setEnabled(enable and self.prediction_gradient_checkbox.isChecked())
        self.prediction_filled_color_button.setEnabled(enable and self.prediction_filled_checkbox.isChecked())
        self.prediction_contour_color_button.setEnabled(enable and self.prediction_contour_checkbox.isChecked())

        # re-training tab widgets
        self.retrainInputFileListWidget.setEnabled(enable)
        self.retrainOutputFilePathTextEdit.setEnabled(enable)
        self.retrainInputDirButton.setEnabled(enable)
        self.retrainOutputFilePathSelectButton.setEnabled(enable)
        self.retrainSelectAllButton.setEnabled(enable)
        self.retrainDeselectAllButton.setEnabled(enable)
        self.retrainPrevButton.setEnabled(enable)
        self.retrainNextButton.setEnabled(enable)

        # retrain tab mask widgets
        self.retrain_gradient_checkbox.setEnabled(enable)
        self.retrain_filled_checkbox.setEnabled(enable)
        self.retrain_contour_checkbox.setEnabled(enable)

        self.retrain_gradient_colormap_combo.setEnabled(enable and self.retrain_gradient_checkbox.isChecked())
        self.retrain_filled_color_button.setEnabled(enable and self.retrain_filled_checkbox.isChecked())
        self.retrain_contour_color_button.setEnabled(enable and self.retrain_contour_checkbox.isChecked())

    # helper to create a QPushButton
    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    # helper to create a QLabel
    def createLabel(self, text):
        label = QLabel(text)
        return label

    # helper to create a QComboBox
    def createComboBox(self):
        comboBox = QComboBox()
        return comboBox

    # helper to create a QPlaintextEdit
    def createPlainTextEdit(self):
        textEdit = QPlainTextEdit()
        textEdit.setReadOnly(True)
        return textEdit

    # helper to create a QLineEdit
    def createTextEdit(self):
        lineEdit = QLineEdit()
        lineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return lineEdit

    # updates the mask style based on checkbox selection
    def setMaskStyle(self, tab_type):
        show_gradient = getattr(self, f"{tab_type}_gradient_checkbox").isChecked()
        show_filled = getattr(self, f"{tab_type}_filled_checkbox").isChecked()
        show_contour = getattr(self, f"{tab_type}_contour_checkbox").isChecked()

        setattr(self, f"{tab_type}_show_gradient", show_gradient)
        setattr(self, f"{tab_type}_show_filled", show_filled)
        setattr(self, f"{tab_type}_show_contour", show_contour)

        self.settings.setValue(f"showGradient{tab_type.capitalize()}", str(show_gradient).lower())
        self.settings.setValue(f"showFilled{tab_type.capitalize()}", str(show_filled).lower())
        self.settings.setValue(f"showContour{tab_type.capitalize()}", str(show_contour).lower())

        if tab_type == "prediction":
            self.applyPredictionMask()
        else:
            self.applyRetrainMask()

    # updates the selected colormap for gradient overlay
    def setGradientColormap(self, tab_type):
        colormap_combo = getattr(self, f"{tab_type}_gradient_colormap_combo")
        cmap = colormap_combo.currentText()
        setattr(self, f"{tab_type}_gradient_colormap", cmap)
        self.settings.setValue(f"gradientColormap{tab_type.capitalize()}", cmap)

        if getattr(self, f"{tab_type}_gradient_checkbox").isChecked():
            if tab_type == "prediction":
                self.applyPredictionMask()
            else:
                self.applyRetrainMask()

    # opens a color dialog to pick the contour color
    def pickContourColor(self, tab_type):
        current_color = getattr(self, f"{tab_type}_contour_color")
        color = QColorDialog.getColor(current_color, self, "Select Contour Color")
        if color.isValid():
            setattr(self, f"{tab_type}_contour_color", color)
            self.settings.setValue(f"contourColor{tab_type.capitalize()}", color.name())

            button = getattr(self, f"{tab_type}_contour_color_button")
            button.setStyleSheet(f"background-color: {color.name()};")

            if getattr(self, f"{tab_type}_contour_checkbox").isChecked():
                if tab_type == "prediction":
                    self.applyPredictionMask()
                else:
                    self.applyRetrainMask()

    # opens a color dialog to pick the filled mask color
    def pickFilledColor(self, tab_type):
        current_color = getattr(self, f"{tab_type}_filled_color")
        color = QColorDialog.getColor(current_color, self, "Select Fill Color", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            setattr(self, f"{tab_type}_filled_color", color)
            self.settings.setValue(f"filledColor{tab_type.capitalize()}", color.name(QColor.HexArgb))

            button = getattr(self, f"{tab_type}_filled_color_button")
            button.setStyleSheet(f"background-color: {color.name(QColor.HexArgb)};")

            if getattr(self, f"{tab_type}_filled_checkbox").isChecked():
                if tab_type == "prediction":
                    self.applyPredictionMask()
                else:
                    self.applyRetrainMask()

    # opens a dialog to select the output directory for predictions
    def outputDirSelect(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select a prediction output folder:')
        if dir != "":
            self.outputFilePathTextEdit.clear()
            self.outputFilePathTextEdit.insert(dir)
            self.settings.setValue("lastOutputDir", dir)

            self.updateOutputBasenames()

            if self.inputFileListWidget.count() > 0:
                self.markAnalyzedFiles()

    # opens a dialog to select the output directory for re-training
    def retrainOutputDirSelect(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select a re-training output folder:')
        if dir != "":
            self.retrainOutputFilePathTextEdit.clear()
            self.retrainOutputFilePathTextEdit.insert(dir)
            self.settings.setValue("lastRetrainOutputDir", dir)

 # updates the set of basenames for analyzed files in the output directory
    def updateOutputBasenames(self):
        self.outputBasenames = set()
        output_dir = self.outputFilePathTextEdit.text().strip()
        if os.path.isdir(output_dir):
            self.outputBasenames = {
                os.path.splitext(f)[0]
                for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
            }

    # marks files in the input list that have already been analyzed
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
                item.setForeground(QBrush(QColor("#A9A9A9")))
            else:
                item.setText(original_text)
                item.setForeground(Qt.black)

    # removes the '[✓]' suffix from a filename string
    def cleanFilename(self, text):
        return text.split(" [")[0].strip()

    # opens a dialog to select the input directory for predictions
    def loadInputDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Prediction Input Folder")
        if dir_path:
            self.settings.setValue("lastInputDir", dir_path)
            self.loadFilesFromDirectory(dir_path)

    # opens a dialog to select the input directory for re-training
    def loadRetrainInputDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Re-training Input Folder")
        if dir_path:
            self.settings.setValue("lastRetrainInputDir", dir_path)
            self.loadRetrainFilesFromDirectory(dir_path)

    # loads image files from a specified directory for prediction
    def loadFilesFromDirectory(self, dir_path):
        self.update_progress_text_signal.emit(
            f"Scanning directory for prediction files: {dir_path}.\n")
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

    # loads .v and .rdf files from a specified directory for re-training
    def loadRetrainFilesFromDirectory(self, dir_path):
        self.update_retrain_progress_text_signal.emit(f"Scanning directory for re-training files: {dir_path}.\n")
        self.retrainInputFileListWidget.clear()
        self.retrainPreviewList = []

        self.processingRunning = True
        self.enableWidgets(False)
        self.predictionButton.setEnabled(False)

        self.retrainFileLoader = FileLoaderWorker(dir_path, (".v", ".rdf"))
        self.retrainFileLoader.filesLoaded.connect(self.onRetrainFilesLoaded)
        self.retrainFileLoader.errorOccurred.connect(self.showRetrainProgressMessage)
        self.retrainFileLoader.finished.connect(self.retrainFileLoadingFinished)
        self.retrainFileLoader.start()

    # callback when prediction files are loaded by the worker
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

        self.update_progress_text_signal.emit(
            f"Found {len(file_list)} compatible prediction files.\n")

    # callback when re-training files are loaded by the worker
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

    # callback for file loading errors
    def onFileLoadError(self, error_message):
        self.update_progress_text_signal.emit(f"[ERROR] Error loading prediction files: {error_message}.\n")
        self.imagePreviewLabel.setText("Failed to load images")

    # callback when prediction file loading is finished
    def predictionFileLoadingFinished(self):
        self.processingRunning = False
        self.setProgressBarText()
        self.enableWidgets(True)
        self.predictionButton.setEnabled(True)
        self.updateOutputBasenames()
        self.markAnalyzedFiles()

    # callback when re-training file loading is finished
    def retrainFileLoadingFinished(self):
        self.processingRunning = False
        self.enableWidgets(True)
        self.predictionButton.setEnabled(True)

    # updates the prediction preview list based on selected items
    def updatePreviewList(self):
        items = self.inputFileListWidget.selectedItems()
        self.originalPredictionImage = None
        self.imagePreviewLabel.clear()
        self.imagePreviewLabel.setText("Image Preview")
        self.imageFilenameLabel.setText("No file selected")

        if items:
            self.previewList = [self.cleanFilename(item.text()) for item in items]
            self.previewIndex = 0
            self.showImageAtIndex(self.previewIndex)
        else:
            self.previewList = []
            self.applyMaskButton.setEnabled(False)
            self.removeMaskButton.setEnabled(False)

    # updates the re-training preview list based on selected items
    def updateRetrainPreviewList(self):
        items = self.retrainInputFileListWidget.selectedItems()
        self.originalRetrainImage = None
        self.retrainImagePreviewLabel.clear()
        self.retrainImagePreviewLabel.setText("Image Preview")
        self.retrainImageFilenameLabel.setText("No file selected")

        if items:
            self.retrainPreviewList = [self.cleanFilename(item.text()) for item in items]
            self.retrainPreviewIndex = 0
            self.showRetrainImageAtIndex(self.retrainPreviewIndex)
        else:
            self.retrainPreviewList = []
            self.retrainApplyMaskButton.setEnabled(False)
            self.retrainRemoveMaskButton.setEnabled(False)

    # displays the selected image in the prediction preview
    def showImageOnItemClick(self, item):
        filename = self.cleanFilename(item.text())
        self.previewList = [filename]
        self.previewIndex = 0
        self.showImageAtIndex(self.previewIndex)

    # displays the selected image in the re-training preview
    def showRetrainImageOnItemClick(self, item):
        filename = self.cleanFilename(item.text())
        self.retrainPreviewList = [filename]
        self.retrainPreviewIndex = 0
        self.showRetrainImageAtIndex(self.retrainPreviewIndex)

    # creates a gradient mask overlay using a specified colormap
    def create_gradient_mask_overlay(self, original_image_pil, mask_pil, colormap_name="jet", alpha=150):
        # get the colormap
        try:
            cmap = plt.get_cmap(colormap_name)
        except ValueError:
            logger.warning(f"Colormap '{colormap_name}' not found. Falling back to 'jet'.")
            cmap = plt.get_cmap('jet')

        # convert PIL images to NumPy arrays
        original_image_np = np.array(original_image_pil.convert('RGB'))
        mask_np = np.array(mask_pil.convert('L'))

        # calculate brightness from the original image (average of R, G, B channels)
        brightness = original_image_np.astype(np.float32).mean(axis=2)

        # get brightness values only for pixels under the mask
        masked_brightness_values = brightness[mask_np > 0]

        if masked_brightness_values.size > 0:
            min_masked_brightness = masked_brightness_values.min()
            max_masked_brightness = masked_brightness_values.max()

            if max_masked_brightness > min_masked_brightness:
                # normalize brightness based on the min/max of ONLY the masked pixels
                norm_brightness = (brightness - min_masked_brightness) / (max_masked_brightness - min_masked_brightness)
                norm_brightness = np.clip(norm_brightness, 0, 1)
            else:
                # if all masked pixels have the same brightness, map them to the middle of the colormap
                norm_brightness = np.full_like(brightness, 0.5)
        else:
            # if no mask pixels are active (or mask is empty), provide a default norm_brightness
            norm_brightness = np.full_like(brightness, 0.5)

        # apply the colormap to the normalized brightness values
        colored_mask = cmap(norm_brightness)[:, :, :3] * 255
        colored_mask = colored_mask.astype(np.uint8)

        # create an alpha channel for the overlay
        overlay_alpha = np.where(mask_np > 0, alpha, 0).astype(np.uint8)
        overlay_alpha = np.expand_dims(overlay_alpha, axis=2)

        # combine the colored mask with its alpha channel
        overlay_rgba = np.concatenate((colored_mask, overlay_alpha), axis=2)

        # convert original image to RGBA for alpha compositing
        original_image_rgba_np = np.array(original_image_pil.convert('RGBA'))

        # perform alpha compositing using NumPy
        # convert to float for calculation
        alpha_orig = original_image_rgba_np[:, :, 3] / 255.0
        alpha_overlay = overlay_rgba[:, :, 3] / 255.0

        # create output alpha channel
        out_alpha = alpha_overlay + alpha_orig * (1 - alpha_overlay)
        out_alpha = np.clip(out_alpha, 0, 1)

        # create output RGB channels
        out_rgb = (overlay_rgba[:, :, :3].astype(np.float32) * alpha_overlay[:, :, np.newaxis] +
                   original_image_rgba_np[:, :, :3].astype(np.float32) * alpha_orig[:, :, np.newaxis] * (1 - alpha_overlay[:, :, np.newaxis])) / (out_alpha[:, :, np.newaxis] + 1e-8) # Add small epsilon to prevent division by zero

        # convert back to uint8 and combine for final image
        combined_image_np = np.concatenate((out_rgb, (out_alpha * 255)[:, :, np.newaxis]), axis=2).astype(np.uint8)

        # convert NumPy array back to PIL Image
        return Image.fromarray(combined_image_np, 'RGBA')

    # creates an overlay with only the contours of the mask
    def create_contour_mask_overlay(self, original_image_pil, mask_pil, contour_color: QColor, thickness=2):
        original_image = np.array(original_image_pil.convert('RGB'))
        mask_np = np.array(mask_pil.convert('L'))

        # normalize mask to binary
        binary_mask = (mask_np > 0).astype(np.uint8)

        # find contours with OpenCV
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert RGB to RGBA
        overlay = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)

        # convert QColor to BGR(A)
        r, g, b, a = contour_color.getRgb()
        color = (r, g, b, a)

        # draw contours with desired thickness
        cv2.drawContours(overlay, contours, -1, color, thickness=thickness)

        return Image.fromarray(overlay)

    # creates an overlay where the mask area is filled with a semi-transparent color
    def create_filled_mask_overlay(self, original_image_pil, mask_pil, fill_color: QColor):
        original_image_rgb = original_image_pil.convert('RGB')
        mask_np = np.array(mask_pil.convert('L'))

        # Create an empty RGBA image for the overlay
        overlay = Image.new('RGBA', original_image_rgb.size, (0, 0, 0, 0))
        overlay_pixels = overlay.load()

        r, g, b, a = fill_color.getRgb()
        a = min(a, 60)

        for y in range(mask_np.shape[0]):
            for x in range(mask_np.shape[1]):
                if mask_np[y, x] > 0:
                    overlay_pixels[x, y] = (r, g, b, a)

        return Image.alpha_composite(original_image_rgb.convert('RGBA'), overlay)

    # displays the image at the given index in the prediction preview
    def showImageAtIndex(self, index):
        if not self.previewList:
            self.imagePreviewLabel.setText("No files selected")
            return

        filename = self.previewList[index]
        self.imageFilenameLabel.setText(filename)
        full_input_path = os.path.join(self.selectedInputDirectory, filename)

        # reset state
        self.applyMaskButton.setEnabled(False)
        self.removeMaskButton.setEnabled(False)
        self.originalPredictionImage = None
        self.imagePreviewLabel.setText("Loading...")

        try:
            # load and convert image
            image = Image.open(full_input_path)
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")

            self.originalPredictionImage = image

            # Check if we have a masked pixmap cached
            if filename in self.predictionMaskedPixmaps:
                pixmap = self.predictionMaskedPixmaps[filename]
            else:
                # Convert PIL to QPixmap if no mask
                qimg = QImage(image.tobytes("raw", "RGB"),
                              image.width, image.height,
                              QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

            # scale image height to 600px and compute new width
            fixed_height = 600
            aspect_ratio = pixmap.width() / pixmap.height()
            new_width = int(fixed_height * aspect_ratio)

            # resize label accordingly
            self.imagePreviewLabel.setFixedSize(new_width, fixed_height)

            # scale the pixmap
            scaled_pixmap = pixmap.scaled(new_width, fixed_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # create rounded pixmap
            rounded_pixmap = QPixmap(scaled_pixmap.size())
            rounded_pixmap.fill(Qt.transparent)

            painter = QPainter(rounded_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            radius = 8
            path.addRoundedRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height(), radius, radius)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, scaled_pixmap)
            painter.end()

            self.imagePreviewLabel.setPixmap(rounded_pixmap)
            self.imagePreviewLabel.setText("")

            # enable/disable buttons based on mask state
            if filename in self.predictionMaskedPixmaps:
                self.applyMaskButton.setEnabled(False)
                self.removeMaskButton.setEnabled(True)
            else:
                self.removeMaskButton.setEnabled(False)
                for i in range(self.inputFileListWidget.count()):
                    item = self.inputFileListWidget.item(i)
                    if self.cleanFilename(item.text()) == filename and "[✓]" in item.text():
                        self.applyMaskButton.setEnabled(True)
                        break

        except Exception as e:
            self.imagePreviewLabel.setText("Failed to load image.\n")
            self.update_progress_text_signal.emit(f"[ERROR] Error loading image '{full_input_path}': {e}.\n")
            self.originalPredictionImage = None
            self.applyMaskButton.setEnabled(False)
            self.removeMaskButton.setEnabled(False)

    # displays the image at the given index in the re-training preview
    def showRetrainImageAtIndex(self, index):
        if not self.retrainPreviewList:
            self.retrainImagePreviewLabel.setText("No files selected")
            return

        filename_with_ext = self.cleanFilename(self.retrainPreviewList[index])
        filename_base, _ = os.path.splitext(filename_with_ext)
        v_filename = filename_base + ".v"
        v_file_path = os.path.join(self.selectedRetrainInputDirectory, v_filename)

        self.retrainImageFilenameLabel.setText(filename_base)
        self.retrainImagePreviewLabel.setText("Loading...")
        self.retrainApplyMaskButton.setEnabled(False)
        self.retrainRemoveMaskButton.setEnabled(False)
        self.originalRetrainImage = None

        try:
            # ✅ Check if a masked pixmap is already cached
            if filename_base in self.retrainMaskedPixmaps:
                pixmap = self.retrainMaskedPixmaps[filename_base]

                # Resize image height to 600px and calculate width proportionally
                fixed_height = 600
                aspect_ratio = pixmap.width() / pixmap.height()
                new_width = int(fixed_height * aspect_ratio)

                self.retrainImagePreviewLabel.setFixedSize(new_width, fixed_height)

                scaled_pixmap = pixmap.scaled(new_width, fixed_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Apply rounded corners
                rounded_pixmap = QPixmap(scaled_pixmap.size())
                rounded_pixmap.fill(Qt.transparent)

                painter = QPainter(rounded_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                path = QPainterPath()
                radius = 8
                path.addRoundedRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height(), radius, radius)
                painter.setClipPath(path)
                painter.drawPixmap(0, 0, scaled_pixmap)
                painter.end()

                self.retrainImagePreviewLabel.setPixmap(rounded_pixmap)
                self.retrainImagePreviewLabel.setText("")
                self.retrainApplyMaskButton.setEnabled(False)
                self.retrainRemoveMaskButton.setEnabled(True)
                return  # ✅ Exit early if mask is shown

            # If no cached mask, load the original V file
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

            self.originalRetrainImage = original_image_pil  # Store PIL Image

            # Convert PIL to QPixmap
            qimg = QImage(self.originalRetrainImage.tobytes("raw", "RGB"),
                          self.originalRetrainImage.width,
                          self.originalRetrainImage.height,
                          QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # Resize image height to 600px and calculate width proportionally
            fixed_height = 600
            aspect_ratio = pixmap.width() / pixmap.height()
            new_width = int(fixed_height * aspect_ratio)

            self.retrainImagePreviewLabel.setFixedSize(new_width, fixed_height)

            scaled_pixmap = pixmap.scaled(new_width, fixed_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Apply rounded corners
            rounded_pixmap = QPixmap(scaled_pixmap.size())
            rounded_pixmap.fill(Qt.transparent)

            painter = QPainter(rounded_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            radius = 8
            path.addRoundedRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height(), radius, radius)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, scaled_pixmap)
            painter.end()

            self.retrainImagePreviewLabel.setPixmap(rounded_pixmap)
            self.retrainImagePreviewLabel.setText("")
            self.retrainApplyMaskButton.setEnabled(True)

        except Exception as e:
            self.retrainImagePreviewLabel.setText(f"Error loading image: {e}")
            self.update_retrain_progress_text_signal.emit(f"[ERROR] {e}\n")
            self.originalRetrainImage = None
            self.retrainApplyMaskButton.setEnabled(False)
            self.retrainRemoveMaskButton.setEnabled(False)

    # generates the CNN mask using thrass and overlays it on the original image based on the selected mask style
    def process_single_image_and_mask(self, filename, input_dir, output_dir, mask_settings, signals):
        full_input_path = os.path.join(input_dir, filename)
        base_name, _ = os.path.splitext(filename)
        v_file_path_in_output = os.path.join(output_dir, base_name + ".v")
        generated_mask_path = os.path.join(output_dir, base_name + "_cnn.v")

        # This is a blocking operation, but it's safe here because it's in a worker thread.
        original_image_pil = Image.open(full_input_path)
        original_image_pil = ImageOps.exif_transpose(original_image_pil)
        if original_image_pil.mode != "RGB":
            original_image_pil = original_image_pil.convert("RGB")

        try:
            v_file_mtime = os.path.getmtime(v_file_path_in_output)
            v_file_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(v_file_mtime))
        except Exception as e:
            signals.errorOccurred.emit(f"[ERROR] Failed to get timestamp for {v_file_path_in_output}: {e}\n")
            raise

        cache_dir = "/tmp/marai_cnn_masks_prediction"
        os.makedirs(cache_dir, exist_ok=True)
        cached_mask_name = f"{base_name}_{v_file_timestamp}_cnn.v"
        cached_mask_path = os.path.join(cache_dir, cached_mask_name)

        if os.path.exists(cached_mask_path) and os.path.getmtime(cached_mask_path) >= v_file_mtime:
            signals.progress_message.emit(f"[INFO] Using cached mask for {filename}\n")
            mask_file_to_use = cached_mask_path
        else:
            signals.progress_message.emit(f"Running thrass for {filename} mask generation...\n")
            thrass_command = ["thrass", "-t", "cnnPrepare", "-b", os.path.basename(v_file_path_in_output)]
            env = os.environ.copy()
            result = subprocess.run(thrass_command, capture_output=True, text=True, cwd=output_dir, env=env)

            signals.progress_message.emit(f"Thrass stdout:\n{result.stdout}\n")
            if result.stderr:
                signals.progress_message.emit(f"Thrass stderr:\n{result.stderr}\n")
            if result.returncode != 0:
                raise RuntimeError(f"Thrass failed with exit code {result.returncode}")

            wait_time = 0
            while not os.path.exists(generated_mask_path) and wait_time < 5:
                time.sleep(0.5)
                wait_time += 0.5
            if not os.path.exists(generated_mask_path):
                raise FileNotFoundError(f"Mask file not found: {generated_mask_path}")

            shutil.copy2(generated_mask_path, cached_mask_path)
            mask_file_to_use = cached_mask_path

            for fname in os.listdir(output_dir):
                if fname.startswith(base_name + "_cnn") and fname.endswith(".v"):
                    try:
                        os.remove(os.path.join(output_dir, fname))
                        signals.progress_message.emit(f"[INFO] Deleted Thrass output file: {fname}\n")
                    except Exception as e:
                        signals.progress_message.emit(f"[WARNING] Failed to delete {fname}: {e}\n")

        mask_data = np.squeeze(pmedio.read(mask_file_to_use).toarray())
        if mask_data.max() > mask_data.min():
            normalized_mask = ((mask_data - mask_data.min()) / (mask_data.max() - mask_data.min()) * 255).astype(
                np.uint8)
        else:
            normalized_mask = np.zeros_like(mask_data, dtype=np.uint8)

        mask_pil = Image.fromarray(normalized_mask.T).convert('L')
        mask_pil = mask_pil.rotate(180, expand=False)
        mask_pil = mask_pil.resize(original_image_pil.size, Image.Resampling.NEAREST)

        combined_image_pil = original_image_pil.convert('RGBA')

        if mask_settings['show_gradient']:
            combined_image_pil = self.create_gradient_mask_overlay(
                combined_image_pil, mask_pil, colormap_name=mask_settings['gradient_colormap'], alpha=150)

        if mask_settings['show_filled']:
            combined_image_pil = self.create_filled_mask_overlay(
                combined_image_pil, mask_pil, fill_color=mask_settings['filled_color'])

        if mask_settings['show_contour']:
            combined_image_pil = self.create_contour_mask_overlay(
                combined_image_pil, mask_pil, contour_color=mask_settings['contour_color'])

        qimg = QImage(combined_image_pil.tobytes("raw", "RGBA"),
                      combined_image_pil.width, combined_image_pil.height,
                      QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)

    # generates the CNN mask for re-training data and overlays it on the original .v image
    def process_single_image_and_mask_retrain(self, filename, input_dir, output_dir, mask_settings, signals=None):
        base_name, _ = os.path.splitext(filename)
        v_filename = base_name + ".v"
        v_file_path = os.path.join(input_dir, v_filename)
        generated_mask_path = os.path.join(input_dir, base_name + "_cnn.v")

        try:
            v_file_mtime = os.path.getmtime(v_file_path)
            v_file_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(v_file_mtime))
        except Exception as e:
            if signals:
                signals.errorOccurred.emit(f"[ERROR] Failed to get timestamp: {e}\n")
            raise

        cache_dir = "/tmp/marai_cnn_masks_retrain"
        os.makedirs(cache_dir, exist_ok=True)
        cached_mask_name = f"{base_name}_{v_file_timestamp}_cnn.v"
        cached_mask_path = os.path.join(cache_dir, cached_mask_name)

        if os.path.exists(cached_mask_path) and os.path.getmtime(cached_mask_path) >= v_file_mtime:
            if signals:
                signals.progress_message.emit(f"[INFO] Using cached mask for {filename}\n")
            mask_file_to_use = cached_mask_path
        else:
            if signals:
                signals.progress_message.emit(f"Running thrass for retrain mask generation on {filename}...\n")

            thrass_command = ["thrass", "-t", "cnnPrepare", "-b", v_filename]
            env = os.environ.copy()
            result = subprocess.run(
                thrass_command,
                capture_output=True,
                text=True,
                cwd=input_dir,
                env=env
            )

            if signals:
                signals.progress_message.emit(f"Thrass stdout:\n{result.stdout}\n")
                if result.stderr:
                    signals.progress_message.emit(f"Thrass stderr:\n{result.stderr}\n")

            if result.returncode != 0:
                raise RuntimeError(f"Thrass failed with exit code {result.returncode} for file: {v_file_path}")

            wait_time = 0
            while not os.path.exists(generated_mask_path) and wait_time < 5:
                time.sleep(0.5)
                wait_time += 0.5

            if not os.path.exists(generated_mask_path):
                raise FileNotFoundError(f"Thrass mask file not found at: {generated_mask_path}\n"
                                        f"Check RDF for {v_file_path}.")

            shutil.copy2(generated_mask_path, cached_mask_path)
            if signals:
                signals.progress_message.emit(f"[INFO] Cached mask: {cached_mask_path}\n")
            mask_file_to_use = cached_mask_path

            for fname in os.listdir(input_dir):
                if fname.startswith(base_name + "_cnn") and fname.endswith(".v"):
                    try:
                        os.remove(os.path.join(input_dir, fname))
                        if signals:
                            signals.progress_message.emit(f"[INFO] Deleted Thrass output file: {fname}\n")
                    except Exception as e:
                        if signals:
                            signals.progress_message.emit(f"[WARNING] Failed to delete {fname}: {e}\n")

        # Load and normalize mask
        mask_data = np.squeeze(pmedio.read(mask_file_to_use).toarray())
        if mask_data.max() > mask_data.min():
            normalized_mask = ((mask_data - mask_data.min()) / (mask_data.max() - mask_data.min()) * 255).astype(
                np.uint8)
        else:
            normalized_mask = np.zeros_like(mask_data, dtype=np.uint8)

        mask_pil = Image.fromarray(normalized_mask.T).convert('L')

        # Load original V file for background
        v_data = np.squeeze(pmedio.read(v_file_path).toarray())
        if v_data.max() > v_data.min():
            normalized_bg = ((v_data - v_data.min()) / (v_data.max() - v_data.min()) * 255).astype(np.uint8)
        else:
            normalized_bg = np.zeros_like(v_data, dtype=np.uint8)

        original_pil = Image.fromarray(normalized_bg.T)
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')

        combined_pil = original_pil.convert('RGBA')

        # Apply overlays
        if mask_settings['show_gradient']:
            combined_pil = self.create_gradient_mask_overlay(
                combined_pil, mask_pil, colormap_name=mask_settings['gradient_colormap'], alpha=150)

        if mask_settings['show_filled']:
            combined_pil = self.create_filled_mask_overlay(
                combined_pil, mask_pil, fill_color=mask_settings['filled_color'])

        if mask_settings['show_contour']:
            combined_pil = self.create_contour_mask_overlay(
                combined_pil, mask_pil, contour_color=mask_settings['contour_color'])

        qimg = QImage(combined_pil.tobytes("raw", "RGBA"),
                      combined_pil.width, combined_pil.height,
                      QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def onMaskingFinishedBatch(self, tab_type, filename, pixmap):
        if tab_type == "prediction":
            self.predictionMaskedPixmaps[filename] = pixmap
        else:
            filename_base = os.path.splitext(filename)[0]
            self.retrainMaskedPixmaps[filename_base] = pixmap

    def onMaskingError(self, error_message):
        self.showProgressMessage(f"[ERROR] Masking failed: {error_message}")

    def onMaskingFinishedAll(self):
        self.progressBar.hide()
        self.progressBarLabel.hide()
        self.enableWidgets(True)
        self.applyMaskButton.setEnabled(True)
        self.removeMaskButton.setEnabled(True)
        self.showProgressMessage("Batch mask application complete.")

        if self.previewList and 0 <= self.previewIndex < len(self.previewList):
            self.showImageAtIndex(self.previewIndex)

    def onRetrainMaskingFinishedAll(self):
        self.retrainProgressBar.hide()
        self.retrainProgressBarLabel.hide()
        self.enableWidgets(True)
        self.retrainApplyMaskButton.setEnabled(True)
        self.retrainRemoveMaskButton.setEnabled(True)
        self.showProgressMessage("Retrain mask batch complete.")

        if self.retrainPreviewList and 0 <= self.retrainPreviewIndex < len(self.retrainPreviewList):
            self.showRetrainImageAtIndex(self.retrainPreviewIndex)

    # applies the selected mask style to the current prediction image
    def applyPredictionMask(self):
        if not self.previewList:
            self.showProgressMessage("No images to process in the batch.")
            return

        self.setProgressBarText("Applying masks to all images...")
        self.showProgressMessage("Starting batch mask application...")
        self.enableWidgets(False)

        mask_settings = {
            'show_gradient': self.prediction_show_gradient,
            'show_filled': self.prediction_show_filled,
            'show_contour': self.prediction_show_contour,
            'gradient_colormap': self.prediction_gradient_colormap,
            'filled_color': self.prediction_filled_color,
            'contour_color': self.prediction_contour_color
        }

        self.maskWorker = MaskBatchWorker(
            app_instance=self,
            filenames=self.previewList,
            input_dir=self.selectedInputDirectory,
            output_dir=self.outputFilePathTextEdit.text().strip(),
            mask_settings=mask_settings,
            tab_type="prediction"
        )

        self.maskWorker.progress.connect(self.updateProgressBarDetailed)
        self.maskWorker.progress_message.connect(self.showProgressMessage)
        self.maskWorker.errorOccurred.connect(self.onMaskingError)
        self.maskWorker.result.connect(self.onMaskingFinishedBatch)
        self.maskWorker.finished.connect(self.onMaskingFinishedAll)

        self.maskWorker.start()

    def applyRetrainMask(self):
        if not self.retrainPreviewList:
            self.showProgressMessage("No retrain images to process.")
            return

        self.setRetrainProgressBarText("Applying retrain masks...")
        self.showProgressMessage("Starting retrain mask batch...")
        self.enableWidgets(False)

        mask_settings = {
            'show_gradient': self.retrain_show_gradient,
            'show_filled': self.retrain_show_filled,
            'show_contour': self.retrain_show_contour,
            'gradient_colormap': self.retrain_gradient_colormap,
            'filled_color': self.retrain_filled_color,
            'contour_color': self.retrain_contour_color
        }

        self.retrainMaskWorker = MaskBatchWorker(
            app_instance=self,
            filenames=self.retrainPreviewList,
            input_dir=self.selectedRetrainInputDirectory,
            output_dir=self.retrainOutputFilePathTextEdit.text().strip(),
            mask_settings=mask_settings,
            tab_type="retrain"
        )

        self.retrainMaskWorker.progress.connect(self.updateRetrainProgressBarDetailed)
        self.retrainMaskWorker.progress_message.connect(self.showProgressMessage)
        self.retrainMaskWorker.errorOccurred.connect(self.onMaskingError)
        self.retrainMaskWorker.result.connect(self.onMaskingFinishedBatch)
        self.retrainMaskWorker.finished.connect(self.onRetrainMaskingFinishedAll)

        self.retrainMaskWorker.start()

    # removes the mask overlay and displays the original prediction image
    def removePredictionMask(self):
        # clear cached masked images
        self.predictionMaskedPixmaps.clear()

        # refresh current image (to show original image instead of masked)
        self.showImageAtIndex(self.previewIndex)

    # removes the mask overlay and displays the original re-training image
    def removeRetrainMask(self):
        # clear all cached retrain masks
        self.retrainMaskedPixmaps.clear()

        # refresh current image
        self.showRetrainImageAtIndex(self.retrainPreviewIndex)

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

        selected_filenames = [self.cleanFilename(item.text()) for item in selected_items if "[✓]" in item.text()]

        if not selected_filenames:
            self.update_progress_text_signal.emit(
                "[ERROR] None of the selected files are marked as analyzed. Skipping.\n")
            return

        # call the new method with the list of files
        self.openMultipleFilesInRover(selected_filenames)

    # selects all files in the re-training input list
    def selectAllRetrainFiles(self):
        self.retrainInputFileListWidget.selectAll()
        self.updateRetrainPreviewList()

    # deselects all files in the re-training input list
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

    def setRetrainProgressBarText(self, text=None):
        if text is None:
            self.retrainProgressBarLabel.setText("")
            self.retrainProgressBar.setValue(0)
            self.retrainProgressBar.setMaximum(100)
            self.retrainProgressBar.hide()
            self.retrainProgressBarLabel.hide()
        else:
            self.retrainProgressBarLabel.setText(text)
            self.retrainProgressBar.setValue(0)
            self.retrainProgressBar.setMaximum(0)
            self.retrainProgressBar.show()
            self.retrainProgressBarLabel.show()

    def updateProgressBarDetailed(self, current_count, total_count, filename, stage_indicator):
        if total_count > 0:
            percentage = int((current_count / total_count) * 100)
            self.progressBar.setMaximum(total_count)
            self.progressBar.setValue(current_count)
            self.progressBarLabel.setText(f"{stage_indicator}: {current_count}/{total_count} ({percentage}%)")
            self.progressBar.show()
            self.progressBarLabel.show()
            self.update_progress_text_signal.emit(f"Completed {filename} ({current_count}/{total_count}).\n")
        else:
            self.setProgressBarText("Starting prediction...")

    def updateRetrainProgressBarDetailed(self, current_count, total_count, filename, stage_indicator):
        if total_count > 0:
            percentage = int((current_count / total_count) * 100)
            self.retrainProgressBar.setMaximum(total_count)
            self.retrainProgressBar.setValue(current_count)
            self.retrainProgressBarLabel.setText(f"{stage_indicator}: {current_count}/{total_count} ({percentage}%)")
            self.retrainProgressBar.show()
            self.retrainProgressBarLabel.show()
            self.update_retrain_progress_text_signal.emit(f"Completed {filename} ({current_count}/{total_count}).\n")
        else:
            self.retrainProgressBarLabel.setText("Starting re-training...")

    # switching between two states of elements for prediction and user interaction mode
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

            # find all .v files matching the base filename
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

        command = ["rover", "-R", "1"] + v_files_to_open

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

    ####################################################
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

    def retrainButtonPressed(self):
        pass

class PyMarAiThread(QtCore.QThread):
    progress_update = pyqtSignal(int, int, str, str)
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

            self.finished.emit()  # emit finished signal manually if stopped by user

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
            self.wait()  # wait for the thread's run method to complete

        def _close_pipe(self):
            # Only close if it's still open
            if self.pipe and not self.pipe.closed:
                try:
                    self.pipe.close()
                    logger.debug("[DEBUG] Emitter pipe closed.")
                except OSError as e:
                    logger.warning(f"[WARNING] Error closing Emitter pipe: {e}")

    class ProgressEmitter(QtCore.QThread):
        progress = pyqtSignal(int, int, str, str)

        def __init__(self, pipe_connection):
            super().__init__()
            self.pipe = pipe_connection
            self._running = True

        def run(self):
            while self._running:
                try:
                    if self.pipe.poll(0.1):
                        data = self.pipe.recv()
                        if isinstance(data, tuple) and len(data) == 4:
                            current, total, filename, stage_indicator = data
                            self.progress.emit(current, total, filename, stage_indicator)
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


class MaskBatchWorker(QThread):
    progress = pyqtSignal(int, int, str, str)
    progress_message = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    finished = pyqtSignal()
    result = pyqtSignal(str, str, QPixmap)

    def __init__(self, app_instance, filenames, input_dir, output_dir, mask_settings, tab_type, parent=None):
        super().__init__(parent)
        self.app = app_instance
        self.filenames = filenames
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mask_settings = mask_settings
        self.tab_type = tab_type
        self._abort = False

    def run(self):
        total = len(self.filenames)

        for i, filename in enumerate(self.filenames):
            if self._abort:
                self.message.emit("Batch mask application aborted.\n")
                break

            try:
                # Select method based on tab_type
                if self.tab_type == "prediction":
                    pixmap = self.app.process_single_image_and_mask(
                        filename, self.input_dir, self.output_dir, self.mask_settings, signals=self
                    )
                elif self.tab_type == "retrain":
                    pixmap = self.app.process_single_image_and_mask_retrain(
                        filename, self.input_dir, self.output_dir, self.mask_settings, signals=self
                    )
                else:
                    raise ValueError(f"Unknown tab_type: {self.tab_type}")

                self.result.emit(self.tab_type, filename, pixmap)
                self.progress.emit(i + 1, total, filename, "Masking")

            except Exception as e:
                self.errorOccurred.emit(f"[ERROR] Failed to mask {filename}: {str(e)}\n{traceback.format_exc()}")
                continue

        self.finished.emit()

    def abort(self):
        self._abort = True


# main function to start GUI
def main():
    app = QApplication(sys.argv)
#    app.setStyleSheet("""
#/* Global Font */
#* {
#    font-family: 'Segoe UI', 'Ubuntu', 'Arial';
#    font-size: 14px;
#}
#
#/* QPushButton */
#QPushButton {
#    background-color: #0078d7;
#    color: white;
#    border: none;
#    border-radius: 5px;
#    padding: 4px 10px;
#}
#QPushButton:hover {
#    background-color: #005fa3;
#}
#QPushButton:pressed {
#    background-color: #004e8c;
#}
#QPushButton:disabled {
#    background-color: #cccccc;
#    color: #666666;
#}
#
#/* QLabel */
#QLabel {
#    color: #2e2e2e;    font-weight: normal;
#}
#
#/* QComboBox */
#QComboBox {
#    background-color: #ffffff;
#    border: 1px solid #cccccc;
#    border-radius: 6px;
#    padding: 4px 6px;
#    selection-background-color: #0078d7;
#}
#QComboBox::drop-down {
#    border: none;
#}
#
#/* QGroupBox */
#QGroupBox {
#        border: 1px solid #cccccc;
#        border-radius: 8px;
#        margin-top: 10px;
#        background-color: #f9f9f9;
#        padding: 10px;
#    }
#    QGroupBox:title {
#        subcontrol-origin: margin;
#        left: 10px;
#        padding: 0 3px 0 3px;
#    }
#
#/* QTabWidget */
#QTabWidget::pane {
#    border: 1px solid #c2c7cb;
#    border-top: none;  
#    border-bottom-left-radius: 6px;
#    border-bottom-right-radius: 6px;
#    background: #ffffff;
#    padding: 6px;
#    margin-top: -1px; 
#}
#
#/* Tab Bar alignment */
#QTabWidget::tab-bar {
#    left: 0px;
#}
#
#/* QTabBar Tabs */
#QTabBar::tab {
#    background: #f5f5f5;
#    border: 1px solid #c2c7cb;
#    border-bottom: none;
#    border-top-left-radius: 8px;
#    border-top-right-radius: 8px;
#    min-width: 100px;
#    padding: 8px 16px;
#    margin-right: 1px;
#    font-weight: normal;
#    color: #333333;
#}
#
#/* Selected tab */
#QTabBar::tab:selected {
#    background: #ffffff;
#    border: 1px solid #c2c7cb;
#    border-bottom-color: #ffffff;
#    font-weight: 500;
#    color: #0078d7;
#}
#
#/* Hovered tab */
#QTabBar::tab:hover {
#    background: #eaeaea;
#}
#
#/* Disabled tab */
#QTabBar::tab:!enabled {
#    color: #999999;
#}
#
#/* QLineEdit (for folder path) */
#QLineEdit {
#    background-color: #ffffff;
#    border: 1px solid #ccc;
#    border-radius: 4px;
#    padding: 4px 6px;
#}
#
#/* QTextEdit & QPlainTextEdit for logs */
#QTextEdit, QPlainTextEdit {
#    background-color: #ffffff;
#    border: 1px solid #ccc;
#    border-radius: 4px;
#    padding: 6px;
#    font-family: 'Segoe UI', 'Ubuntu', 'Arial';
#    font-size: 13px;
#    color: #2d2d2d;
#}
#
#/* QListWidget */
#QListWidget {
#    background-color: #ffffff;
#    border: 1px solid #ccc;
#    border-radius: 4px;
#    padding: 2px;
#}
#QListWidget::item:selected {
#    background-color: #a6d8fa; 
#    color: black;
#}
#QListWidget::item:hover {
#    background-color: #e6f7ff; 
#}
#
#/* QProgressBar */
#QProgressBar {
#    background-color: #f0f0f0;
#    border: 1px solid #bbb;
#    border-radius: 6px;
#    text-align: center;
#    min-height: 20px;
#    font: 13px 'Segoe UI', 'Ubuntu', 'Arial', sans-serif;
#    color: #333;
#    padding: 2px;
#}
#
#QProgressBar::chunk {
#    background-color: #0078d7;
#    border-radius: 6px;
#    margin: 0px;
#}
#""")
    config = AppConfig()
    window = PyMarAiGuiApp(config)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
