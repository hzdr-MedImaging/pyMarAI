import datetime
import filecmp
import sys
import os
import subprocess
import multiprocessing
import logging

from pymarai import login

import pymarai.login
import pymarai.predict
import traceback
import shutil
import pmedio
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import re
from pymarai.__init__ import __version__

from PyQt5 import QtCore
from multiprocessing import Pipe

from pymarai.config import AppConfig

from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QTableWidgetItem,
                             QLabel, QProgressBar, QWidget, QTabWidget, QCheckBox, QPushButton, QSizePolicy, QPlainTextEdit, QTableWidget,
                             QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QGroupBox, QColorDialog, QToolTip, QSplitter,
                             QMainWindow)

from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush, QPainter, QPainterPath
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal, QThreadPool, QCoreApplication, QByteArray
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main GUI application class for Spheroids-DNN
class PyMarAiGuiApp(QMainWindow):

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
        self.utils = config.get_utils()

        self.settings = QSettings()
        self.selectedInputDirectory = self.settings.value("lastInputDir", os.getcwd())
        self.selectedRetrainInputDirectory = self.settings.value("lastRetrainInputDir", os.getcwd())
        self.lastRetrainOutputDirectory = self.settings.value("lastRetrainOutputDir", os.getcwd())

        self.hiddenOutputDir = os.path.join(self.selectedInputDirectory, f".pymarai-{os.getlogin()}")

        self.previewList = []
        self.retrainPreviewList = []
        self.previewIndex = 0
        self.retrainPreviewIndex = 0
        self.predictionThread = None
        self.retrainThread = None
        self.fileLoader = None
        self.statusWorker = None
        self.processingRunning = False
        self.outputBasenames = set()

        self.file_status = {}  # stores "good", "bad", or None

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
        #self.tab_widget.addTab(self.retrain_tab, "Re-training")

        self.tab_widget.currentChanged.connect(self.onTabChanged)

        self.setupPredictionTab()
        self.setupRetrainTab()

        self.setWindowTitle(f"pyMarAI v{__version__} – Spheroids Auto Delineation (hzdr.de)")
        self.initElements()
        self.setCentralWidget(self.tab_widget)

        # Initial window size/pos last saved.
        self.restoreGeometry(self.settings.value("geometry", QByteArray()))
        self.restoreState(self.settings.value("windowState", QByteArray()))
        self.predictionTopSplitterWidget.restoreState(self.settings.value("predictionTopSplitterWidget", QByteArray()))
        self.predictionBottomSplitterWidget.restoreState(self.settings.value("predictionBottomSplitterWidget", QByteArray()))
        self.imagePreviewLabel.setZoom(int(self.settings.value("imagePreviewLabelZoom", 1)))

    # close event to perform certain things while the main gui is being closed
    def closeEvent(self, e):
        # Write window size and position to QSettings
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("predictionTopSplitterWidget", self.predictionTopSplitterWidget.saveState())
        self.settings.setValue("predictionBottomSplitterWidget", self.predictionBottomSplitterWidget.saveState())
        self.settings.setValue("imagePreviewLabelZoom", self.imagePreviewLabel.getZoom())
        super().closeEvent(e)
        e.accept()

    # sets up the layout and widgets for the prediction tab
    def setupPredictionTab(self):
        # --- Input File Section ---
        inputFileLabel = self.createLabel("Input folder:")
        self.inputDirButton = self.createButton("Change Folder", self.loadInputDirectory)
        self.selectAllButton = self.createButton("Select All", self.selectAllFiles)
        self.deselectAllButton = self.createButton("Deselect All", self.deselectAllFiles)

        inputFilePathButtonsLayout = QHBoxLayout()
        inputFilePathButtonsLayout.addWidget(self.inputDirButton)
        inputFilePathButtonsLayout.addWidget(self.selectAllButton)
        inputFilePathButtonsLayout.addWidget(self.deselectAllButton)
        inputFilePathButtonsLayout.addStretch()

        self.inputFileListWidget = QListWidget()
        self.inputFileListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.inputFileListWidget.itemSelectionChanged.connect(self.updatePreviewList)
        self.inputFileListWidget.itemDoubleClicked.connect(self.openAnalyzedFile)

        inputFileSectionWidget = QWidget()
        inputFileSectionLayout = QVBoxLayout(inputFileSectionWidget)
        inputFileSectionLayout.addWidget(inputFileLabel)
        inputFileSectionLayout.addWidget(self.inputFileListWidget)
        inputFileSectionLayout.addLayout(inputFilePathButtonsLayout)

        # --- Image Preview Section ---
        self.imagePreviewLabel = ScaledLabel("Image Preview")
        self.imagePreviewLabel.setAlignment(Qt.AlignCenter)
        self.imagePreviewLabel.setStyleSheet("""
            border: 1px solid #cccccc;
            background-color: #fafafa;
            border-radius: 6px;
        """)
        self.imagePreviewLabel.zoom_changed_signal.connect(self.setZoomPercentageLabel)

        self.imageFilenameLabel = self.createLabel("No file selected")
        self.imageFilenameLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum);
        self.imageFilenameLabel.setAlignment(Qt.AlignCenter)
        self.imageFilenameLabel.setMaximumHeight(16)

        self.prevButton = self.createButton("↑ Previous", self.showPreviousImage)
        self.nextButton = self.createButton("Next ↓", self.showNextImage)
        self.markGoodButton = self.createButton("→ Mark as GOOD", self.markFileAsGood)
        self.markGoodButton.setToolTip(
            "Marks the currently selected file as GOOD result of prediction. This action can also be triggered by pushing right arrow ->.")
        self.markBadButton = self.createButton("Mark as BAD ←", self.markFileAsBad)
        self.markBadButton.setToolTip(
            "Marks the currently selected file as BAD result of prediction. This action can also be triggered by pushing left arrow <-.")

        imagePreviewContainerSubLayout = QVBoxLayout()
        imagePreviewContainerSubLayout.addWidget(self.imageFilenameLabel)
        imagePreviewContainerSubLayout.addWidget(self.imagePreviewLabel)

        prevNextButtonsHLayout = QHBoxLayout()
        prevNextButtonsHLayout.addStretch()
        prevNextButtonsHLayout.addWidget(self.prevButton)
        prevNextButtonsHLayout.addWidget(self.nextButton)
        prevNextButtonsHLayout.addWidget(self.markGoodButton)
        prevNextButtonsHLayout.addWidget(self.markBadButton)
        prevNextButtonsHLayout.addStretch()
        imagePreviewContainerSubLayout.addLayout(prevNextButtonsHLayout)

        # --- Mask Display Options ---
        self.refresh_mask_button = QPushButton("Refresh Mask")
        self.refresh_mask_button.clicked.connect(self.refresh_mask)

        mask_options_group_box = self.setupMaskDisplayOptions("prediction")

        self.zoom_percent_label = QLabel("100%")
        self.zoom_percent_label.setAlignment(Qt.AlignCenter)
        self.zoomInButton = self.createButton("+", self.imagePreviewLabel.zoomIn)
        self.zoomOutButton = self.createButton("-", self.imagePreviewLabel.zoomOut)

        zoom_buttons_layout = QHBoxLayout()
        zoom_buttons_layout.addWidget(self.zoomOutButton)
        zoom_buttons_layout.addWidget(self.zoomInButton)

        zoom_main_layout = QVBoxLayout()
        zoom_main_layout.addWidget(self.zoom_percent_label)
        zoom_main_layout.addLayout(zoom_buttons_layout)

        zoomLevelGroupBox = QGroupBox("Image Zoom:")
        zoomLevelGroupBox.setLayout(zoom_main_layout)

        maskControlsVLayout = QVBoxLayout()
        maskControlsVLayout.addSpacing(21)
        maskControlsVLayout.addWidget(self.refresh_mask_button)
        maskControlsVLayout.addWidget(mask_options_group_box)
        maskControlsVLayout.addWidget(zoomLevelGroupBox)
        maskControlsVLayout.addStretch()

        imagePreviewContainerWidget = QWidget()
        imagePreviewContainerLayout = QHBoxLayout(imagePreviewContainerWidget)
        imagePreviewContainerLayout.addLayout(imagePreviewContainerSubLayout)
        imagePreviewContainerLayout.addLayout(maskControlsVLayout)

        # --- Analysed Files Toolbox ---
        self.statusToolsGroup = QGroupBox("Status Tools")
        self.statusToolsGroup.setCheckable(True)
        self.statusToolsGroup.setChecked(True)

        self.selectAllGoodButton = self.createButton("Select all GOOD", self.selectAllGoodFiles)
        self.selectAllBadButton = self.createButton("Select all BAD", self.selectAllBadFiles)
        #self.selectAllUntaggedButton = self.createButton("Select All Untagged", self.selectAllUntaggedFiles)
        self.openRoverButton = self.createButton("Open in ROVER", self.openAllSelectedFilesInRover)
        self.openRoverButton.setToolTip(
            "Opens all currently selected files in the ROVER application. It can also be triggered by double-clicking the file of interest.")
        self.saveOutputButton = self.createButton("Export Images", self.saveSelectedOutputs)
        self.saveOutputButton.setToolTip(
            "Saves the selected output files (.v and .rdf) to the user-selected directory.")
        self.generateStatsButton = self.createButton("Generate CSV", self.generateStatisticsTable)
        self.generateStatsButton.setToolTip(
            "Generates a statistics table for all selected files that are marked as 'GOOD'.")

        statusToolsWidget = QWidget()
        statusToolsLayout = QHBoxLayout(statusToolsWidget)
        statusToolsLayout.addWidget(self.selectAllGoodButton)
        statusToolsLayout.addWidget(self.selectAllBadButton)
        #statusToolsLayout.addWidget(self.selectAllUntaggedButton)
        statusToolsLayout.addWidget(self.openRoverButton)
        statusToolsLayout.addWidget(self.saveOutputButton)
        statusToolsLayout.addWidget(self.generateStatsButton)
        statusToolsLayout.addStretch()

        # --- Microscope Selection ---
        self.microscopeLabel = self.createLabel("Microscope:")
        self.microscopeComboBox = self.createComboBox()
        self.microscopeComboBox.addItems(self.microscopes)
        index = self.microscopeComboBox.findText(self.defaultMicroscopeType)
        if index != -1:
            self.microscopeComboBox.setCurrentIndex(index)

        microscopeLayout = QHBoxLayout()
        microscopeLayout.addWidget(self.microscopeLabel)
        microscopeLayout.addWidget(self.microscopeComboBox)
        microscopeLayout.addStretch()

        microscopeWidget = QWidget()
        microscopeWidget.setLayout(microscopeLayout)

        # --- Prediction Button, Progress Bar, and Progress Label (Grouped) ---
        self.predictionButton = self.createButton("Run Prediction", self.predictionButtonPressed)
        self.predictionButton.setFixedSize(140, 36)

        self.progressBarLabel = QLabel()
        self.progressBarLabel.hide()
        self.progressBar = QProgressBar()
        self.progressBar.setFixedHeight(13)
        self.progressBar.setFixedWidth(300)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.hide()

        predictionRunWidget = QWidget()
        predictionRunLayout = QHBoxLayout(predictionRunWidget)
        predictionRunLayout.addWidget(self.predictionButton, alignment=Qt.AlignCenter)
        progressBarHLayout = QHBoxLayout()
        progressBarHLayout.addWidget(self.progressBarLabel)
        progressBarHLayout.addWidget(self.progressBar)
        predictionRunLayout.addLayout(progressBarHLayout)
        predictionRunLayout.addStretch()

        # --- Progress Output Label and Text Area ---
        progressLabel = self.createLabel("Progress Output:")
        self.progressPlainTextEdit = self.createPlainTextEdit()

        predictionLogWidget = QWidget()
        predictionLogLayout = QVBoxLayout(predictionLogWidget)
        predictionLogLayout.addWidget(progressLabel)
        predictionLogLayout.addWidget(self.progressPlainTextEdit)

        # Create horizontal splitter to divide top area
        self.predictionTopSplitterWidget = QSplitter(Qt.Orientation.Horizontal)
        self.predictionTopSplitterWidget.addWidget(inputFileSectionWidget)
        self.predictionTopSplitterWidget.addWidget(imagePreviewContainerWidget)

        # Create bottom layout part
        predictionControlWidget = QWidget()
        predictionControlLayout = QVBoxLayout(predictionControlWidget)
        predictionControlLayout.addWidget(statusToolsWidget)
        predictionControlLayout.addWidget(microscopeWidget)
        predictionControlLayout.addWidget(predictionRunWidget)
        predictionControlLayout.addWidget(predictionLogWidget)

        # Create horizontal splitter to divide top area
        self.predictionBottomSplitterWidget = QSplitter(Qt.Orientation.Vertical)
        self.predictionBottomSplitterWidget.setStyleSheet('QSplitter::handle {border: 2px solid lightgrey; }')
        self.predictionBottomSplitterWidget.addWidget(self.predictionTopSplitterWidget)
        self.predictionBottomSplitterWidget.addWidget(predictionControlWidget)

        # --- Main Prediction Tab Layout ---
        prediction_tab_layout = QVBoxLayout(self.prediction_tab)
        prediction_tab_layout.addWidget(self.predictionBottomSplitterWidget)

    # sets up the layout and widgets for the re-training tab
    def setupRetrainTab(self):
        # --- Input File Section ---
        retrainInputFileLabel = self.createLabel("Input folder:")
        self.retrainInputDirButton = self.createButton("Change Folder", self.loadRetrainInputDirectory)
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
        #self.retrainImagePreviewLabel.setFixedHeight(600)
        #self.retrainImagePreviewLabel.setMinimumWidth(800)
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
        self.retrainProgressBar.setFixedHeight(13)
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
            self.prediction_show_contour = self.settings.value("showContourPrediction", "true") == "true"
            self.prediction_gradient_colormap = self.settings.value("gradientColormapPrediction", "jet")
            self.prediction_filled_color = QColor(self.settings.value("filledColorPrediction", "#0078d7"))
            self.prediction_contour_color = QColor(self.settings.value("contourColorPrediction", "#aa0000"))

            gradient_checkbox = self.prediction_gradient_checkbox = QCheckBox("Gradient")
            filled_checkbox = self.prediction_filled_checkbox = QCheckBox("Filled")
            contour_checkbox = self.prediction_contour_checkbox = QCheckBox("Contour")

        elif tab_type == 'retrain':
            self.retrain_show_gradient = self.settings.value("showGradientRetrain", "false") == "true"
            self.retrain_show_filled = self.settings.value("showFilledRetrain", "false") == "true"
            self.retrain_show_contour = self.settings.value("showContourRetrain", "true") == "true"
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

        if os.path.isdir(self.lastRetrainOutputDirectory):
            self.retrainOutputFilePathTextEdit.insert(self.lastRetrainOutputDirectory)

        if os.path.isdir(self.selectedInputDirectory):
            self.loadFilesFromDirectory(self.selectedInputDirectory)

        if os.path.isdir(self.selectedRetrainInputDirectory):
            self.loadRetrainFilesFromDirectory(self.selectedRetrainInputDirectory)

        self.retrainApplyMaskButton.setEnabled(False)
        self.retrainRemoveMaskButton.setEnabled(False)
        self.markGoodButton.setEnabled(False)
        self.markBadButton.setEnabled(False)

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
        self.inputDirButton.setEnabled(enable)
        self.microscopeComboBox.setEnabled(enable)
        self.selectAllButton.setEnabled(enable)
        self.deselectAllButton.setEnabled(enable)
        self.prevButton.setEnabled(enable)
        self.nextButton.setEnabled(enable)
        self.openRoverButton.setEnabled(enable)
        self.selectAllGoodButton.setEnabled(enable)
        self.selectAllBadButton.setEnabled(enable)
        #self.selectAllUntaggedButton.setEnabled(enable)
        self.saveOutputButton.setEnabled(enable)
        self.generateStatsButton.setEnabled(enable)

        # prediction tab mask widgets
        self.prediction_gradient_checkbox.setEnabled(enable)
        self.prediction_filled_checkbox.setEnabled(enable)
        self.prediction_contour_checkbox.setEnabled(enable)

        self.zoomInButton.setEnabled(enable)
        self.zoomOutButton.setEnabled(enable)

        self.prediction_gradient_colormap_combo.setEnabled(enable and self.prediction_gradient_checkbox.isChecked())
        self.prediction_filled_color_button.setEnabled(enable and self.prediction_filled_checkbox.isChecked())
        self.prediction_contour_color_button.setEnabled(enable and self.prediction_contour_checkbox.isChecked())

        self.markGoodButton.setEnabled(enable and self.current_preview_filename is not None)
        self.markBadButton.setEnabled(enable and self.current_preview_filename is not None)

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
        output_dir = self.hiddenOutputDir
        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                fp = os.path.join(output_dir, f)
                if not os.path.isfile(fp):
                    continue
                base_no_ext = os.path.splitext(f)[0]
                base_without_status, _ = self.parseStatusFromFilename(base_no_ext)
                self.outputBasenames.add(base_without_status)

    # marks files in the input list that have already been analyzed
    def markAnalyzedFiles(self):
        output_dir = self.hiddenOutputDir
        items = [self.inputFileListWidget.item(i) for i in range(self.inputFileListWidget.count())]

        self.inputFileListWidget.setUpdatesEnabled(False)
        self.file_status.clear()

        self.statusWorker = FileStatusWorker(
            input_items=items,
            output_dir=output_dir,
            parse_status_func=self.parseStatusFromFilename,
            strip_microscope_tag_func=self.stripMicroscopeTag
        )

        def apply_batch(batch):
            for row, text, color, status_found in batch:
                item = self.inputFileListWidget.item(row)
                full_path = item.data(Qt.UserRole)
                self.file_status[full_path] = status_found
                item.setText(text)
                item.setForeground(QBrush(color))

        def finish():
            self.inputFileListWidget.setUpdatesEnabled(True)
            if self.inputFileListWidget.selectedItems():
                self.updatePreviewList()

        self.statusWorker.batch_ready.connect(apply_batch)
        self.statusWorker.finished_all.connect(finish)
        self.statusWorker.start()

    # removes the suffix from a filename string
    def cleanFilename(self, text):
        return text.split(" [")[0].strip()

    # removes _m<number> from the end of filename (before extension)
    def stripMicroscopeTag(self, filename):
        return re.sub(r"_m\d+$", "", filename)

    # update the text of the zoom_percent_label with the new percentage
    def setZoomPercentageLabel(self, zoom_factor):
        percentage = int(zoom_factor * 100)
        self.zoom_percent_label.setText(f"{percentage}%")

    # opens a dialog to select the input directory for predictions
    def loadInputDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Prediction Input Folder")
        if dir_path:
            self.settings.setValue("lastInputDir", dir_path)
            self.loadFilesFromDirectory(dir_path)
            self.hiddenOutputDir = os.path.join(dir_path, f".pymarai-{os.getlogin()}")
            self.updatePreviewList()
            self.updateOutputBasenames()

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

        self.file_status = {}

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
            full_path = os.path.join(dir_path, file)
            display_name = os.path.basename(file)
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, full_path)
            self.inputFileListWidget.addItem(item)

        self.updateOutputBasenames()
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
            f"Found {len(file_list)} compatible prediction files.\n"
        )

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
        self.update_progress_text_signal.emit(f"[ERROR] Error loading prediction files: {error_message.strip()}\n")
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
            # get the full paths of all selected items
            self.previewList = [item.data(Qt.UserRole) for item in items]
            self.previewIndex = 0

            # get the full path and filename of the first selected item for preview
            full_path = self.previewList[0]
            self.current_preview_filename = full_path

            # show the first image in the selection
            self.showImageAtIndex(self.previewIndex)
        else:
            # handle case where no files are selected
            self.previewList = []
            self.markGoodButton.setEnabled(False)
            self.markBadButton.setEnabled(False)

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

        full_input_path = self.previewList[index]
        filename = os.path.basename(full_input_path)
        self.imageFilenameLabel.setText(filename)

        self.current_preview_filename = full_input_path
        self.originalPredictionImage = None
        self.imagePreviewLabel.setText("Loading...")

        try:
            # load original image
            image = Image.open(full_input_path)
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.originalPredictionImage = image

            # check if analyzed
            status = self.file_status.get(full_input_path)
            is_analyzed = status in ("TO DO", "GOOD", "BAD")

            if is_analyzed:
                # mask overlay (auto applied)
                if full_input_path not in self.predictionMaskedPixmaps:
                    mask_settings = {
                        'show_gradient': self.prediction_show_gradient,
                        'show_filled': self.prediction_show_filled,
                        'show_contour': self.prediction_show_contour,
                        'gradient_colormap': self.prediction_gradient_colormap,
                        'filled_color': self.prediction_filled_color,
                        'contour_color': self.prediction_contour_color
                    }
                    pixmap = self.process_single_image_and_mask(
                        full_input_path,
                        self.selectedInputDirectory,
                        mask_settings,
                        signals=None
                    )
                    self.predictionMaskedPixmaps[full_input_path] = pixmap
                else:
                    pixmap = self.predictionMaskedPixmaps[full_input_path]
            else:
                # no analysis yet → show plain image
                qimg = QImage(image.tobytes("raw", "RGB"),
                              image.width, image.height,
                              QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

            # display
            self.imagePreviewLabel.setPixmap(pixmap)
            self.imagePreviewLabel.setMinimumWidth(100)
            self.imagePreviewLabel.setMinimumHeight(100)
            self.imagePreviewLabel.setText("")

            # GOOD/BAD marking enabled only if analyzed
            self.markGoodButton.setEnabled(is_analyzed)
            self.markBadButton.setEnabled(is_analyzed)

        except Exception as e:
            self.imagePreviewLabel.setText("Failed to load image.\n")
            self.update_progress_text_signal.emit(
                f"[ERROR] Error loading image '{full_input_path}': {e}.\n"
            )
            self.originalPredictionImage = None
            self.markGoodButton.setEnabled(False)
            self.markBadButton.setEnabled(False)

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
            # check if a masked pixmap is already cached
            if filename_base in self.retrainMaskedPixmaps:
                pixmap = self.retrainMaskedPixmaps[filename_base]

                # resize image height to 600px and calculate width proportionally
                fixed_height = 600
                aspect_ratio = pixmap.width() / pixmap.height()
                new_width = int(fixed_height * aspect_ratio)

                self.retrainImagePreviewLabel.setFixedSize(new_width, fixed_height)

                scaled_pixmap = pixmap.scaled(new_width, fixed_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # apply rounded corners
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
                return

            # if no cached mask, load the original V file
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

            self.originalRetrainImage = original_image_pil

            # convert PIL to QPixmap
            qimg = QImage(self.originalRetrainImage.tobytes("raw", "RGB"),
                          self.originalRetrainImage.width,
                          self.originalRetrainImage.height,
                          QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # resize image height to 600px and calculate width proportionally
            fixed_height = 600
            aspect_ratio = pixmap.width() / pixmap.height()
            new_width = int(fixed_height * aspect_ratio)

            self.retrainImagePreviewLabel.setFixedSize(new_width, fixed_height)

            scaled_pixmap = pixmap.scaled(new_width, fixed_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # apply rounded corners
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
    def process_single_image_and_mask(self, filename, input_dir, mask_settings, signals=None):

        def emit(msg):
            if signals:
                try:
                    signals.progress_message.emit(msg)
                except Exception:
                    pass

        # resolve full path
        if os.path.isabs(filename) and os.path.exists(filename):
            full_input_path = filename
        else:
            full_input_path = os.path.join(input_dir, os.path.basename(filename))

        base_name = os.path.splitext(os.path.basename(full_input_path))[0]

        # load the original image first
        original_image_pil = Image.open(full_input_path)
        original_image_pil = ImageOps.exif_transpose(original_image_pil)
        if original_image_pil.mode != "RGB":
            original_image_pil = original_image_pil.convert("RGB")

        # check for the hidden output directory
        if not os.path.isdir(self.hiddenOutputDir):
            emit(f"[ERROR] Output directory not found: {self.hiddenOutputDir}\n")
            return self._load_and_convert_image(full_input_path)

        # find all files in the directory that start with the base name and contain "_cnn"
        matching_files = [f for f in os.listdir(self.hiddenOutputDir) if f.startswith(base_name) and "_cnn" in f]

        if not matching_files:
            emit(f"[WARNING] Corresponding CNN mask file not found for {base_name}. Displaying original image.\n")
            combined_image_pil = original_image_pil.convert('RGBA')
            qimg = QImage(combined_image_pil.tobytes("raw", "RGBA"),
                          combined_image_pil.width,
                          combined_image_pil.height,
                          QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg)

        # sort by modification time to get the most recent one if multiple matches exist
        matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.hiddenOutputDir, f)), reverse=True)
        mask_file_to_use = os.path.join(self.hiddenOutputDir, matching_files[0])

        # load mask data
        mask_data = np.squeeze(pmedio.read(mask_file_to_use).toarray())
        if mask_data.max() > mask_data.min():
            normalized_mask = ((mask_data - mask_data.min()) /
                               (mask_data.max() - mask_data.min()) * 255).astype(np.uint8)
        else:
            normalized_mask = np.zeros_like(mask_data, dtype=np.uint8)

        mask_pil = Image.fromarray(normalized_mask.T).convert('L')
        mask_pil = mask_pil.rotate(180, expand=False)
        mask_pil = mask_pil.resize(original_image_pil.size, Image.Resampling.NEAREST)

        combined_image_pil = original_image_pil.convert('RGBA')

        # apply mask styles
        if mask_settings['show_gradient']:
            combined_image_pil = self.create_gradient_mask_overlay(
                combined_image_pil, mask_pil,
                colormap_name=mask_settings['gradient_colormap'],
                alpha=150
            )

        if mask_settings['show_filled']:
            combined_image_pil = self.create_filled_mask_overlay(
                combined_image_pil, mask_pil,
                fill_color=mask_settings['filled_color']
            )

        if mask_settings['show_contour']:
            combined_image_pil = self.create_contour_mask_overlay(
                combined_image_pil, mask_pil,
                contour_color=mask_settings['contour_color']
            )

        qimg = QImage(combined_image_pil.tobytes("raw", "RGBA"),
                      combined_image_pil.width, combined_image_pil.height,
                      QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)

    # generates the CNN mask for re-training data and overlays it on the original .v image
    def process_single_image_and_mask_retrain(self, filename, input_dir, mask_settings, signals=None):
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

        cache_dir = os.path.join(self.hiddenOutputDir, "cnn_masks_retrain")
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

            thrass_command = [self.utils['thrass'], "-t", "cnnPrepare", "-b", v_filename]
            result = subprocess.run(
                thrass_command,
                capture_output=True,
                text=True,
                cwd=input_dir,
                env=os.environ.copy()
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
            self.retrainMaskedPixmaps[filename] = pixmap

        try:
            finished_index = self.previewList.index(filename)
            if finished_index == self.previewIndex:
                self.showImageAtIndex(finished_index)

        except ValueError:
            pass

    def onMaskingError(self, error_message):
        self.showProgressMessage(f"[ERROR] Masking failed: {error_message}")

    def onMaskingFinishedAll(self):
        self.progressBar.hide()
        self.progressBarLabel.hide()
        self.enableWidgets(True)
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
            mask_settings=mask_settings,
            tab_type="prediction"
        )

        self.maskWorker.progress.connect(self.updateProgressBarDetailed)
        self.maskWorker.progress_message.connect(self.showProgressMessage)
        self.maskWorker.errorOccurred.connect(self.onMaskingError)
        self.maskWorker.result.connect(self.onMaskingFinishedBatch)
        self.maskWorker.finished.connect(self.onMaskingFinishedAll)

        self.maskWorker.start()
        self.updatePreviewList()

    def applyRetrainMask(self):
        if not self.retrainPreviewList:
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
        self.updatePreviewList()

    # removes the mask overlay and displays the original re-training image
    def removeRetrainMask(self):
        # clear all cached retrain masks
        self.retrainMaskedPixmaps.clear()

        # refresh current image
        self.showRetrainImageAtIndex(self.retrainPreviewIndex)

    def showNextImage(self):
        if self.previewList and len(self.previewList) > 1:
            # multi-selection behavior
            self.previewIndex = (self.previewIndex + 1) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)
        else:
            # navigate through entire list if only one file was selected
            count = self.inputFileListWidget.count()
            if count == 0:
                return
            current_row = self.inputFileListWidget.currentRow()
            next_row = (current_row + 1) % count
            self.inputFileListWidget.setCurrentRow(next_row)
            self.updatePreviewList()

    def showPreviousImage(self):
        if self.previewList and len(self.previewList) > 1:
            # multi-selection behavior
            self.previewIndex = (self.previewIndex - 1 + len(self.previewList)) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)
        else:
            # navigate through entire list if only one file was selected
            count = self.inputFileListWidget.count()
            if count == 0:
                return
            current_row = self.inputFileListWidget.currentRow()
            prev_row = (current_row - 1 + count) % count
            self.inputFileListWidget.setCurrentRow(prev_row)
            self.updatePreviewList()

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

        analyzed_filenames = []
        for item in selected_items:
            full_path = item.data(Qt.UserRole)
            # check if the file has an analyzed status in the file_status dictionary
            status = self.file_status.get(full_path)
            if status in ("TO DO", "GOOD", "BAD"):
                analyzed_filenames.append(full_path)

        if not analyzed_filenames:
            self.update_progress_text_signal.emit(
                "[ERROR] None of the selected files are marked as analyzed. Skipping.\n")
            return

        # Now, call the function to open the files
        self.openMultipleFilesInRover(analyzed_filenames)

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

        if "[TO DO]" not in text:
            self.update_progress_text_signal.emit(
                f"[ERROR] '{filename}' is not marked as analyzed. Skipping open with ROVER.\n")
            return

        self.openMultipleFilesInRover([filename])

    def openMultipleFilesInRover(self, selected_filenames):
        output_dir = self.hiddenOutputDir
        if not os.path.isdir(output_dir):
            self.update_progress_text_signal.emit("[ERROR] Output directory is not valid.\n")
            QMessageBox.warning(self, "Open in ROVER", "Please select a valid output directory.")
            return

        # create corrections folder
        corrections_dir = os.path.join(output_dir, "corrections")
        os.makedirs(corrections_dir, exist_ok=True)

        output_files = os.listdir(output_dir)
        files_to_open = []

        for filename in selected_filenames:
            base, _ = os.path.splitext(os.path.basename(filename))
            clean_base = base.replace("_GOOD", "").replace("_BAD", "")

            matching_files = [
                f for f in output_files
                if base in f and (f.endswith(".v") or f.endswith(".rdf")) and "_cnn" not in f
            ]

            if not matching_files:
                self.update_progress_text_signal.emit(
                    f"[ERROR] No valid .v or .rdf file found for '{filename}' (ignoring _cnn). Skipping.\n")
                continue

            for f in matching_files:
                src_path = os.path.join(output_dir, f)
                ext = os.path.splitext(f)[1]
                dest_name = f.replace("_GOOD", "").replace("_BAD", "")
                dest_path = os.path.join(corrections_dir, dest_name)
                rel_src_path = os.path.relpath(src_path, start=corrections_dir)

                try:
                    if ext == ".rdf":
                        shutil.copy2(src_path, dest_path)
                    elif ext == ".v":
                        if not os.path.exists(dest_path):
                            os.symlink(rel_src_path, dest_path)
                        # only .v files go to ROVER
                        files_to_open.append(dest_path)
                except Exception as e:
                    self.update_progress_text_signal.emit(f"[ERROR] Failed to copy/symlink '{f}': {e}\n")
                    continue

        if not files_to_open:
            self.update_progress_text_signal.emit("[ERROR] No valid .v files found to open in ROVER.\n")
            QMessageBox.warning(self, "Open in ROVER", "No valid .v files were found to open.")
            return

        command = [self.utils['rover'], "-R", "1"] + files_to_open

        self.update_progress_text_signal.emit(
            f"Opening {len(files_to_open)} .v file(s) in ROVER (from corrections folder):\n"
            + "\n".join(files_to_open) + "\n"
        )

        try:
            subprocess.Popen(command)
        except Exception as e:
            self.update_progress_text_signal.emit(f"[ERROR] Failed to open ROVER: {e}\n")
            QMessageBox.warning(
                self, "Error Opening ROVER",
                f"Could not open ROVER for selected files: {e}\nPlease ensure ROVER is installed and in your system PATH."
            )

    def saveSelectedOutputs(self):
        selected_items = self.inputFileListWidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No files selected", "Please select one or more files to save.")
            return

        # check for analyzed files before opening the save dialog
        files_not_analyzed = []
        analyzed_files_info = []

        for item in selected_items:
            full_path = item.data(Qt.UserRole)
            if full_path in self.file_status:
                base_filename = os.path.splitext(os.path.basename(full_path))[0]
                analyzed_files_info.append({
                    'full_path': full_path,
                    'base_filename': base_filename
                })
            else:
                files_not_analyzed.append(os.path.basename(full_path))

        if not analyzed_files_info:
            if files_not_analyzed:
                QMessageBox.information(self, "No Files to Save", "None of the selected files have been analyzed yet.")
            return

        destination_dir = QFileDialog.getExistingDirectory(self, "Select Destination Directory", os.getcwd())
        if not destination_dir:
            return

        source_dir = self.hiddenOutputDir
        all_output_files = os.listdir(source_dir)

        files_copied = 0
        files_failed_to_copy = []

        # perform the copy operation for the analyzed files
        for file_info in analyzed_files_info:
            base_filename = file_info['base_filename']

            # find all matching files for the current base_filename,
            # ignoring any files containing '_cnn' in their name
            matching_output_files = [
                f for f in all_output_files
                if f.startswith(base_filename) and (f.endswith('.v') or f.endswith('.rdf')) and '_cnn' not in f
            ]

            if not matching_output_files:
                continue

            for output_file in matching_output_files:
                source_path = os.path.join(source_dir, output_file)

                # determine the new, clean filename
                file_extension = os.path.splitext(output_file)[1]
                new_filename = f"{base_filename}{file_extension}"
                destination_path = os.path.join(destination_dir, new_filename)

                try:
                    shutil.copy2(source_path, destination_path)
                    files_copied += 1
                except Exception as e:
                    logger.error(f"Failed to copy {source_path}: {e}")
                    files_failed_to_copy.append(output_file)

        # display summary messages
        final_message = []
        if files_copied > 0:
            final_message.append(f"Successfully copied {files_copied} output files to:\n{destination_dir}")
        if files_not_analyzed:
            final_message.append("Skipped the following files as they have not been analyzed:")
            final_message.extend([f"• {f}" for f in files_not_analyzed])
        if files_failed_to_copy:
            final_message.append("Failed to copy the following files due to errors:")
            final_message.extend([f"• {f}" for f in files_failed_to_copy])

        if final_message:
            QMessageBox.information(self, "Copying complete", "\n\n".join(final_message))

    # generates a statistics table using the 'thrass' command for all selected files that have a 'GOOD' status
    def generateStatisticsTable(self):
        selected_items = self.inputFileListWidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No files selected", "Please select one or more files to save.")
            return

        good_files = []
        for item in selected_items:
            full_path = item.data(Qt.UserRole)
            if self.file_status.get(full_path) == "GOOD":
                good_files.append(full_path)

        if not good_files:
            QMessageBox.information(self, "No 'GOOD' Files",
                                    "Please select at least one file marked as 'GOOD' to generate statistics.")
            return

        source_dir = self.hiddenOutputDir

        # check if the hidden directory exists and get a list of its contents
        if not os.path.isdir(source_dir):
            QMessageBox.critical(self, "Error", f"Hidden output directory not found: {source_dir}")
            return

        output_files = os.listdir(source_dir)

        table_header = []
        table_data = []

        for full_path in good_files:
            base_filename = os.path.splitext(os.path.basename(full_path))[0]

            matching_v_file = None
            for f in output_files:
                if f.startswith(base_filename) and f.endswith('.v') and '_cnn' not in f:
                    matching_v_file = f
                    break

            if matching_v_file:
                source_v_path = os.path.join(source_dir, matching_v_file)
                try:
                    command = [self.utils['thrass'], '-t', 'spheroids', '-e', source_v_path]
                    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', env=os.environ.copy())
                    output_lines = result.stdout.strip().split('\n')

                    if not table_header and len(output_lines) > 0:
                        header_line = output_lines[0].replace('name fileName', 'fileName').strip().split()
                        table_header = header_line

                    if len(output_lines) > 1:
                        for line in output_lines[1:]:
                            data_row = line.strip().split()
                            if data_row:
                                # clean the filename before appending it to the table_data
                                cleaned_filename = re.sub(r'_m\d+_(GOOD|BAD)', '', data_row[0])
                                data_row[0] = cleaned_filename
                                table_data.append(data_row)

                except subprocess.CalledProcessError as e:
                    QMessageBox.warning(self, "Command Error",
                                        f"Error generating stats for {matching_v_file}:\n{e.stderr}")
                    return
                except FileNotFoundError:
                    QMessageBox.critical(self, "Command Not Found",
                                         "'thrass' command not found. Please ensure it is in your system's PATH.")
                    return
            else:
                QMessageBox.warning(self, "File Not Found",
                                    f"Skipping {base_filename}: corresponding .v file not found in the hidden output directory.")

        if table_data:
            self.showStatisticsDialog(table_header, table_data)
        else:
            QMessageBox.information(self, "No Statistics", "No statistics could be generated for the selected files.")

    # displays the collected statistics and provides an option to save as a CSV
    def showStatisticsDialog(self, header, data):
        dialog = QDialog(self)
        dialog.setWindowTitle("Statistics Table Output")
        dialog.setGeometry(100, 100, 900, 700)
        layout = QVBoxLayout(dialog)

        # create a QTableWidget to display the data as a table
        table_widget = QTableWidget(dialog)
        table_widget.setColumnCount(len(header))
        table_widget.setRowCount(len(data))
        table_widget.setHorizontalHeaderLabels(header)
        table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # populate the table with data
        for row_idx, row_data in enumerate(data):
            for col_idx, item in enumerate(row_data):
                table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(item)))

        layout.addWidget(table_widget)

        button_layout = QHBoxLayout()
        save_csv_button = QPushButton("Save as CSV", dialog)
        save_csv_button.clicked.connect(lambda: self.saveTableAsCsv(header, data))
        button_layout.addWidget(save_csv_button)

        """save_excel_button = QPushButton("Save as Excel", dialog)
        save_excel_button.clicked.connect(lambda: self.saveTableAsExcel(header, data))
        button_layout.addWidget(save_excel_button)"""

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)
        dialog.exec_()

    # saves the statistics table data to a CSV file selected by the user
    def saveTableAsCsv(self, header, data):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Statistics Table", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data)
                QMessageBox.information(self, "Success", f"Statistics table saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving the file: {e}")

    # saves the statistics table data to an Excel file selected by the user
    """def saveTableAsExcel(self, header, data):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Statistics Table", "", "Excel Files (*.xlsx)")
        if file_path:
            try:
                workbook = xlsxwriter.Workbook(file_path)
                worksheet = workbook.add_worksheet()
                for col_num, col_name in enumerate(header):
                    worksheet.write(0, col_num, col_name)
                for row_num, row_data in enumerate(data, start=1):
                    for col_num, cell_value in enumerate(row_data):
                        worksheet.write(row_num, col_num, cell_value)

                workbook.close()
                QMessageBox.information(self, "Success", f"Statistics table saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving the Excel file:\n{e}")"""

    def onTabChanged(self, index):
        if index == 0:
            self.tabType = "prediction"
        else:
            self.tabType = "retrain"

    # overridden keyPressEvent to handle arrow keys
    def keyPressEvent(self, event):
        if self.tabType == "prediction":
            if event.key() == Qt.Key_Right:
                self.markFileAsGood()
                return
            elif event.key() == Qt.Key_Left:
                self.markFileAsBad()
                return
            elif event.key() == Qt.Key_Up:
                self.showPreviousImage()
                return
            elif event.key() == Qt.Key_Down:
                self.showNextImage()
                return
        else:
            if event.key() == Qt.Key_Up:
                self.showPreviousRetrainImage()
                return
            elif event.key() == Qt.Key_Down:
                self.showNextRetrainImage()
                return

        # fall back to default handling for other keys
        super().keyPressEvent(event)

    # mark the current file as "good"
    def markFileAsGood(self):
        if not self.current_preview_filename:
            return

        base_input, _ = os.path.splitext(os.path.basename(self.current_preview_filename))

        output_dir = self.hiddenOutputDir
        if not os.path.isdir(output_dir):
            self.update_progress_text_signal.emit(f"[ERROR] Output directory not found: {output_dir}\n")
            return

        renamed = []
        for fname in os.listdir(output_dir):
            fp = os.path.join(output_dir, fname)
            if not os.path.isfile(fp):
                continue
            ext = os.path.splitext(fname)[1]
            if ext.lower() not in ('.v', '.rdf'):
                continue

            file_base_with_status = os.path.splitext(fname)[0]
            file_base = file_base_with_status.removesuffix('_BAD').removesuffix('_GOOD')

            if file_base.startswith(base_input):
                new_base = f"{file_base}_GOOD"
                new_fname = new_base + ext
                new_fp = os.path.join(output_dir, new_fname)
                try:
                    # overwrite if destination exists
                    os.replace(fp, new_fp)
                    renamed.append((fname, new_fname))
                except Exception as e:
                    self.update_progress_text_signal.emit(f"[ERROR] Failed to rename {fp} -> {new_fp}: {e}\n")

        if not renamed:
            self.update_progress_text_signal.emit(
                f"[ERROR] No output mask files found to mark for {self.current_preview_filename}\n")
        else:
            self.file_status[self.current_preview_filename] = "GOOD"
            self.updateOutputBasenames()
            self.updateFileStatusInList(self.current_preview_filename, "GOOD")
            self.showProgressMessage(f"[INFO] Marked {len(renamed)} output file(s) as GOOD for {base_input}.\n")

    # mark the current file as "bad"
    def markFileAsBad(self):
        if not self.current_preview_filename:
            return

        base_input, _ = os.path.splitext(os.path.basename(self.current_preview_filename))

        output_dir = self.hiddenOutputDir
        if not os.path.isdir(output_dir):
            self.update_progress_text_signal.emit(f"[ERROR] Output directory not found: {output_dir}\n")
            return

        renamed = []
        for fname in os.listdir(output_dir):
            fp = os.path.join(output_dir, fname)
            if not os.path.isfile(fp):
                continue
            ext = os.path.splitext(fname)[1]
            if ext.lower() not in ('.v', '.rdf'):
                continue

            file_base_with_status = os.path.splitext(fname)[0]
            file_base = file_base_with_status.removesuffix('_BAD').removesuffix('_GOOD')

            if file_base.startswith(base_input):
                new_base = f"{file_base}_BAD"
                new_fname = new_base + ext
                new_fp = os.path.join(output_dir, new_fname)
                try:
                    os.replace(fp, new_fp)
                    renamed.append((fname, new_fname))
                except Exception as e:
                    self.update_progress_text_signal.emit(f"[ERROR] Failed to rename {fp} -> {new_fp}: {e}\n")

        if not renamed:
            self.update_progress_text_signal.emit(
                f"[ERROR] No output mask files found to mark for {self.current_preview_filename}\n")
        else:
            self.file_status[self.current_preview_filename] = "BAD"

            self.updateOutputBasenames()
            self.updateFileStatusInList(self.current_preview_filename, "BAD")
            self.showProgressMessage(f"[INFO] Marked {len(renamed)} output file(s) as BAD for {base_input}.\n")

    # extract status from filename suffix like _m1_GOOD, _m2_BAD etc
    def parseStatusFromFilename(self, filename):
        match = re.search(r'(_m\d+)?_(GOOD|BAD)$', filename, re.IGNORECASE)

        if match:
            base = filename[:match.start()]
            status = match.group(2).upper()
            return base, status

        filename = re.sub(r'_m\d+$', '', filename, flags=re.IGNORECASE)

        return filename, None

    def updateFileStatusInList(self, full_path: str, forced_status: str = None) -> None:
        # use the full path to get the original filename and extension
        original_filename = os.path.basename(full_path)
        base_name, extension = os.path.splitext(original_filename)

        status = self.file_status.get(full_path)
        if forced_status:
            status = forced_status

        for i in range(self.inputFileListWidget.count()):
            item = self.inputFileListWidget.item(i)

            # match the item using the full path stored in its data
            if item.data(Qt.UserRole) == full_path:
                if status:
                    item.setText(f"{base_name}{extension} [{status}]")
                    if status == "GOOD":
                        item.setForeground(QBrush(QColor("green")))
                    elif status == "BAD":
                        item.setForeground(QBrush(QColor("red")))
                    else:
                        item.setForeground(QBrush(QColor("#A9A9A9")))
                else:
                    # if no status, show the original filename with extension
                    item.setText(original_filename)
                    item.setForeground(Qt.black)
                break

    # select all files marked as GOOD in the list
    def selectAllGoodFiles(self):
        self.inputFileListWidget.clearSelection()
        for i in range(self.inputFileListWidget.count()):
            item = self.inputFileListWidget.item(i)
            if "[GOOD]" in item.text():
                item.setSelected(True)
        self.updatePreviewList()

    # select all files marked as BAD in the list
    def selectAllBadFiles(self):
        self.inputFileListWidget.clearSelection()
        for i in range(self.inputFileListWidget.count()):
            item = self.inputFileListWidget.item(i)
            if "[BAD]" in item.text():
                item.setSelected(True)
        self.updatePreviewList()

    # selects all files that have not yet been analysed (have no status)
    def selectAllUntaggedFiles(self):
        self.inputFileListWidget.clearSelection()
        untagged_items = []

        # Iterate through the list widget items once to find items with no status
        for row in range(self.inputFileListWidget.count()):
            item = self.inputFileListWidget.item(row)
            file_path = item.data(Qt.UserRole)

            if file_path not in self.file_status:
                untagged_items.append(item)

        # Set the selection for all identified unanalysed items
        for item in untagged_items:
            item.setSelected(True)


    ####################################################
    # handle the event of Run Prediction button pressing

    def predictionButtonPressed(self):
        if not self.processingRunning:
            # check for already-analyzed files first
            selected_items = self.inputFileListWidget.selectedItems()
            if not selected_items:
                self.update_progress_text_signal.emit("No files selected.\n")
                return

            selected_filenames = [self.cleanFilename(item.text()) for item in selected_items]
            output_dir = self.hiddenOutputDir

            already_analyzed = []
            for filename in selected_filenames:
                status = self.file_status.get(os.path.join(self.selectedInputDirectory, filename))
                if status in ("TO DO", "GOOD", "BAD"):
                    already_analyzed.append(filename)

            # ask user whether to include them
            if already_analyzed:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Re-analyze Already Processed Files?")
                msg.setText(f"{len(already_analyzed)} of the selected files have already been analyzed.\n"
                            f"Do you want to include them for re-analysis?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                result = msg.exec_()

                if result == QMessageBox.No:
                    # Remove already analyzed files
                    selected_filenames = [f for f in selected_filenames if f not in already_analyzed]

                    if not selected_filenames:
                        QMessageBox.information(self, "No Files to Process",
                                                "All selected files were already analyzed.")
                        return

            # SSH key check and conditional login window
            ssh_dir = os.path.expanduser("~/.ssh")
            ssh_key_paths = [
                # sorted by priority (most secure)
                os.path.join(ssh_dir, "id_ed25519"),
                os.path.join(ssh_dir, "id_rsa"),
            ]

            ssh_keys = [f for f in ssh_key_paths if os.path.exists(f)]

            username = os.getlogin()
            password = ""
            login_successful = False

            if ssh_keys:
                self.update_progress_text_signal.emit("SSH key found for user. Proceeding without password prompt.\n")
                login_successful = True # Assume success if key exists
            else:
                self.update_progress_text_signal.emit("No SSH key found. Prompting for password via Login Window.\n")
                login_dialog = login.LoginDialog(self)
                if login_dialog.exec_() == QMessageBox.Accepted:
                    username, password = login_dialog.get_credentials()
                    if not username or not password:
                        self.update_progress_text_signal.emit("Username or password cannot be empty.\n")
                        QMessageBox.warning(self, "Login Error", "Username or password cannot be empty.")
                        return
                    login_successful = True
                else:
                    self.update_progress_text_signal.emit("[ERROR] Login cancelled.\n")
                    return

            if login_successful:
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
                    password=password,
                    ssh_keys=ssh_keys
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
            # we are running, so the button acts as "Stop"
            if self.predictionThread and self.predictionThread.isRunning():
                self.update_progress_text_signal.emit("\n*** Aborting prediction ***\n")
                self.predictionThread.stop_prediction_process()


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
        unprocessed_items = [item for item in selected_items if "[TO DO]" not in item.text()]
        processed_items = [item for item in selected_items if "[TO DO]" in item.text()]

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
        output_dir = self.hiddenOutputDir

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

        filenames_to_log = [os.path.basename(f) for f in input_files]
        log_message = ", ".join(filenames_to_log)
        self.update_progress_text_signal.emit(f"Running prediction with {len(input_files)} file(s).\n"
                                              f"Input files: {log_message}\n"
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

    def refresh_mask(self):
        try:
            if not self.current_preview_filename:
                self.update_progress_text_signal.emit("[INFO] No preview file selected, skipping refresh.\n")
                return

            correction_dir = os.path.join(self.hiddenOutputDir, "corrections")
            self.update_progress_text_signal.emit(f"[DEBUG] correction_dir = '{correction_dir}'\n")
            if not os.path.isdir(correction_dir):
                self.update_progress_text_signal.emit(f"[INFO] No corrections dir at {correction_dir}\n")
                return

            # base name without extension (from input .tif file)
            filename_only = os.path.basename(self.current_preview_filename)
            base_name = os.path.splitext(filename_only)[0]

            # find matching _mN files in corrections
            pattern = re.compile(rf"^{re.escape(base_name)}_m\d+")
            rdf_corr_candidates = [f for f in os.listdir(correction_dir) if f.endswith(".rdf") and pattern.match(f)]
            if not rdf_corr_candidates:
                self.update_progress_text_signal.emit(
                    f"[INFO] No corrections RDF for {self.current_preview_filename}\n")
                return

            for rdf_corr_file in rdf_corr_candidates:
                rdf_corr_path = os.path.join(correction_dir, rdf_corr_file)

                # find matching user RDF in hiddenOutputDir (may have GOOD/BAD tag)
                rdf_user_candidates = [
                    f for f in os.listdir(self.hiddenOutputDir)
                    if f.endswith(".rdf") and f.startswith(rdf_corr_file[:-4])
                ]
                if not rdf_user_candidates:
                    self.update_progress_text_signal.emit(f"[WARNING] No user RDF for {rdf_corr_file}, skipping.\n")
                    continue

                rdf_user_file = rdf_user_candidates[0]
                rdf_user_path = os.path.join(self.hiddenOutputDir, rdf_user_file)

                # extract status suffix (_GOOD/_BAD) if present
                status_suffix = ""
                if "_GOOD" in rdf_user_file:
                    status_suffix = "_GOOD"
                elif "_BAD" in rdf_user_file:
                    status_suffix = "_BAD"

                # compare correction RDF and user RDF
                import filecmp
                try:
                    if os.path.exists(rdf_user_path) and filecmp.cmp(rdf_corr_path, rdf_user_path, shallow=False):
                        self.update_progress_text_signal.emit(
                            f"[INFO] RDF {rdf_corr_file} identical to {rdf_user_file}, skipping.\n"
                        )
                        continue
                except Exception as e:
                    self.update_progress_text_signal.emit(f"[WARNING] Could not compare RDFs: {e}\n")

                # find matching .v file in corrections
                v_candidates = [f for f in os.listdir(correction_dir) if f.endswith(".v") and pattern.match(f)]
                if not v_candidates:
                    self.update_progress_text_signal.emit(f"[WARNING] No .v file in corrections for {rdf_corr_file}\n")
                    continue

                v_file_name = v_candidates[0]
                v_file_path = os.path.join(correction_dir, v_file_name)

                # run thrass
                self.update_progress_text_signal.emit(f"Running thrass on {v_file_name} in corrections...\n")
                thrass_command = ["thrass", "-t", "cnnPrepare", "-b", v_file_name]
                env = os.environ.copy()
                result = subprocess.run(
                    thrass_command, capture_output=True, text=True, cwd=correction_dir, env=env
                )
                self.update_progress_text_signal.emit(f"Thrass stdout:\n{result.stdout}\n")
                if result.stderr:
                    self.update_progress_text_signal.emit(f"Thrass stderr:\n{result.stderr}\n")

                if result.returncode != 0:
                    raise RuntimeError(f"Thrass failed with exit code {result.returncode} for {v_file_path}")

                # handle CNN files
                cnn_candidates = [f for f in os.listdir(correction_dir) if pattern.match(f) and f.endswith("_cnn.v")]
                if not cnn_candidates:
                    raise FileNotFoundError(f"Thrass cnn file not found for {v_file_name}")

                cnn_file_name = cnn_candidates[0]
                cnn_path = os.path.join(correction_dir, cnn_file_name)

                # delete _cnn_005.v if exists
                cnn_005_candidates = [f for f in os.listdir(correction_dir) if
                                      pattern.match(f) and f.endswith("_cnn_0005.v")]
                for f in cnn_005_candidates:
                    try:
                        os.remove(os.path.join(correction_dir, f))
                        self.update_progress_text_signal.emit(f"[INFO] Deleted {f}\n")
                    except Exception as e:
                        self.update_progress_text_signal.emit(f"[WARNING] Could not delete {f}: {e}\n")

                # backup old RDF in hiddenOutputDir
                if os.path.exists(rdf_user_path):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rdf_backup_name = rdf_user_file.replace(".rdf", f"_old_{timestamp}.rdf")
                    rdf_backup_path = os.path.join(self.hiddenOutputDir, rdf_backup_name)
                    shutil.move(rdf_user_path, rdf_backup_path)
                    self.update_progress_text_signal.emit(f"[INFO] Renamed old RDF to {rdf_backup_name}\n")

                # rename and move new RDF
                new_rdf_name = rdf_corr_file.replace(".rdf", f"{status_suffix}.rdf")
                new_rdf_path = os.path.join(self.hiddenOutputDir, new_rdf_name)
                shutil.move(rdf_corr_path, new_rdf_path)

                # rename and move CNN file with status
                new_cnn_name = cnn_file_name.replace("_cnn.v", f"_cnn{status_suffix}.v")
                shutil.move(cnn_path, os.path.join(self.hiddenOutputDir, new_cnn_name))

                # delete symlink .v in corrections
                try:
                    os.remove(v_file_path)
                    self.update_progress_text_signal.emit(f"[INFO] Deleted symlink {v_file_name}\n")
                except Exception as e:
                    self.update_progress_text_signal.emit(f"[WARNING] Could not delete {v_file_name}: {e}\n")

                self.update_progress_text_signal.emit(
                    f"[INFO] Updated {new_rdf_name} and {new_cnn_name} → {self.hiddenOutputDir}\n"
                )

            # refresh mask visualization for current preview
            mask_settings = {
                'show_gradient': self.prediction_show_gradient,
                'show_filled': self.prediction_show_filled,
                'show_contour': self.prediction_show_contour,
                'gradient_colormap': self.prediction_gradient_colormap,
                'filled_color': self.prediction_filled_color,
                'contour_color': self.prediction_contour_color
            }
            pixmap = self.process_single_image_and_mask(
                self.current_preview_filename,
                self.selectedInputDirectory,
                mask_settings
            )
            self.predictionMaskedPixmaps[self.current_preview_filename] = pixmap
            self.imagePreviewLabel.setPixmap(pixmap)
            self.imagePreviewLabel.setText("")
            self.update_progress_text_signal.emit("[INFO] Mask visualization refreshed.\n")

        except Exception as e:
            self.update_progress_text_signal.emit(f"[ERROR] refresh_mask failed: {e}\n")


class PyMarAiThread(QtCore.QThread):
    progress_update = pyqtSignal(int, int, str, str)
    text_update = pyqtSignal(str)
    error_message = pyqtSignal(str)

    def __init__(self, parent, params, username, password, ssh_keys):
        super().__init__()
        self.parent = parent
        self.params = params
        self.username = username
        self.password = password
        self.ssh_keys = ssh_keys
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
                target=pymarai.predict.gui_entry_point,
                params=self.params,
                username=self.username,
                password=self.password,
                ssh_keys=self.ssh_keys,
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
        # target, params, username, password, ssh_keys, stdout_pipe_child, progress_pipe_child, hostname
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

    def __init__(self, app_instance, filenames, input_dir, mask_settings, tab_type, parent=None):
        super().__init__(parent)
        self.app = app_instance
        self.filenames = filenames
        self.input_dir = input_dir
        self.mask_settings = mask_settings
        self.tab_type = tab_type
        self._abort = False

    def run(self):
        total = len(self.filenames)

        for i, filename in enumerate(self.filenames):
            if self._abort:
                self.progress_message.emit("Batch mask application aborted.\n")
                break

            try:
                # select method based on tab_type
                if self.tab_type == "prediction":
                    pixmap = self.app.process_single_image_and_mask(
                        filename, self.input_dir, self.mask_settings, signals=self
                    )
                elif self.tab_type == "retrain":
                    pixmap = self.app.process_single_image_and_mask_retrain(
                        filename, self.input_dir, self.mask_settings, signals=self
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

class FileStatusWorker(QThread):
    batch_ready = pyqtSignal(list)  # list of (row, text, color, status)
    finished_all = pyqtSignal()

    def __init__(self, input_items, output_dir, parse_status_func, strip_microscope_tag_func, parent=None):
        super().__init__(parent)
        self.input_items = input_items
        self.output_dir = output_dir
        self.parse_status_func = parse_status_func
        self.strip_microscope_tag_func = strip_microscope_tag_func

    def run(self):
        status_map = {}
        if os.path.isdir(self.output_dir):
            for f in os.listdir(self.output_dir):
                if not os.path.isfile(os.path.join(self.output_dir, f)):
                    continue
                base_no_ext = os.path.splitext(f)[0]
                base, status = self.parse_status_func(base_no_ext)
                if status:  # GOOD/BAD
                    status_map[base] = status
                else:
                    status_map.setdefault(base, "TO DO")

        batch = []
        for row, item in enumerate(self.input_items):
            full_path = item.data(Qt.UserRole)
            base_name = os.path.splitext(os.path.basename(full_path))[0]
            base_name = self.strip_microscope_tag_func(base_name)

            if base_name in status_map:
                status_found = status_map[base_name]
                text = f"{os.path.basename(full_path)} [{status_found}]"
                if status_found == "GOOD":
                    color = QColor("green")
                elif status_found == "BAD":
                    color = QColor("red")
                elif status_found == "TO DO":
                    color = QColor("orange")
                else:
                    color = QColor("#A9A9A9")
            else:
                status_found = None
                text = os.path.basename(full_path)
                color = QColor("black")

            batch.append((row, text, color, status_found))

            if len(batch) >= 50:
                self.batch_ready.emit(batch)
                batch = []
                self.msleep(1)

        if batch:
            self.batch_ready.emit(batch)

        self.finished_all.emit()

class ScaledLabel(QLabel):
    zoom_changed_signal = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.zoom_factor = 1
        self._pixmap = None
        if self.pixmap():
            self._pixmap = QPixmap(self.pixmap())

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.updatePixmap()

    def setZoom(self, zoom_factor):
        self.zoom_factor = zoom_factor
        self.updatePixmap()
        self.zoom_changed_signal.emit(self.zoom_factor)

    def getZoom(self):
        return self.zoom_factor

    def zoomIn(self):
        if self.zoom_factor < 10:
            self.setZoom(self.zoom_factor + 1)

    def zoomOut(self):
        if self.zoom_factor > 1:
            self.setZoom(self.zoom_factor - 1)

    def updatePixmap(self):
        if self._pixmap is not None:
            QLabel.setPixmap(self, self._pixmap.scaled(self.width() * self.zoom_factor, self.height() * self.zoom_factor, Qt.KeepAspectRatio))

    def resizeEvent(self, event):
        self.updatePixmap()

# main function to start GUI
def main():
    QCoreApplication.setApplicationVersion(__version__);
    QCoreApplication.setOrganizationName("Helmholtz-Zentrum Dresden-Rossendorf");
    QCoreApplication.setOrganizationDomain("hzdr.de");
    QCoreApplication.setApplicationName("PyMarAI");
    app = QApplication(sys.argv)
    config = AppConfig()
    window = PyMarAiGuiApp(config)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
