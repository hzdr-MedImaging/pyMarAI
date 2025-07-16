import sys
import os
import platform
import subprocess
import multiprocessing

import login
import predict

from PyQt5 import QtCore
from multiprocessing import Pipe
from threading import Thread

from config import AppConfig

from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QCheckBox, QSpacerItem,
                             QProgressBar, QPushButton, QSizePolicy, QPlainTextEdit, QLineEdit, QFileDialog, QListWidget, QListWidgetItem)

from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageOps

class PyMarAiGuiApp(QDialog):

    # we use constructor to create the GUI layout
    def __init__(self, config: AppConfig, parent=None):
        super(PyMarAiGuiApp, self).__init__(parent)

        self.config = config

        self.microscopes = config.get_microscopes()
        self.defaultMicroscopeType = config.get_default_microscope()

        # load persistent settings
        self.settings = QSettings("PyMarAi", "PyMarAiGuiApp")
        self.selectedInputDirectory = self.settings.value("lastInputDir", os.getcwd())
        self.lastOutputDirectory = self.settings.value("lastOutputDir", os.getcwd())
        self.currentImage = None
        self.currentImageIsPillow = False

        self.previewList = []
        self.previewIndex = 0
        self.predictionThread = None
        self.loadDataThread = None
        self.processingRunning = False
        self.outputBasenames = set()

        # creates GUI components
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
        }

        QProgressBar::chunk {
            background-color: #0078d7;
            border-radius: 4px;
        }
        """)
        self.progressBar.setFixedHeight(8)
        self.progressBar.setFixedWidth(300)
        self.progressBar.setMinimum(1)
        self.progressBar.setMaximum(0)
        self.progressBar.hide()
        predictionButtonLayout.addWidget(self.progressBarLabel)
        predictionButtonLayout.addWidget(self.progressBar)

        progressLabel = self.createLabel("Progress Output:")
        self.progressPlainTextEdit = self.createPlainTextEdit()

        # now we add all our GUI components to dialog using grid layout
        mainLayout = QGridLayout()
        mainLayout.setColumnMinimumWidth(0, 75)
        mainLayout.setColumnStretch(1, 1)
        mainLayout.setColumnStretch(3, 1)

        row = 0
        mainLayout.addWidget(inputFileLabel, row, 0)

        row += 1
        mainLayout.addLayout(inputFileShowLayout, row, 0, 1, 4)

        row += 1
        mainLayout.addLayout(inputFilePathButtonsLayout, row, 0)
        mainLayout.addLayout(navLayout, row, 3)

        row += 1
        mainLayout.addLayout(outputFilePathLayout, row, 0, 1, 4)

        row += 1
        mainLayout.addLayout(microscopeLayout, row, 0)

        row += 1
        mainLayout.addLayout(predictionButtonLayout, row, 0, 1, 4)

        row += 1
        mainLayout.addWidget(progressLabel, row, 0, 1, 4)

        row += 1
        mainLayout.addWidget(self.progressPlainTextEdit, row, 0, 1, 4)

        # layout finished, add it as the main layout
        self.setLayout(mainLayout)

        self.setWindowTitle("Spheroids-DNN: Auto Inference Tool")
        self.resize(1000, 800)

        self.initElements()

    # initial state of elements
    def initElements(self):
        self.enableWidgets(True)

        # update output path field with last used directory
        if os.path.isdir(self.lastOutputDirectory):
            self.outputFilePathTextEdit.insert(self.lastOutputDirectory)

        # autoload last input folder if it exists
        if os.path.isdir(self.selectedInputDirectory):
            self.loadFilesFromDirectory(self.selectedInputDirectory)

    def enableWidgets(self, enable):
        self.inputFileListWidget.setEnabled(enable)
        self.outputFilePathTextEdit.setEnabled(enable)
        self.inputDirButton.setEnabled(enable)
        self.outputFilePathSelectButton.setEnabled(enable)
        self.microscopeComboBox.setEnabled(enable)
        self.predictionButton.setEnabled(enable)
        self.selectAllButton.setEnabled(enable)
        self.deselectAllButton.setEnabled(enable)
        self.prevButton.setEnabled(enable)
        self.nextButton.setEnabled(enable)

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
        dir = QFileDialog.getExistingDirectory(self, 'Select an output folder:')
        if dir != "":
            self.outputFilePathTextEdit.clear()
            self.outputFilePathTextEdit.insert(dir)
            self.settings.setValue("lastOutputDir", dir)

            self.updateOutputBasenames()

            if self.inputFileListWidget.count() > 0:
                self.markAnalyzedFiles()

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
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if dir_path:
            self.settings.setValue("lastInputDir", dir_path)
            self.loadFilesFromDirectory(dir_path)

    def loadFilesFromDirectory(self, dir_path):
        self.progressPlainTextEdit.appendPlainText(f"Scanning directory: {dir_path}.\n")
        self.inputFileListWidget.clear()
        self.setProgressBar("Loading files...")
        self.previewList = []

        self.processingRunning = True

        self.updateOutputBasenames()

        self.fileLoader = FileLoaderWorker(dir_path)
        self.fileLoader.filesLoaded.connect(self.onFilesLoaded)
        self.fileLoader.errorOccurred.connect(self.onFileLoadError)
        self.fileLoader.finished.connect(self.fileLoadingFinished)
        self.fileLoader.start()

        self.switchElementsToPrediction(False)

    def onFilesLoaded(self, file_list, dir_path):
        self.selectedInputDirectory = dir_path
        self.inputFileListWidget.clear()
        self.inputFileListWidget.clearSelection()

        for file in file_list:
            item = QListWidgetItem(file)
            self.inputFileListWidget.addItem(item)

        self.markAnalyzedFiles()  # mark the list visually

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

    def onFileLoadError(self, error_message):
        self.progressPlainTextEdit.appendPlainText(f"Error loading files: {error_message}.\n")
        self.imagePreviewLabel.setText("Failed to load images")

    def fileLoadingFinished(self):
        self.processingRunning = False
        self.setProgressBar()

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
                    raise ValueError("QPixmap could not load the image.\n")
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
            print(f"Error loading image '{full_path}': {e}.\n")

    def showNextImage(self):
        if self.previewList:
            self.previewIndex = (self.previewIndex + 1) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)

    def showPreviousImage(self):
        if self.previewList:
            self.previewIndex = (self.previewIndex - 1 + len(self.previewList)) % len(self.previewList)
            self.showImageAtIndex(self.previewIndex)

    def selectAllFiles(self):
        self.inputFileListWidget.selectAll()
        self.updatePreviewList()

    def deselectAllFiles(self):
        self.inputFileListWidget.clearSelection()
        self.updatePreviewList()

    def setProgressBar(self, text=None):
        if text == None:
            self.progressBarLabel.setText("")
            self.progressBar.setMinimum(1)
            self.progressBar.setMaximum(1)
            self.progressBar.hide()
            self.progressBarLabel.hide()
        else:
            self.progressBarLabel.setText(text)
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.progressBar.show()
            self.progressBarLabel.show()

    def toggleDeviceList(self):
        if self.useGpuCheckBox.isChecked():
            self.deviceComboBox.setEnabled(True)
        else:
            self.deviceComboBox.setEnabled(False)

    # switching between two states of elements
    # for prediction and user interaction mode

    def switchElementsToPrediction(self, isPrediction):
        if isPrediction:
            self.predictionButton.setText("Stop")
            self.predictionButton.setEnabled(True)
        else:
            self.predictionButton.setText("Run Prediction")

    def openAnalyzedFile(self, item):
        text = item.text()
        if "[✓]" not in text:
            self.progressPlainTextEdit.appendPlainText(f"'{text}' is not marked as analyzed. Skipping open with ROVER.\n")
            return

        filename = self.cleanFilename(text)
        base = os.path.splitext(filename)[0]

        output_dir = self.outputFilePathTextEdit.text().strip()
        if not os.path.isdir(output_dir):
            self.progressPlainTextEdit.appendPlainText("Output directory is not valid.\n")
            return

        # Find all .v files matching the base filename in output_dir
        vFiles = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f)) and f.startswith(base) and f.endswith('.v')
        ]

        # Find all .rdf files matching the base filename in output_dir
        rdfFiles = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f)) and f.startswith(base) and f.endswith('.rdf')
        ]

        if not vFiles:
            self.progressPlainTextEdit.appendPlainText(f"No .v file found for '{filename}'.\n")
            return

        if not rdfFiles:
            self.progressPlainTextEdit.appendPlainText(f"No .rdf file found for '{filename}'.\n")
            return

        # Show which files you will open
        self.progressPlainTextEdit.appendPlainText(
            f"Opening ROVER for '{filename}':\n"
            + "\n".join(vFiles + rdfFiles)
        )

        # Open each .v file with rover, with -R 1 option
        for vFile in vFiles:
            try:
                # Correct way to pass command + arguments:
                subprocess.Popen(["rover", "-R", "1", vFile])
            except Exception as e:
                self.progressPlainTextEdit.appendPlainText(f"Failed to open {vFile}: {e}\n")

    #############################################
    # handle the event of Run Prediction button pressing

    def predictionButtonPressed(self):
        if not self.processingRunning:

            # show login dialog for SSH credentials
            login_dialog = login.LoginDialog(self)
            if login_dialog.exec_() == QDialog.Accepted:
                username, password = login_dialog.get_credentials()

                if not username or not password:
                    self.progressPlainTextEdit.appendPlainText("Username or password cannot be empty.\n")
                    return

                # save credentials in instance variable or pass them forward
                self.ssh_username = username
                self.ssh_password = password

                # clear the progress output and prepare for job processing
                self.progressPlainTextEdit.clear()
                self.processJobs = []

                # build the prediction job
                prediction_result = self.prediction()
                if prediction_result is None:
                    return

                target, args = prediction_result
                self.processJobs.append({'app': 'prediction', 'target': target, 'args': args})

                # Create and start the prediction thread
                self.predictionThread = PyMarAiThread(self, self.processJobs)
                self.predictionThread.started.connect(self.processingStarted)
                self.predictionThread.finished.connect(self.processingFinished)

                self.predictionThread.start()
                self.switchElementsToPrediction(True)

            else:
                self.progressPlainTextEdit.appendPlainText("Login cancelled.\n")

        else:
            # we first have to make sure that we terminate an already running
            self.predictionThread.terminate()

            self.showProgressMessage("\n*** Aborting prediction ***\n")

            self.predictionThread.terminate()
            self.predictionThread.wait()
            self.processingFinished()

    # function to combine the selected parameters and actually
    # start the prediction by passing these parameters
    # to runPrediction.py main function

    def prediction(self):
        # collect input files
        selected_items = self.inputFileListWidget.selectedItems()
        if not selected_items:
            self.progressPlainTextEdit.appendPlainText("No input files selected.\n")
            return None

        input_files = [
            os.path.join(self.selectedInputDirectory, self.cleanFilename(item.text()))
            for item in selected_items
        ]

        # output directory
        output_dir = self.outputFilePathTextEdit.text().strip()
        if not os.path.isdir(output_dir):
            self.progressPlainTextEdit.appendPlainText("Invalid output directory.\n")
            return None

        # microscope selection
        microscope_text = self.microscopeComboBox.currentText()
        if microscope_text.strip() == "-":
            self.progressPlainTextEdit.appendPlainText("No microscope selected.\n")
            return None

        microscope_code = microscope_text.split(":")[0].strip()

        # Form the args list
        args = []

        for input_file in input_files:
            args.append("--input")
            args.append(input_file)

        args.append("--output")
        args.append(output_dir)

        args.append("--microscope")
        args.append(microscope_code)

        self.progressPlainTextEdit.appendPlainText(f"Running prediction with:\n"
                                                   f"Input files: {input_files}\n"
                                                   f"Output dir: {output_dir}\n"
                                                   f"Microscope: {microscope_code}\n")

        return (predict.main, args)

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

    # call when processing thread finished
    def processingFinished(self):
        self.processingRunning = False
        self.enableWidgets(True)
        self.progressBar.hide()
        self.progressBarLabel.hide()
        self.updateOutputBasenames()
        self.markAnalyzedFiles()
        self.switchElementsToPrediction(False)

class PyMarAiThread(QtCore.QThread):
    def __init__(self, parent, processJobs):
        super().__init__()
        self.parent = parent
        self.processJobs = processJobs

        # create pipe to communicate with the thread
        mother_pipe, child_pipe = Pipe()
        self.transport = child_pipe

        # create an emitter to forward messages to GUI
        self.emitter = self.Emitter(mother_pipe)
        self.emitter.message.connect(self.parent.showProgressMessage)
        self.emitter.start()

    def run(self):
        # redirect standard IO
        sys.stdout = self.StdIORedirector(self.transport)
        sys.stderr = self.StdIORedirector(self.transport)

        for job in self.processJobs:
            app_name = job.get('app', 'Prediction')
            self.parent.setProgressBar(f"Running '{app_name}'")

            if platform.system() == 'Windows' and sys.version_info < (3, 5):
                # fallback to subprocess for older Windows Python versions
                cmd = [item for sublist in [[job['app']], job['cli_args']] for item in sublist]
                p = subprocess.Popen(cmd, bufsize=0, universal_newlines=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                while p.poll() is None:
                    output = p.stdout.readline()
                    if output:
                        print(output, end='')
                p.wait()
            else:
                # run run_analysis with kwargs in a separate process
                p = self.Process(target=job['target'], args=job['args'], stdout=sys.stdout, stderr=sys.stderr)

                p.start()
                p.join()

        self.parent.setProgressBar()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    class StdIORedirector(object):
        def __init__(self, transport):
            self.transport = transport

        def write(self, string):
            self.transport.send(string)

        def flush(self):
            pass

    class Emitter(QtCore.QObject, Thread):
        message = QtCore.pyqtSignal(str)

        def __init__(self, transport, parent=None):
            QtCore.QObject.__init__(self, parent)
            Thread.__init__(self)
            self.transport = transport
            self.daemon = True

        def run(self):
            while True:
                try:
                    msg = self.transport.recv()
                except EOFError:
                    break
                else:
                    self.message.emit(msg)

    class Process(multiprocessing.Process):
        def __init__(self, target, args, stdout, stderr):
            super().__init__(daemon=True)
            self.target = target
            self.args = args
            self.stdout = stdout
            self.stderr = stderr

        def run(self):
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            self.target(self.args)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


# thread to load data
class FileLoaderWorker(QThread):
    filesLoaded = pyqtSignal(list, str)
    errorOccurred = pyqtSignal(str)

    def __init__(self, dir_path):
        super().__init__()
        self.dir_path = dir_path

    def run(self):
        if not os.path.isdir(self.dir_path):
            self.errorOccurred.emit("Selected input directory is invalid.\n")
            return

        file_list = [
            f for f in os.listdir(self.dir_path)
            if os.path.isfile(os.path.join(self.dir_path, f)) and f.lower().endswith(('.tif', '.png'))
        ]

        if not file_list:
            self.errorOccurred.emit("No compatible image files (.tif or .png) found in the selected folder.\n")
            return

        self.filesLoaded.emit(file_list, self.dir_path)


# main function to start GUI
def main():
    app = QApplication(sys.argv)
    config = AppConfig()
    window = PyMarAiGuiApp(config)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()