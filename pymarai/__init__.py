#
# pyMarAI - Tumor Spheroids Auto Delineation Tool
#           https://github.com/hzdr-MedImaging/pyMarAI
#
# Copyright (C) 2025 hzdr.de and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from datetime import datetime
import signal

from importlib.metadata import version, PackageNotFoundError

try:
  __version__ = version("pymarai")
except PackageNotFoundError:
  __version__ = "0.0.0+unknown"

__copyright__ = 'Copyright (c) 2025-2026 Jens Maus, Varvara Melnyk, www.hzdr.de'

def error(*objs):
  print("ERROR:", *objs, file=sys.stderr)
  sys.exit(1)

def warning(*objs):
  print("WARNING:", *objs, file=sys.stderr)

def signal_handler(signal, frame):
  warning("User abort (CTRL-C) received.")
  sys.exit(0)

def mkdate(datestr):
  try:
    return datetime.strptime(datestr, '%H:%M')
  except ValueError:
    try:
      return datetime.strptime(datestr, '%H:%M:%S')
    except ValueError:
      try:
        return datetime.strptime(datestr, '%m/%d/%Y %H:%M:%S')
      except ValueError:
        raise argparse.ArgumentTypeError(datestr + ' is not a proper datetime string ("hh:mm", "hh:mm:ss" or "MM/DD/YYYY hh:mm:ss").')

signal.signal(signal.SIGINT, signal_handler)
