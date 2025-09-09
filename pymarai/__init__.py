import sys
from datetime import datetime
import signal

__version__ = '0.8.0'
__copyright__ = 'Copyright (c) 2025 Varvara Melnyk, Jens Maus, www.hzdr.de'

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
