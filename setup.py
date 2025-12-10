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

from setuptools import setup
setup()

# this is hackish but there doesn't seem to be a nice solution for this :(
#if platform.system() != 'Windows':
#  os.system("chmod 664 /usr/local/lib64/python*/dist-packages/pymarai-*-py*.egg")
#  os.system("chmod 775 /usr/local/bin/pymarai")
