#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

def download_github_code(filename):
    os.system("wget https://github.com/maxVeremchuk/univ1Mag/blob/master/coursework/{}/?raw=true -O {}".format(filename, filename))

def setup_project():
    #download_github_code("requirements_colab.txt")
    download_github_code("utils.py")
    download_github_code("dialogue_manager.py")

