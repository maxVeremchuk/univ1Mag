#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

def download_github_code(path):
    filename = path.rsplit("/")[-1]
    os.system("wget https://github.com/maxVeremchuk/univ1Mag/blob/master/coursework/{} -O {}".format(path, filename))

def setup_project():
    #download_github_code("requirements_colab.txt")
    download_github_code("utils.py")
    download_github_code("dialogue_manager.py")

