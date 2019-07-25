#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:55:12 2018

@author: paf
"""
import sys, os

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

# Get or default
def getOrDefault(provider, default):
    try:
        return provider()
    except:
        return default
    

def canTFUseGPU():
    """
    Returns True if tensorflow has access to a gpu device
    """
    from tensorflow.python.client import device_lib
    for dev in device_lib.list_local_devices():
        if dev.device_type == 'GPU':
            return True
    return False