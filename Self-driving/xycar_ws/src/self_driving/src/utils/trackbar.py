#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, shutil
import cv2 as cv
from collections import OrderedDict
from datetime import datetime
from numpy.random import sample


class Trackbar():
    filename = None
    config = {}

    def __init__(self, filename=None, winname=None, debug=False):
        self.filename = filename
        self.debug = debug

        if not winname:
            self.winname = "{:%Y%m%d_%H%M%S}_{}".format(datetime.now(), sample())
        else:
            self.winname = winname

        if filename:
            print("[Load param]", filename)
            with open(filename) as fp:
                config = json.load(fp)

            self.config = config

        if debug:
            cv.namedWindow(self.winname)
            if "default" in self.config:
                for trackbarname, [min_val, val, max_val] in sorted(config["default"].items()):
                    self._create_default(trackbarname, min_val, val, max_val)
            if "kernel" in self.config:
                for trackbarname, [min_val, val, max_val] in sorted(config["kernel"].items()):
                    self._cerate_kernel(trackbarname, min_val, val, max_val)
            self._create_save("save", 0, 0, 1)


    def _create_default(self, trackbarname, min_val, val, max_val):
        callback = self._callback_default(trackbarname)
        cv.createTrackbar(trackbarname, self.winname, min_val, max(max_val-min_val, max_val), callback)
        cv.setTrackbarMin(trackbarname, self.winname, min_val)
        cv.setTrackbarPos(trackbarname, self.winname, val)


    def _cerate_kernel(self, trackbarname, min_val, val, max_val):
        callback = self._callback_kernel(trackbarname)
        cv.createTrackbar(trackbarname, self.winname, min_val, max(max_val-min_val, max_val), callback)
        cv.setTrackbarMin(trackbarname, self.winname, min_val)
        cv.setTrackbarPos(trackbarname, self.winname, val)


    def _create_save(self, trackbarname, min_val, val, max_val):
        callback = self._callback_save(trackbarname)
        cv.createTrackbar(trackbarname, self.winname, min_val, max(max_val-min_val, max_val), callback)
        cv.setTrackbarMin(trackbarname, self.winname, min_val)
        cv.setTrackbarPos(trackbarname, self.winname, val)


    def _callback_default(self, trackbarname):
        def callback(pos):
            min_val = self.config["default"][trackbarname][0]
            self.config["default"][trackbarname][1] = pos
        return callback


    def _callback_kernel(self, trackbarname):
        def callback(pos):
            if pos % 2:
                min_val = self.config["kernel"][trackbarname][0]
                self.config["kernel"][trackbarname][1] = pos
        return callback


    def _callback_save(self, trackbarname):
        def callback(pos):
            if pos == 1:
                filename = self.filename
                if filename:
                    shutil.copyfile(filename, "{}_{:%Y%m%d_%H%M%S}_bak".format(filename, datetime.now()))
                else:
                    filename = "Unknown"

                with open(filename, 'w') as fp:
                    json.dump(self.config, fp, indent=4)
                print("SAVE:", filename)
                self.setTrackbarPos(trackbarname, 0)
        return callback


    def setTrackbarPos(self, trackbarname, value):
        cv.setTrackbarPos(trackbarname, self.winname, value)


    def getValue(self, trackbarname, type=None, min_val=None, val=None, max_val=None):
        if "default" in self.config and trackbarname in self.config["default"]:
            return self.config["default"][trackbarname][1]
        elif "kernel" in self.config and trackbarname in self.config["kernel"]:
            return self.config["kernel"][trackbarname][1]

        if None in (min_val, val, max_val):
            raise Exception("At least one of min_value, value, and max_value is None.")
        else:
            if type == "default":
                if "default" not in self.config:
                    self.config["default"] = {}
                self.config["default"][trackbarname] = [min_val, val, max_val]
                if self.debug:
                    self._create_default(trackbarname, min_val, val, max_val)
                return self.config["default"][trackbarname][1]

            elif type == "kernel":
                if "kernel" not in self.config:
                    self.config["kernel"] = {}
                self.config["kernel"][trackbarname] = [min_val, val, max_val]
                if self.debug:
                    self._cerate_kernel(trackbarname, min_val, val, max_val)
                return self.config["kernel"][trackbarname][1]
            else:
                raise Exception("type key Error: 'default' or 'kernel'")


    def show(self, frame):
        if self.debug:
            cv.imshow(self.winname, frame)
