#!/bin/bash

cat ros/src/tl_detector/light_classification/frozen_faster_rcnn.tar.bz2.parta* > ros/src/tl_detector/light_classification/frozen_faster_rcnn.tar.bz2
tar -vxjf ros/src/tl_detector/light_classification/frozen_faster_rcnn.tar.bz2 -C ros/src/tl_detector/light_classification/


