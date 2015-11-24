#!/bin/bash

# clean weka launch
java -Xmx6g -cp $WEKA_HOME/weka.jar -Dweka.packageManager.offline=true weka.gui.GUIChooser
