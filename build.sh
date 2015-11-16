#!/bin/bash

CLASSPATH=$WEKA_HOME/weka.jar:$WEKA_HOME/packages/wekaPython/wekaPython.jar:$WEKA_HOME/packages/PyScriptClassifier/PyScriptClassifier.jar
echo $CLASSPATH

#ant clean && ant exejar -Dpackage=weka-lasagne
ant clean && ant make_package -Dpackage=weka-lasagne
cd dist
java weka.core.WekaPackageManager -offline -install-package weka-lasagne.zip
