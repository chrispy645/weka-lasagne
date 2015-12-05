package weka.lasagne;

import weka.classifiers.functions.LasagneNet;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.lasagne.layers.Conv2DLayer;
import weka.lasagne.layers.DenseLayer;
import weka.lasagne.layers.Layer;
import weka.lasagne.objectives.CategoricalCrossEntropy;
import weka.nolearn.BatchIterator;
import weka.nolearn.ImageBatchIterator;

public class TestMnist {
	
	public static void main(String[] args) throws Exception {
		
		LasagneNet net = new LasagneNet();
		Conv2DLayer conv = new Conv2DLayer();
		conv.setFilterSizeX(5);
		conv.setFilterSizeY(5);
		conv.setNumFilters(5);
		net.setLayers(new Layer[] {
			conv
		});
		
		DataSource mnist = new DataSource("datasets/mnist.meta.arff");
		Instances data = mnist.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		
		net.setOutFile("/tmp/out.txt");
		net.setNumEpochs(100);
		net.setBatchSize("100000");
		ImageBatchIterator it = new ImageBatchIterator();
		it.setBatchSize(128);
		it.setWidth(28);
		it.setHeight(28);
		it.setPrefix("/Users/cjb60/github/weka-lasagne/mnist-data");
		net.setBatchIterator(it);
		CategoricalCrossEntropy cat = new CategoricalCrossEntropy();
		net.setLossFunction(cat);
		
		System.out.println( net.getOutputString(data) );
		
		net.buildClassifier(data);
	}

}
