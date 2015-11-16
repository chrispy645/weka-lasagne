package weka.lasagne;

import weka.classifiers.functions.LasagneNet;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.lasagne.layers.DenseLayer;
import weka.lasagne.layers.Layer;

public class TestPythonScript {
	
	public static void main(String[] args) throws Exception {
		
		LasagneNet net = new LasagneNet();
		net.setLayers(new Layer[] {
			new DenseLayer(), new DenseLayer()
		});
		
		DataSource iris = new DataSource("datasets/iris.arff");
		Instances data = iris.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		System.out.println( net.getOutputString(data) );
		
	}

}
