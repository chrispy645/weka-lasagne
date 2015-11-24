package weka.lasagne;

import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.lasagne.layers.DenseLayer;
import weka.lasagne.nonlinearities.Sigmoid;

public class TestClassifiers {
	
	public Instances loadIris() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		return data;
	}
	
	@Test
	public void testDenseLayer() throws Exception {
		DenseLayer dense = new DenseLayer();
		dense.setNonLinearity(new Sigmoid());
		dense.setNumUnits(10);
		System.out.println(dense.getOutputString());
	}

}
