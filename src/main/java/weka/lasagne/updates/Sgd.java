package weka.lasagne.updates;

import java.util.Enumeration;

import weka.classifiers.functions.LasagneNet;
import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.nonlinearities.NonLinearity;

public class Sgd extends Update {

	private static final long serialVersionUID = -1679083807696737507L;

	@Override
	public String getOutputString() {
		return String.format( "sgd(%s, %s, learning_rate=%f)", "loss", "all_params", getLearningRate() );
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
	}

}
