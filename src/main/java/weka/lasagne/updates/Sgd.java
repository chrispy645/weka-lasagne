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
		StringBuilder sb = new StringBuilder();
		sb.append("kw[\"update\"] = sgd; ");
		sb.append(String.format("kw[\"update_learning_rate\"] = %f;", getLearningRate()));
		return sb.toString();
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
