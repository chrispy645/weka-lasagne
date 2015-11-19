package weka.lasagne.nonlinearities;

import java.util.Enumeration;

import weka.core.Option;

public class Softmax implements NonLinearity {

	private static final long serialVersionUID = 9070641680257518953L;

	@Override
	public String getOutputString() {
		return "softmax";
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
	}

	@Override
	public String[] getOptions() {
		return new String[] { };
	}

}
