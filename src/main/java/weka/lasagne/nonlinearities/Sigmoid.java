package weka.lasagne.nonlinearities;

import java.util.Enumeration;

import weka.core.Option;

public class Sigmoid implements NonLinearity {

	private static final long serialVersionUID = 590967905073786429L;

	@Override
	public String getOutputString() {
		return "sigmoid";
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
	
	@Override
	public String toString() {
		return "sigmoid";
	}

}
