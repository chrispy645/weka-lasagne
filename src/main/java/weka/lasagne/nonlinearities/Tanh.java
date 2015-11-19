package weka.lasagne.nonlinearities;

import java.util.Enumeration;

import weka.core.Option;

public class Tanh implements NonLinearity {

	private static final long serialVersionUID = 4844562499377455810L;

	@Override
	public String getOutputString() {
		return "tanh";
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
