package weka.lasagne.nonlinearities;

import java.util.Enumeration;

import weka.core.Option;
import weka.core.Utils;

public class Rectify implements NonLinearity {
	
	private static final long serialVersionUID = -3624647127669425993L;

	@Override
	public String getOutputString() {
		return "rectify";
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
