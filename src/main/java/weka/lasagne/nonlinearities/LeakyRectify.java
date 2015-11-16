package weka.lasagne.nonlinearities;

import java.util.Enumeration;

import weka.core.Option;
import weka.core.Utils;

public class LeakyRectify implements NonLinearity {

	private static final long serialVersionUID = 413521184290226994L;
	
	public final double DEFAULT_LEAKINESS = 0.01;
	
	private double m_leakiness = DEFAULT_LEAKINESS;
	
	public double getLeakiness() {
		return m_leakiness;
	}
	
	public void setLeakiness(double leakiness) {
		m_leakiness = leakiness;
	}
	
	@Override
	public String getOutputString() {
		return String.format("LeakyRectify(leakiness=%f)", m_leakiness);
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption('l', options);
		setLeakiness( Double.parseDouble(tmp) );
	}

	@Override
	public String[] getOptions() {
		return new String[] {"-l", "" + getLeakiness() };
	}

}
