package weka.lasagne.objectives;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.Returnable;

public abstract class Objective implements Returnable, OptionHandler {
	
	private static final long serialVersionUID = 8282033740442572536L;
	
	public double m_l1 = 0.0;
	public double m_l2 = 0.0;
	
	public void setL2(double l2) {
		m_l2 = l2;
	}
	
	public void setL1(double l1) {
		m_l1 = l1;
	}
	
	public double getL1() {
		return m_l1;
	}
	
	public double getL2() {
		return m_l2;
	}
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("kw[\"objective_l1\"] = %f; ", getL1() ) );
		sb.append(String.format("kw[\"objective_l2\"] = %f", getL2() ) );
		return sb.toString();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.L1, options);
		if(!tmp.equals("")) setL1( Double.parseDouble(tmp) );
		tmp = Utils.getOption(Constants.L2, options);
		if(!tmp.equals("")) setL2( Double.parseDouble(tmp) );
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add( "-" + Constants.L1 );
		result.add( "" + getL1() );
		result.add( "-" + Constants.L2 );
		result.add( "" + getL2() );
		return result.toArray(new String[result.size()]);
	}	
	
}
