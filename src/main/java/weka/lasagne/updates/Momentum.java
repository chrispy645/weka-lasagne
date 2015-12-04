package weka.lasagne.updates;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;

public class Momentum extends Update {

	private static final long serialVersionUID = -7180508681020311794L;

	public static double DEFAULT_MOMENTUM = 0.9;
	
	private double m_momentum = DEFAULT_MOMENTUM;
	
	public double getMomentum() {
		return m_momentum;
	}
	
	public void setMomentum(double momentum) {
		m_momentum = momentum;
	}
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append("kw[\"update\"] = momentum; ");
		sb.append(String.format("kw[\"update_learning_rate\"] = %f; ", getLearningRate()));
		sb.append(String.format("kw[\"update_momentum\"] = %f;", getMomentum()));
		return sb.toString();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}
	
	@Override
	public String[] getOptions() {
		String[] tmp = super.getOptions();
		Vector<String> result = new Vector<String>();
		for(int x = 0; x < tmp.length; x++) {
			result.add(tmp[x]);
		}
		result.add("-" + Constants.MOMENTUM);
		result.add("" + getMomentum());
		return result.toArray( new String[result.size()] );
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String tmp = Utils.getOption(Constants.MOMENTUM, options);
		if(!tmp.equals("")) setMomentum( Double.parseDouble(tmp) );
	}

}
