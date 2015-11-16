package weka.lasagne.updates;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Utils;

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
		// momentum(loss_or_grads, params, learning_rate, momentum=0.9)
		return String.format( "momentum(%s, %s, learning_rate=%f, momentum=%f)", "loss", "all_params", getLearningRate(), getMomentum() );
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
		result.add("-m");
		result.add("" + getMomentum());
		return result.toArray( new String[result.size()] );
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String tmp = Utils.getOption('m', options);
		setMomentum( Double.parseDouble(tmp) );
	}

}
