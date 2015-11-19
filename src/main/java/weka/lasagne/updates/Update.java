package weka.lasagne.updates;

import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.Returnable;

public abstract class Update implements Returnable, OptionHandler {
	
	private static final long serialVersionUID = 3488106242996090270L;

	private static final double DEFAULT_LEARNING_RATE = 0.01;
	
	private double m_learningRate = DEFAULT_LEARNING_RATE;
	
	public double getLearningRate() {
		return m_learningRate;
	}
	
	public void setLearningRate(double learningRate) {
		m_learningRate = learningRate;
	}
	
	@Override
	public String[] getOptions() {
		return new String[] { "-" + Constants.LEARNING_RATE, "" + getLearningRate() };
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.LEARNING_RATE, options);
		if(!tmp.equals("")) setLearningRate( Double.parseDouble(tmp) );
	}
}
