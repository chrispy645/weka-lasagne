package weka.lasagne.updates;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;

/**
 * Adagrad updates. Scale learning rates by dividing with the square root of accumulated squared gradients.
 * See <a href="http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf">this</a> for further description.
 * @author cjb60
 */
public class Adagrad extends Update {

	private static final long serialVersionUID = -4960565404498185386L;
	
	private static final double DEFAULT_EPSILON = 1e-6;
	
	private double m_epsilon = DEFAULT_EPSILON;
	
	public String globalInfo() {
		return "Adagrad updates. Scale learning rates by dividing with the square root of accumulated squared gradients";
	}
	
	public double getEpsilon() {
		return m_epsilon;
	}
	
	public void setEpsilon(double epsilon) {
		m_epsilon = epsilon;
	}
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append("kw[\"update\"] = adagrad; ");
		sb.append(String.format("kw[\"update_learning_rate\"] = %f; ", getLearningRate()));
		sb.append(String.format("kw[\"update_epsilon\"] = %f;", getEpsilon()));
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
		result.add("-" + Constants.EPSILON);
		result.add("" + getEpsilon());
		return result.toArray( new String[result.size()] );
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String tmp = Utils.getOption(Constants.EPSILON, options);
		if(!tmp.equals("")) setEpsilon( Double.parseDouble(tmp) );
	}

}
