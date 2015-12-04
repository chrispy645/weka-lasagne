package weka.lasagne.layers;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;

/**
 * Dropout layer. Sets values to zero with probability p.
 * @author cjb60
 */
public class DropoutLayer extends Layer {
	
	private static final long serialVersionUID = -5960114193635163469L;
	
	@Override
	public String getClassName() {
		return "DropoutLayer";
	}

	public static final double DEFAULT_P = 0.5;
	
	private double m_p = DEFAULT_P;
	
	public double getP() {
		return m_p;
	}
	
	public void setP(double p) {
		m_p = p;
	}

	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("kw[\"%s_%s\"] = %d", getName(), "p", getP() ) );
		return sb.toString();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.P, options);
		if(!tmp.equals("")) setP( Double.parseDouble(tmp) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-" + Constants.P);
		result.add( "" + getP() );
		return result.toArray(new String[result.size()]);
	}

}
