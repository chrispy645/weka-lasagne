package weka.lasagne.layers;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.functions.LasagneNet;
import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.nonlinearities.NonLinearity;
import weka.lasagne.nonlinearities.Sigmoid;

/**
 * A fully-connected layer.
 * @author cjb60
 */
public class DenseLayer extends Layer {
	
	private static final long serialVersionUID = -4158355845708293518L;
	
	@Override
	public String getClassName() {
		return "DenseLayer";
	}
	
	public static final NonLinearity DEFAULT_NONLINEARITY = new Sigmoid();
	public static final int DEFAULT_NUM_UNITS = 1;
	
	private int m_numUnits = DEFAULT_NUM_UNITS;
	
	public int getNumUnits() {
		return m_numUnits;
	}
	
	public void setNumUnits(int numUnits) {
		m_numUnits = numUnits;
	}
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("kw[\"%s_%s\"] = %d; ", getName(), "num_units", getNumUnits() ) );
		sb.append(String.format("kw[\"%s_%s\"] = %s", getName(), "nonlinearity", getNonLinearity().getOutputString() ) );
		return sb.toString();
	}
	
	private NonLinearity m_nonLinearity = DEFAULT_NONLINEARITY;
	
	public NonLinearity getNonLinearity() {
		return m_nonLinearity;
	}
	
	public void setNonLinearity(NonLinearity nonLinearity) {
		m_nonLinearity = nonLinearity;
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.NUM_UNITS, options);
		if(!tmp.equals("")) setNumUnits( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.NON_LINEARITY, options);
		if(!tmp.equals("")) setNonLinearity( (NonLinearity) LasagneNet.specToObject(tmp, NonLinearity.class) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-" + Constants.NUM_UNITS);
		result.add( "" + getNumUnits() );
		result.add("-" + Constants.NON_LINEARITY);
		result.add( "" + LasagneNet.getSpec(getNonLinearity()) );
	    return result.toArray(new String[result.size()]);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("DenseLayer\\n");
		sb.append( String.format("  %s = %d\\n", Constants.NUM_UNITS, getNumUnits()) );
		sb.append( String.format("  %s = %s\\n", Constants.NON_LINEARITY, getNonLinearity().toString()) );
		return sb.toString();
	}

}
