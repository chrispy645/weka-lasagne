package weka.lasagne.layers;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.functions.LasagneNet;
import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.nonlinearities.NonLinearity;
import weka.lasagne.nonlinearities.Rectify;

public class NonLinearityLayer extends Layer {

	private static final long serialVersionUID = -5641193155799249411L;
	
	private static NonLinearity DEFAULT_NONLINEARITY = new Rectify();
	private NonLinearity m_nonlinearity = DEFAULT_NONLINEARITY;

	public NonLinearity getNonLinearity() {
		return m_nonlinearity;
	}
	
	public void setNonLinearity(NonLinearity nonlinearity) {
		m_nonlinearity = nonlinearity;
	}
	
	@Override
	public String getOutputString() {
		return String.format("NonlinearityLayer(l_prev, %s)", getNonLinearity().getOutputString());
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.NON_LINEARITY, options);
		if(!tmp.equals("")) setNonLinearity( (NonLinearity) LasagneNet.specToObject(tmp, NonLinearity.class));
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-" + Constants.NON_LINEARITY);
		result.add( LasagneNet.getSpec(getNonLinearity()) );
		return result.toArray(new String[result.size()]);
	}

}
