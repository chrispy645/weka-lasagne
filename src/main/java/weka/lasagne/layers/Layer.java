package weka.lasagne.layers;

import weka.classifiers.functions.LasagneNet;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.lasagne.Returnable;
import weka.lasagne.nonlinearities.NonLinearity;
import weka.lasagne.nonlinearities.Sigmoid;

public abstract class Layer implements Returnable, OptionHandler {
	
	private static final long serialVersionUID = 4134419933250851585L;

	public final NonLinearity DEFAULT_NONLINEARITY = new Sigmoid();
	
	private NonLinearity m_nonLinearity = DEFAULT_NONLINEARITY;
	
	public NonLinearity getNonLinearity() {
		return m_nonLinearity;
	}
	
	public void setNonLinearity(NonLinearity nonLinearity) {
		m_nonLinearity = nonLinearity;
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String nonLinearityStr = Utils.getOption('n', options);
		String[] nonLinearityOptions = Utils.splitOptions(nonLinearityStr);
		String nonLinearity = nonLinearityOptions[0];
		nonLinearityOptions[0] = "";
		setNonLinearity( (NonLinearity) Utils.forName(NonLinearity.class, nonLinearity, nonLinearityOptions));
	}
	
	@Override
	public String[] getOptions() {
		return new String[] { "-n", "" + LasagneNet.getSpec( getNonLinearity() ) };
	}

}
