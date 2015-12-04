package weka.lasagne.layers;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;

/**
 * Gaussian noise layer. Add zero-mean Gaussian noise of given standard deviation to the input.
 * See <a href="https://clgiles.ist.psu.edu/papers/IEEE.TNN.synaptic.noise.recurrent.nets.pdf">here</a>.
 * @author cjb60
 */
public class GaussianNoiseLayer extends Layer {

	private static final long serialVersionUID = -585452878729099259L;
	
	@Override
	public String getClassName() {
		return "GaussianNoiseLayer";
	}

	public static double DEFAULT_SIGMA = 0.1;
	
	public double m_sigma = DEFAULT_SIGMA;
	
	public double getSigma() {
		return m_sigma;
	}
	
	public void setSigma(double sigma) {
		m_sigma = sigma;
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.SIGMA, options);
		if(!tmp.equals("")) setSigma( Double.parseDouble(tmp) );
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-" + Constants.SIGMA);
		result.add( "" + getSigma() );
		return result.toArray(new String[result.size()]);
	}
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("kw[\"%s_%s\"] = %d", getName(), "sigma", getSigma() ) );
		return sb.toString();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("GaussianNoiseLayer\\n");
		sb.append( String.format("  %s = %s\\n", Constants.SIGMA, getSigma()) );
		return sb.toString();
	}

}
