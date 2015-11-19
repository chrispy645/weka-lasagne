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
 * 1D convolutional layer Performs a 1D convolution on its input and optionally adds 
 * a bias and applies an elementwise nonlinearity.
 * @author cjb60
 */
public class Conv1DLayer extends Layer {
	
	private static final long serialVersionUID = -5748457426809180995L;

	public static final int DEFAULT_NUM_FILTERS = 10;
	
	private int m_numFilters = DEFAULT_NUM_FILTERS;
	
	public int getNumFilters() {
		return m_numFilters;
	}
	
	public void setNumFilters(int numFilters) {
		m_numFilters = numFilters;
	}
	
	public static final int DEFAULT_FILTER_SIZE = 1;
	
	private int m_filterSize = DEFAULT_FILTER_SIZE;
	
	public int getFilterSize() {
		return m_filterSize;
	}
	
	public void setFilterSize(int filterSize) {
		m_filterSize = filterSize;
	}
	
	public static final NonLinearity DEFAULT_NON_LINEARITY = new Sigmoid();
	
	private NonLinearity m_nonLinearity = DEFAULT_NON_LINEARITY;
	
	public NonLinearity getNonlinearity() {
		return m_nonLinearity;
	}
	
	public void setNonlinearity(NonLinearity nonLinearity) {
		m_nonLinearity = nonLinearity;
	}

	@Override
	public String getOutputString() {
		return String.format(
				"Conv1DLayer(l_prev, num_filters=%d, filter_size=%d, nonlinearity=%s)",
				getNumFilters(), getFilterSize(), getNonlinearity().getOutputString()
		);	
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.FILTER_SIZE, options);
		setFilterSize( Integer.parseInt(tmp) );
		
		tmp = Utils.getOption(Constants.NUM_FILTERS, options);
		setNumFilters( Integer.parseInt(tmp) );
		
		tmp = Utils.getOption(Constants.NON_LINEARITY, options);
		setNonlinearity( (NonLinearity) LasagneNet.specToObject(tmp, NonLinearity.class) );
		
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add( "-" + Constants.FILTER_SIZE);
		result.add( "" + getFilterSize() );
		result.add("-" + Constants.NUM_FILTERS);
		result.add( "" + getNumFilters() );
		result.add("-" + Constants.NON_LINEARITY);
		result.add( "" + LasagneNet.getSpec(getNonlinearity()) );
		return result.toArray(new String[result.size()]);
	}	

}
