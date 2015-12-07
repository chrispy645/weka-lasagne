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
 * 2D convolutional layer. Performs a 2D convolution on its input and optionally adds
 * a bias and applies an elementwise nonlinearity.
 * @author cjb60
 */
public class Conv2DLayer extends Layer {

	private static final long serialVersionUID = -4243409354666807743L;
	
	public static final int DEFAULT_NUM_FILTERS = 10;
	
	private int m_numFilters = DEFAULT_NUM_FILTERS;
	
	public int getNumFilters() {
		return m_numFilters;
	}
	
	public void setNumFilters(int numFilters) {
		if(numFilters <= 0) throw new IllegalArgumentException("Number of filters cannot be <= 0!");
		m_numFilters = numFilters;
	}
	
	public static final int DEFAULT_FILTER_SIZE = 1;
	
	private int m_filterSizeX = DEFAULT_FILTER_SIZE;
	
	public int getFilterSizeX() {
		return m_filterSizeX;
	}
	
	public void setFilterSizeX(int filterSizeX) {
		if(filterSizeX <= 0) throw new IllegalArgumentException("X filter size cannot be <= 0!");
		m_filterSizeX = filterSizeX;
	}
	
	private int m_filterSizeY = DEFAULT_FILTER_SIZE;
	
	public int getFilterSizeY() {
		return m_filterSizeY;
	}
	
	public void setFilterSizeY(int filterSizeY) {
		if(filterSizeY <= 0) throw new IllegalArgumentException("Y filter size cannot be <= 0!");
		m_filterSizeY = filterSizeY;
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
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("kw[\"%s_%s\"] = %d; ", getName(), Constants.NUM_FILTERS, getNumFilters() ) );
		sb.append(String.format("kw[\"%s_%s\"] = (%d,%d); ", getName(), Constants.FILTER_SIZE, getFilterSizeX(), getFilterSizeY() ) );
		sb.append(String.format("kw[\"%s_%s\"] = %s", getName(), Constants.NON_LINEARITY, getNonlinearity().getOutputString() ) );
		return sb.toString();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.FILTER_SIZE_X, options);
		if(!tmp.equals("")) setFilterSizeX( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.FILTER_SIZE_Y, options);
		if(!tmp.equals("")) setFilterSizeY( Integer.parseInt(tmp) );		
		tmp = Utils.getOption(Constants.NUM_FILTERS, options);
		if(!tmp.equals("")) setNumFilters( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.NON_LINEARITY, options);
		if(!tmp.equals("")) setNonlinearity( (NonLinearity) LasagneNet.specToObject(tmp, NonLinearity.class) );	
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add( "-" + Constants.FILTER_SIZE_X);
		result.add( "" + getFilterSizeX() );
		result.add( "-" + Constants.FILTER_SIZE_Y);
		result.add( "" + getFilterSizeY() );
		result.add("-" + Constants.NUM_FILTERS);
		result.add( "" + getNumFilters() );
		result.add("-" + Constants.NON_LINEARITY);
		result.add( "" + LasagneNet.getSpec(getNonlinearity()) );
		return result.toArray(new String[result.size()]);
	}

	@Override
	public String getClassName() {
		return "Conv2DLayer";
	}
	
}
