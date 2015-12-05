package weka.lasagne.layers;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;

public class MaxPool2DLayer extends Layer {
	
	private static final long serialVersionUID = -3463740636960768980L;
	
	private int m_poolSizeX = 0;
	private int m_poolSizeY = 0;
	
	public int getPoolSizeX() {
		return m_poolSizeX;
	}
	
	public int getPoolSizeY() {
		return m_poolSizeY;
	}
	
	public void setPoolSizeX(int poolSizeX) {
		m_poolSizeX = poolSizeX;
	}
	
	public void setPoolSizeY(int poolSizeY) {
		m_poolSizeY = poolSizeY;
	}

	@Override
	public String getOutputString() {
		return String.format("kw[\"%s_pool_size\"] = (%d,%d)", getName(), getPoolSizeX(), getPoolSizeY());
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-" + Constants.POOL_SIZE_X);
		result.add("" + getPoolSizeX());
		result.add("-" + Constants.POOL_SIZE_Y);
		result.add("" + getPoolSizeY());		
		return result.toArray( new String[result.size()] );
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.POOL_SIZE_X, options);
		if(!tmp.equals("")) setPoolSizeX( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.POOL_SIZE_Y, options);
		if(!tmp.equals("")) setPoolSizeY( Integer.parseInt(tmp) );
	}

	@Override
	public String getClassName() {
		return "MaxPool2DLayer";
	}

}
