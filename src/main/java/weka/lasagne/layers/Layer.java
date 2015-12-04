package weka.lasagne.layers;

import weka.core.OptionHandler;
import weka.lasagne.Returnable;

public abstract class Layer implements Returnable, OptionHandler {
	
	private static final long serialVersionUID = 4134419933250851585L;
	
	public abstract String getClassName();
	
	protected String m_name = null;
	
	public String getName() {
		return m_name;
	}
	
	public void setName(String name) {
		m_name = name;
	}

}
