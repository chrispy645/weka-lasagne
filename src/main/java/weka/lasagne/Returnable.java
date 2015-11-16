package weka.lasagne;

import java.io.Serializable;

public interface Returnable extends Serializable {
	
	/**
	 * Get the representation of this object in the form
	 * of Python code.
	 * @return
	 */
	public String getOutputString();

}
