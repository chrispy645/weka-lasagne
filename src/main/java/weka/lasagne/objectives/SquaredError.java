package weka.lasagne.objectives;

/**
 * Computes the element-wise squared difference between two tensors.
 * @author cjb60
 */
public class SquaredError implements Objective {

	private static final long serialVersionUID = 6121088552637601621L;

	@Override
	public String getOutputString() {
		return "squared_error(prediction, y)";
	}
	
	public String globalInfo() {
		return "Computes the element-wise squared difference between two tensors.";
	}

}
