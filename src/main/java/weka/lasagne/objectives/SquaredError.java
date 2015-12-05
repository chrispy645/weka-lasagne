package weka.lasagne.objectives;

/**
 * Computes the element-wise squared difference between two tensors.
 * @author cjb60
 */
public class SquaredError extends Objective implements RegressionObjective {

	private static final long serialVersionUID = 6121088552637601621L;

	@Override
	public String getOutputString() {
		String base = super.getOutputString();
		return base + ";kw[\"objective_loss_function\"] = squared_error";
	}
	
	public String globalInfo() {
		return "Computes the element-wise squared difference between two tensors.";
	}

}
