package weka.lasagne.objectives;

public class AbsoluteError extends Objective implements RegressionObjective {

	private static final long serialVersionUID = 1400587888155602L;

	@Override
	public String getOutputString() {
		String base = super.getOutputString();
		return base + ";kw[\"objective_loss_function\"] = abs_error";
	}
	
	public String globalInfo() {
		return "Computes the element-wise absolute difference between two tensors.";
	}

}
