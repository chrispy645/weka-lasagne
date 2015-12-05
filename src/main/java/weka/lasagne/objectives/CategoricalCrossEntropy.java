package weka.lasagne.objectives;

/**
 * Computes the categorical cross-entropy between predictions and targets.
 * @author cjb60
 */
public class CategoricalCrossEntropy extends Objective {

	private static final long serialVersionUID = -3976404263265270704L;

	@Override
	public String getOutputString() {
		String base = super.getOutputString();
		return base + "; kw[\"objective_loss_function\"] = categorical_crossentropy";
	}
	
	public String globalInfo() {
		return "Computes the categorical cross-entropy between predictions and targets.";
	}

}
