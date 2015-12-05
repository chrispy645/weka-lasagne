package weka.lasagne.objectives;

/**
 * Computes the binary cross-entropy between predictions and targets.
 * @author cjb60
 *
 */
public class BinaryCrossEntropy extends Objective {

	private static final long serialVersionUID = -3976404263265270704L;

	@Override
	public String getOutputString() {
		String base = super.getOutputString();
		return base + ";kw[\"objective_loss_function\"] = binary_crossentropy";
	}
	
	public String globalInfo() {
		return "Computes the binary cross-entropy between predictions and targets.";
	}

}
