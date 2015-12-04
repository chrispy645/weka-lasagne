package weka.lasagne.objectives;

/**
 * Computes the binary cross-entropy between predictions and targets.
 * @author cjb60
 *
 */
public class BinaryCrossEntropy implements Objective {

	private static final long serialVersionUID = -3976404263265270704L;

	@Override
	public String getOutputString() {
		return "binary_crossentropy";
	}
	
	public String globalInfo() {
		return "Computes the binary cross-entropy between predictions and targets.";
	}

}
