package weka.lasagne.objectives;

public class CategoricalCrossEntropy implements Objective {

	private static final long serialVersionUID = -3976404263265270704L;

	@Override
	public String getOutputString() {
		return "categorical_crossentropy(prediction, y)";
	}

}
