package weka.lasagne.objectives;

import weka.lasagne.objectives.Objective;

public class AutoLoss extends Objective {
	
	private static final long serialVersionUID = 4159114208513688897L;
	
	@Override
	public String getOutputString() {
		String base = super.getOutputString();
		return base + ";kw[\"objective_loss_function\"] = squared_error if args[\"regression\"] else categorical_crossentropy";
	}
}