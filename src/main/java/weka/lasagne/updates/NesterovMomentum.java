package weka.lasagne.updates;

public class NesterovMomentum extends Momentum {

	private static final long serialVersionUID = 6774163157438966678L;

	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append("kw[\"update\"] = nesterov_momentum; ");
		sb.append(String.format("kw[\"update_learning_rate\"] = %f; ", getLearningRate()));
		sb.append(String.format("kw[\"update_momentum\"] = %f;", getMomentum()));
		return sb.toString();
	}

}
