package weka.lasagne.updates;

public class NesterovMomentum extends Momentum {

	private static final long serialVersionUID = 6774163157438966678L;

	@Override
	public String getOutputString() {
		return String.format( "nesterov_momentum(%s, %s, learning_rate=%f, momentum=%f)", "loss", "all_params",
				getLearningRate(), getMomentum() );
	}

}
