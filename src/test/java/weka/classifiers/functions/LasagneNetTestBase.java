package weka.classifiers.functions;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;
import weka.lasagne.objectives.AutoLoss;

public class LasagneNetTestBase extends AbstractClassifierTest {

	public LasagneNetTestBase(String name) {
		super(name);
	}

	@Override
	public Classifier getClassifier() {
		LasagneNet net = new LasagneNet();
		net.setLossFunction(new AutoLoss());
		net.setDebug(false);
		return net;
	}

}
