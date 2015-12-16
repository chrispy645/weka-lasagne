package weka.nolearn;

import java.util.Enumeration;
import weka.core.Option;

public class BatchIterator extends AbstractBatchIterator {

	private static final long serialVersionUID = -4453353872418278852L;
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append("kw[\"input_shape\"] = (None, 1, len(args[\"attributes\"])-1); ");
		sb.append(String.format(
				"kw[\"batch_iterator_train\"] = BatchIterator(shuffle=%d, batch_size=%d); ", getShuffle() ? 1 : 0, getBatchSize()));
		sb.append(String.format(
				"kw[\"batch_iterator_test\"] = BatchIterator(shuffle=0, batch_size=args[\"batch_size\"])"));
		return sb.toString();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}
	
	public String globalInfo() {
		return "Default batch iterator.";
	}

}
