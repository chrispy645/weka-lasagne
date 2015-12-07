package weka.nolearn;

public class ShufflingBatchIterator extends BatchIterator {
	
	private static final long serialVersionUID = 6925053542009337892L;

	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append("kw[\"input_shape\"] = (None, 1, len(args[\"attributes\"])-1); ");
		sb.append(String.format("kw[\"batch_iterator_train\"] = ShufflingBatchIterator(batch_size=%d); ", getBatchSize()));
		sb.append("kw[\"batch_iterator_test\"] = BatchIterator(batch_size=args[\"batch_size\"])");
		return sb.toString();
	}
	
	public String globalInfo() {
		return "Batch iterator that randomly shuffles the data before every epoch.";
	}

}
