package weka.nolearn;

public class ShufflingBatchIterator extends BatchIterator {
	
	private static final long serialVersionUID = 6925053542009337892L;

	@Override
	public String getOutputString() {
		return String.format("ShufflingBatchIterator(batch_size=%d)", getBatchSize());
	}
	
	public String globalInfo() {
		return "Batch iterator that randomly shuffles the data before every epoch.";
	}

}
