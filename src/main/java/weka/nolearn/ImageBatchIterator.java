package weka.nolearn;

public class ImageBatchIterator extends BatchIterator {

	private static final long serialVersionUID = -3749614159512883260L;

	@Override
	public String getOutputString() {
		return String.format("ImageBatchIterator(batch_size=%d)", getBatchSize());
	}
	
	public String globalInfo() {
		return "A batch iterator that expects a string attribute (in addition to the class)"
				+ " that specifies the filenames of the images. The iterator, before every"
				+ " epoch, will load a batch of images (specified by the filenames) for training.";
	}
	
}
