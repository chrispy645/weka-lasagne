package weka.nolearn;

public class BasicRotateImageBatchIterator extends ImageBatchIterator {
	
	private static final long serialVersionUID = -5921135724072877674L;

	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		if( !getIsRgb() ) {
			sb.append(String.format("kw[\"input_shape\"] = (None, 1, %d, %d); ", getWidth(), getHeight()));
		}
		else {
			sb.append(String.format("kw[\"input_shape\"] = (None, 3, %d, %d); ", getWidth(), getHeight()));
		}	
		sb.append("filenames = args[\"attr_values\"][args[\"attributes\"][0]]; ");
		sb.append(String.format(
				"kw[\"batch_iterator_train\"] = BasicRotateImageBatchIterator(filenames=filenames, prefix=%s, shuffle=%d, batch_size=%d); ",
				getPrefix(), getShuffle() ? 1 : 0, getBatchSize()));
		sb.append(String.format(
				"kw[\"batch_iterator_test\"] = BasicRotateImageBatchIterator(filenames=filenames, prefix=%s, shuffle=0, batch_size=args[\"batch_size\"])",
				getPrefix()));
		return sb.toString();
	}

}
