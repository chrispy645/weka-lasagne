package weka.nolearn;

import java.util.Vector;

import weka.core.OptionHandler;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.Returnable;

public abstract class AbstractBatchIterator implements Returnable, OptionHandler {

	private static final long serialVersionUID = -6493863123860372186L;
	
	private boolean m_shuffle = false;
	
	public boolean getShuffle() {
		return m_shuffle;
	}
	
	public void setShuffle(boolean shuffle) {
		m_shuffle = shuffle;
	}
	
	private int m_batchSize = 128;
	
	public int getBatchSize() {
		return m_batchSize;
	}
	
	public void setBatchSize(int batchSize) {
		m_batchSize = batchSize;
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		setShuffle( Utils.getFlag(Constants.SHUFFLE, options) );
		String tmp = Utils.getOption(Constants.SGD_BATCH_SIZE, options);
		setBatchSize( Integer.parseInt(tmp) );
	}

	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		if( getShuffle() ) {
			result.add("-" + Constants.SHUFFLE);
		}
		result.add("-" + Constants.SGD_BATCH_SIZE);
		result.add("" + getBatchSize());
		return result.toArray( new String[result.size()] );
	}

}
