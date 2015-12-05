package weka.nolearn;

import java.util.Enumeration;

import weka.core.Option;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.Returnable;

public class BatchIterator extends AbstractBatchIterator {

	private static final long serialVersionUID = -4453353872418278852L;

	public static final int DEFAULT_BATCH_SIZE = 128;
	
	private int m_batchSize = DEFAULT_BATCH_SIZE;
	
	public int getBatchSize() {
		return m_batchSize;
	}
	
	public void setBatchSize(int batchSize) {
		m_batchSize = batchSize;
	}
	
	@Override
	public String getOutputString() {
		return String.format("BatchIterator(batch_size=%d)", getBatchSize());
	}
	
	@Override
	public String[] getOptions() {
		return new String[] { "-" + Constants.SGD_BATCH_SIZE, "" + getBatchSize() };
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption(Constants.SGD_BATCH_SIZE, options);
		if(!tmp.equals("")) setBatchSize( Integer.parseInt(tmp) );
	}

	@Override
	public Enumeration<Option> listOptions() {
		return null;
	}
	
	public String globalInfo() {
		return "Default batch iterator.";
	}

}
