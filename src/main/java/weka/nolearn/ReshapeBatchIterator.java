package weka.nolearn;

import java.util.Vector;

import weka.core.Utils;
import weka.lasagne.Constants;

public class ReshapeBatchIterator extends BatchIterator {

	private static final long serialVersionUID = -1230637488753353706L;
	
	private int m_width = 0;
	private int m_height = 0;
	
	public int getWidth() {
		return m_width;
	}
	
	public int getHeight() {
		return m_height;
	}
	
	public void setWidth(int width) {
		m_width = width;
	}
	
	public void setHeight(int height) {
		m_height = height;
	}
	
	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format(
				"kw[\"batch_iterator_train\"] = ReshapeBatchIterator((%d,%d), batch_size=%d); ",
				getWidth(), getHeight(), getBatchSize()));
		sb.append(String.format(
				"kw[\"batch_iterator_test\"] = ReshapeBatchIterator((%d,%d), batch_size=%d); ",
				getWidth(), getHeight(), getBatchSize()));
		return sb.toString();
	}

	@Override
	public String[] getOptions() {
		String[] tmp = super.getOptions();
		Vector<String> result = new Vector<String>();
		for(int x = 0; x < tmp.length; x++) {
			result.add(tmp[x]);
		}
		result.add("-" + Constants.WIDTH);
		result.add("" + getWidth());
		result.add("-" + Constants.HEIGHT);
		result.add("" + getHeight());
		return result.toArray( new String[result.size()] );
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String tmp = Utils.getOption(Constants.WIDTH, options);
		if(!tmp.equals("")) setWidth( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.HEIGHT, options);
		if(!tmp.equals("")) setHeight( Integer.parseInt(tmp) );
	}

	public String globalInfo() {
		return "A batch iterator that reshapes data before it is put into a batch. This"
				+ "can then be used with layers that operate on 2D inputs, e.g. Conv2DLayer"
				+ "and MaxPool2DLayer";
	}	

}
