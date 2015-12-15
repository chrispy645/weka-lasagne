package weka.nolearn;

import java.util.Vector;

import weka.core.Utils;
import weka.lasagne.Constants;

public class ImageBatchIterator extends BatchIterator {

	private static final long serialVersionUID = -3749614159512883260L;
	
	private int m_width = 0;
	private int m_height = 0;
	private String m_prefix = "\"\"";
	
	private boolean m_isRgb = false;
	
	public boolean getIsRgb() {
		return m_isRgb;
	}
	
	public void setIsRgb(boolean isRgb) {
		m_isRgb = isRgb;
	}
	
	public int getWidth() {
		return m_width;
	}
	
	public int getHeight() {
		return m_height;
	}
	
	public String getPrefix() {
		return m_prefix;
	}
	
	public void setWidth(int width) {
		m_width = width;
	}
	
	public void setHeight(int height) {
		m_height = height;
	}
	
	public void setPrefix(String prefix) {
		m_prefix = "\"" + prefix + "\"";
	}

	@Override
	public String getOutputString() {
		StringBuilder sb = new StringBuilder();
		if(!getIsRgb()) sb.append(String.format("kw[\"input_shape\"] = (None, 1, %d, %d); ", getWidth(), getHeight()));
		else sb.append(String.format("kw[\"input_shape\"] = (None, 3, %d, %d); ", getWidth(), getHeight()));
		
		sb.append("filenames = args[\"attr_values\"][args[\"attributes\"][0]]; ");
		sb.append(String.format(
				"kw[\"batch_iterator_train\"] = ImageBatchIterator(filenames, %s, batch_size=%d); ", getPrefix(), getBatchSize()));
		sb.append(String.format(
				"kw[\"batch_iterator_test\"] = ImageBatchIterator(filenames, %s, batch_size=args[\"batch_size\"])", getPrefix()));
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
		result.add("-" + Constants.PREFIX);
		result.add("" + getPrefix());
		if(getIsRgb()) result.add( "-" + Constants.RGB);
		return result.toArray( new String[result.size()] );
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String tmp = Utils.getOption(Constants.WIDTH, options);
		if(!tmp.equals("")) setWidth( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.HEIGHT, options);
		if(!tmp.equals("")) setHeight( Integer.parseInt(tmp) );
		tmp = Utils.getOption(Constants.PREFIX, options);
		if(!tmp.equals("")) setPrefix(tmp);
		setIsRgb( Utils.getFlag(Constants.RGB, options) );
	}
	
	public String globalInfo() {
		return "A batch iterator that expects a string attribute (in addition to the class)"
				+ " that specifies the filenames of the images. The iterator, before every"
				+ " epoch, will load a batch of images (specified by the filenames) for training.";
	}
	
}
