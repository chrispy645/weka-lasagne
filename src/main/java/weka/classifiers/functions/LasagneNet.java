package weka.classifiers.functions;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.pyscript.PyScriptClassifier;
import weka.core.BatchPredictor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.lasagne.Constants;
import weka.lasagne.layers.DenseLayer;
import weka.lasagne.layers.Layer;
import weka.lasagne.objectives.CategoricalCrossEntropy;
import weka.lasagne.objectives.Objective;
import weka.lasagne.objectives.RegressionObjective;
import weka.lasagne.objectives.SquaredError;
import weka.lasagne.updates.Sgd;
import weka.lasagne.updates.Update;
import weka.nolearn.AbstractBatchIterator;
import weka.nolearn.BatchIterator;
import weka.nolearn.ImageBatchIterator;

public class LasagneNet extends RandomizableClassifier implements BatchPredictor {

	private static final long serialVersionUID = 1125617871340073201L;
	
	private float m_validSetSize = 0.0f;
	
	public void setValidSetSize(float validSetSize) {
		m_validSetSize = validSetSize;
	}
	
	public float getValidSetSize() {
		return m_validSetSize;
	}
	
	private Layer[] m_layers = new Layer[] { new DenseLayer() };
	
	public void setLayers(Layer[] layers) {
		m_layers = layers;
	}
	
	public Layer[] getLayers() {
		return m_layers;
	}
	
	private AbstractBatchIterator m_batchIterator = new BatchIterator();
	
	public AbstractBatchIterator getBatchIterator() {
		return m_batchIterator;
	}
	
	public void setBatchIterator(AbstractBatchIterator batchIterator) {
		m_batchIterator = batchIterator;
	}
	
	private static final int DEFAULT_NUM_EPOCHS = 1;
	
	private int m_numEpochs = DEFAULT_NUM_EPOCHS;
	
	public int getNumEpochs() {
		return m_numEpochs;
	}
	
	public void setNumEpochs(int numEpochs) {
		m_numEpochs = numEpochs;
	}
	
	private PyScriptClassifier m_cls = new PyScriptClassifier();
	
	/*
	 * Loss functions
	 */
	
	public static final Objective DEFAULT_LOSS_FUNCTION = new CategoricalCrossEntropy();
	
	private Objective m_lossFunction = DEFAULT_LOSS_FUNCTION;
	
	public Objective getLossFunction() {
		return m_lossFunction;
	}
	
	public void setLossFunction(Objective lossFunction) {
		m_lossFunction = lossFunction;
	}
	
	/*
	 * Updates
	 */
	
	private static final Update DEFAULT_UPDATE = new Sgd();
	
	private Update m_update = DEFAULT_UPDATE;
	
	
	public Update getUpdate() {
		return m_update;
	}
	
	public void setUpdate(Update update) {
		m_update = update;
	}
	
	/*
	 * Debugging
	 */	

	private String m_dumpScript = null;
	
	public String getDumpScript() {
		return m_dumpScript;
	}
	
	public void setDumpScript(String dumpScript) {
		m_dumpScript = dumpScript;
	}
	
	private String m_outFile = "";
	
	public String getOutFile() {
		return m_outFile;
	}
	
	public void setOutFile(String outFile) {
		m_outFile = outFile;
	}

	public static String getSpec(Object obj) {
		String result;
		if (obj == null) {
			result = "";
		} else {
			result = obj.getClass().getName();
			if (obj instanceof OptionHandler) {
				result += " "
		          + Utils.joinOptions(((OptionHandler) obj).getOptions());
			}
		}		
		return result;
	}
	
	public static Object specToObject(String str, Class<?> classType) throws Exception {	
		String[] options = Utils.splitOptions(str);
		String base = options[0];
		options[0] = "";
		return Utils.forName(classType, base, options);
	}
	
	public String getTemplate() throws Exception {
		InputStream is = this.getClass().getResourceAsStream("/learner_nolearn.py");
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		StringBuilder sb = new StringBuilder();
		while(br.ready()) {
			sb.append( br.readLine() );
			sb.append("\n");
		}
		br.close();
		return sb.toString();
	}
	
	@Override
	public String[] getOptions() {
	    Vector<String> result = new Vector<String>();
	    String[] options = super.getOptions();
	    for (int i = 0; i < options.length; i++) {
	      result.add(options[i]);
	    }
	    // layers
	    for (int i = 0; i < getLayers().length; i++) {
	      result.add("-" + Constants.LAYER);
	      result.add( getSpec(getLayers()[i]) );
	    }
	    // loss
	    result.add("-" + Constants.LOSS);
	    result.add( getSpec(getLossFunction()) );
	    // update
	    result.add("-" + Constants.UPDATE);
	    result.add( getSpec(getUpdate()) );
	    // num epochs
	    result.add("-" + Constants.NUM_EPOCHS);
	    result.add( "" + getNumEpochs() );
	    // sgd batch size
	    result.add("-" + Constants.BATCH_ITERATOR);
	    result.add( getSpec(getBatchIterator()) );
	    // valid set size
	    result.add("-" + Constants.VALID_SET_SIZE);
	    result.add( "" + getValidSetSize() );
	    // out file
	    result.add( "-" + Constants.OUT_FILE );
	    result.add( getOutFile() );
	    return result.toArray(new String[result.size()]);
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		// layers
		Vector<Layer> layers = new Vector<Layer>();
		String tmpStr = null;
		while ((tmpStr = Utils.getOption(Constants.LAYER, options)).length() != 0) {
			layers.add( (Layer) specToObject(tmpStr, Layer.class) );
		}
		if (layers.size() == 0) {
			layers.add(new DenseLayer());
		}
		setLayers( layers.toArray(new Layer[layers.size()]) );
		// loss
		tmpStr = Utils.getOption(Constants.LOSS, options);
		if(!tmpStr.equals("")) setLossFunction( (Objective) specToObject(tmpStr, Objective.class) );
		// update
		tmpStr = Utils.getOption(Constants.UPDATE, options);
		if(!tmpStr.equals("")) setUpdate( (Update) specToObject(tmpStr, Update.class) );
		// num epochs
		tmpStr = Utils.getOption(Constants.NUM_EPOCHS, options);
		if(!tmpStr.equals("")) setNumEpochs( Integer.parseInt(tmpStr) );
		// batch iterator
		tmpStr = Utils.getOption(Constants.BATCH_ITERATOR, options);
		if(!tmpStr.equals("")) setBatchIterator( (AbstractBatchIterator) specToObject(tmpStr, AbstractBatchIterator.class)  );
		// valid set size
		tmpStr = Utils.getOption(Constants.VALID_SET_SIZE, options);
		if(!tmpStr.equals("")) setValidSetSize( Float.parseFloat(tmpStr) );
		// outfile
		tmpStr = Utils.getOption(Constants.OUT_FILE, options);
		if(!tmpStr.equals("")) setOutFile(tmpStr);
	}
	
	public void checkConfiguration(Instances data) throws Exception {
		// if the problem is a regression, then the loss must be squared error
		if( data.numClasses() == 1 && !(getLossFunction() instanceof RegressionObjective) ) {
			throw new Exception("Bad loss function! Use a regression loss (such as squared error)");
		}
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		checkConfiguration(data);
		
		m_cls = new PyScriptClassifier();
		m_cls.setPrintStdOut(true);
		//m_cls.setBatchSize( getBatchSize() );
		String args = String.format("num_epochs=%d;seed=%d;batch_size=%s",
				getNumEpochs(), getSeed(), getBatchSize() );
		if(data.numClasses() == 1) {
			args = args + ";regression=1";
		} else {
			args = args + ";regression=0";
		}
		
		if( !getOutFile().equals("") ) {
			args = args + ";out_file=" + "'" + new File(getOutFile()).getAbsolutePath() + "'";
		}
		m_cls.setArguments(args);
		m_cls.setSaveScript(true);
		
		File tmpFile = File.createTempFile("script", ".py");
		PrintWriter pw = new PrintWriter(tmpFile);
		String code = getOutputString(data);
		pw.write(code);
		pw.flush();
		pw.close();
		//System.out.println(code);
		
		if( getDumpScript() != null ) {
			pw = new PrintWriter( new File(getDumpScript()) );
			pw.write( getOutputString(data) );
			pw.flush();
			pw.close();
		}
		
		if(getDebug()) {
			System.err.println(code);
		}
		
		m_cls.setPythonFile(tmpFile);
		m_cls.buildClassifier(data);
		
		//System.out.println(cls.getPickledModel());
		
		//System.exit(0);
	}
	
	@Override
	public double[][] distributionsForInstances(Instances data) throws Exception {
		return m_cls.distributionsForInstances(data);
	}
	
	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {
		return m_cls.distributionForInstance(inst);
	}

	public String getOutputString(Instances data) throws Exception {
		String template = getTemplate();		
		String tab = "    ";
		// construct the layers
		StringBuilder layerString = new StringBuilder();
		Layer[] layers = getLayers();
		// build the layer_conf string
		layerString.append("kw = {}\n");
		layerString.append(String.format("%slayer_conf = [\n", tab));
		int counter = 1;
		layerString.append( String.format("%s%s%s,\n", tab, tab, "(\"input\", InputLayer)") );
		for(Layer layer : layers) {
			layer.setName("layer" + counter);
			counter++;
			String tp0 = layer.getName();
			String tp1 = layer.getClassName();
			String tuple = String.format("(%s, %s)", "\"" + tp0 + "\"", tp1);
			layerString.append( String.format("%s%s%s,\n", tab, tab, tuple ) );
		}
		layerString.append( String.format("%s%s%s,\n", tab, tab, "(\"output\", DenseLayer)") );
		layerString.append(String.format("%s]\n", tab));
		layerString.append(String.format("%skw[\"layers\"] = layer_conf\n", tab));
				
		// make sure images get the right input shape
		layerString.append( String.format("%s%s\n", tab, getBatchIterator().getOutputString()) );
		// keywords for layers
		for(Layer layer : layers) {
			layerString.append( String.format("%s%s\n", tab, layer.getOutputString() ));
		}
		// output layer
		layerString.append(String.format("%skw[\"output_nonlinearity\"] = linear if args[\"regression\"] else softmax\n", tab));
		layerString.append(String.format("%skw[\"output_num_units\"] = %d\n", tab, data.numClasses()));
		// batch iterators
		BatchIterator testIterator = new BatchIterator();
		testIterator.setBatchSize( Integer.parseInt(getBatchSize()) );
		layerString.append(String.format("%s%s\n", tab, getBatchIterator().getOutputString() ));
		//System.err.println(getBatchIterator().getOutputString());
		//layerString.append(String.format("%skw[\"batch_iterator_test\"] = %s\n", tab, testIterator.getOutputString()));
		// loss function
		layerString.append(String.format("%s%s\n", tab, getLossFunction().getOutputString()));
		// loss function aux
		//layerString.append(String.format("%s%s"))
		// is it a regression
		layerString.append(String.format("%skw[\"regression\"] = args[\"regression\"]\n", tab));
		// updates
		layerString.append(String.format("%s%s\n", tab, getUpdate().getOutputString()));
		// epochs
		layerString.append(String.format("%skw[\"max_epochs\"] = %d\n", tab, getNumEpochs() ));
		// verbose
		layerString.append(String.format("%skw[\"verbose\"] = 1\n", tab));
		// train split
		layerString.append(String.format("%skw[\"train_split\"] = TrainSplit(eval_size=%f)\n", tab, getValidSetSize()));
		// on epoch finished handler
		if(!getOutFile().equals("")) {
			layerString.append(String.format("%skw[\"on_epoch_finished\"] = [save_stats_at_every(1, %s)]\n", tab, "\"" + getOutFile() + "\"" ));
		}
		// create the net
		layerString.append(String.format("%snet = NeuralNet(**kw)\n", tab));
		layerString.append(String.format("%sreturn net", tab));

		template = template.replace("##GET_NET##", layerString.toString());
		
		return template;
		
	}
	
	@Override
	public String toString() {
		if(m_cls == null) return null;
		else return m_cls.getModelString();
	}
	
	@Override
	public boolean implementsMoreEfficientBatchPrediction() {
		return true;
	}
	
	public static void main(String[] argv) {
		runClassifier(new LasagneNet(), argv);
	}

}
