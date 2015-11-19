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
import weka.classifiers.pyscript.PyScriptClassifier;
import weka.core.BatchPredictor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.lasagne.layers.DenseLayer;
import weka.lasagne.layers.Layer;
import weka.lasagne.objectives.CategoricalCrossEntropy;
import weka.lasagne.objectives.Objective;
import weka.lasagne.updates.Sgd;
import weka.lasagne.updates.Update;

public class LasagneNet extends AbstractClassifier implements BatchPredictor {

	private static final long serialVersionUID = 1125617871340073201L;
	
	private Layer[] m_layers = new Layer[] { new DenseLayer() };
	
	public void setLayers(Layer[] layers) {
		m_layers = layers;
	}
	
	public Layer[] getLayers() {
		return m_layers;
	}
	
	private static final int DEFAULT_SGD_BATCH_SIZE = 1;
	
	private int m_sgdBatchSize = DEFAULT_SGD_BATCH_SIZE;
	
	public int getSgdBatchSize() {
		return m_sgdBatchSize;
	}
	
	public void setSgdBatchSize(int sgdBatchSize) {
		m_sgdBatchSize = sgdBatchSize;
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
		InputStream is = this.getClass().getResourceAsStream("/learner.py");
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
	    for (int i = 0; i < getLayers().length; i++) {
	      result.add("-L");
	      result.add( getSpec(getLayers()[i]) );
	    }
	    return result.toArray(new String[result.size()]);
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		Vector<Layer> layers = new Vector<Layer>();
		String tmpStr = null;
		String layer;
		String[] options2;
		while ((tmpStr = Utils.getOption("L", options)).length() != 0) {
			//options2 = Utils.splitOptions(tmpStr);
			//layer = options2[0];
			//options2[0] = "";
			//layers.add((Layer) Utils.forName(Layer.class, layer, options2));
			
			layers.add( (Layer) specToObject(tmpStr, Layer.class) );
		}
		if (layers.size() == 0) {
			layers.add(new DenseLayer());
		}
		setLayers( layers.toArray(new Layer[layers.size()]) );
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		m_cls = new PyScriptClassifier();
		m_cls.setPrintStdOut(true);
		m_cls.setArguments( String.format("num_epochs=%d;batch_size=%d", getNumEpochs(), getSgdBatchSize() ) );
		
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
		layerString.append( String.format("in_layer = InputLayer( (None, len(args[\"attributes\"])-1) )\n", data.numAttributes()-1) );
		int hiddenLayers = 1;
		String lastLayerName = "in_layer";
		for(Layer layer : layers) {
			layerString.append( String.format("%sl_prev = %s\n", tab, lastLayerName));
			layerString.append( String.format("%s%s = output_shapes.append(l_prev.output_shape)\n", tab, lastLayerName) );
			String thisLayerName = String.format("hidden%d", hiddenLayers);
			layerString.append( String.format("%s%s = %s\n", tab, thisLayerName, layer.getOutputString()) );
			lastLayerName = thisLayerName;
			hiddenLayers++;
		}
		layerString.append( String.format("%sprev_layer = %s\n", tab, lastLayerName) );
		layerString.append( String.format("%s%s = output_shapes.append(l_prev.output_shape)\n", tab, lastLayerName) );
		
		String a = "linear";
		if(data.numClasses() > 1) {
			a = "softmax";
		}
		layerString.append( String.format("%sout_layer = DenseLayer(l_prev, num_units=%d, nonlinearity=%s)\n", tab, data.numClasses(), a) );
		template = template.replace("##NETWORK##", layerString.toString());
		
		// construct the loss
		StringBuilder lossString = new StringBuilder();
		lossString.append( String.format("loss = %s\n", getLossFunction().getOutputString() ) );
		lossString.append( String.format("%sloss = loss.mean()\n", tab) );
		template = template.replace("##LOSS##", lossString.toString());
		
		// construct the updates
		StringBuilder updateString = new StringBuilder();
		updateString.append( String.format("updates = %s", getUpdate().getOutputString()) );
		template = template.replace("##UPDATES##", updateString.toString());
		
		// describe string
		//StringBuilder describeString = new StringBuilder();
		//describeString.append("desc.append('model_description')\n");
		//for(Layer layer : layers) {
		//	describeString.append(String.format("%sdesc.append(%s)\n", tab, layer.toString()));
		//}
		//describeString.append( "\n.join([str(x) for x in model[0]])\n");
		
		//template = template.replace("##DESCRIBE##", describeString.toString());
		template = template.replace("##DESCRIBE##", "#text");
		
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
