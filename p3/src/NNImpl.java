import java.util.*;
import weka.core.*;

public class NNImpl {
	public ArrayList<Node> inputLayer = null;
	public ArrayList<Node> hiddenLayer = null;
	public Node outputNode = null;
	public ArrayList<Instance> trainingSet = null;
	public String firstLabel = "";

	private int batchSize = 2;

	private Double learningRate = 0.1;
	private int maxEpoch = 1;

	public NNImpl (String firstLabel, ArrayList<Instance> trainingSet, int numNodesInputHidden, Double learningRate, int maxEpoch, Double[][] hiddenWeights, Double[] outputWeights) {
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;
		this.firstLabel = firstLabel;

		// make input layer
		this.inputLayer = new ArrayList<Node>();
		// add regular input nodes
		for (int i = 0; i < numNodesInputHidden-1; i++) {
			Node node = new Node(0);
			this.inputLayer.add(node);
		}
		// add bias node
		Node biasToHidden = new Node(1);
		this.inputLayer.add(biasToHidden);

		// make hidden layer
		this.hiddenLayer = new ArrayList<Node>();
		// add regular hidden nodes
		for (int i = 0; i < numNodesInputHidden-1; i++) {
			Node node = new Node(2);
			for (int j = 0; j < numNodesInputHidden; j++) {
				NodeWeightPair nwp = new NodeWeightPair(this.inputLayer.get(j), hiddenWeights[i][j]);
				//System.out.println(hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			this.hiddenLayer.add(node);
		}
		// add bias node
		Node biasToOutput = new Node(3);
		this.hiddenLayer.add(biasToOutput);

		// Make ouput layer
		this.outputNode = new Node(4);
		for (int i = 0; i < outputWeights.length; i++) {
			NodeWeightPair nwp = new NodeWeightPair(this.hiddenLayer.get(i), outputWeights[i]);
			this.outputNode.parents.add(nwp);
		}

		
	}

	public void test(ArrayList<InstanceIndexPair> testSet, HashMap<Integer,FoldPredConf> map, int foldIndex) {
		for (int j = 0; j < testSet.size(); j++) {
			// make predict
			for (int i = 0; i < this.inputLayer.size()-1; i++) {
				inputLayer.get(i).setInput(testSet.get(j).getInstance().value(i));
			}
			for (int i = 0; i < this.hiddenLayer.size()-1; i++) {
				hiddenLayer.get(i).setInput(0.0);
				hiddenLayer.get(i).calculateOutput();
			}
			outputNode.setInput(0.0);
			outputNode.calculateOutput();
			Double outValue = outputNode.getOutput();
			String pred;
			if (outValue < 0.5) { // take the first label
				pred = testSet.get(0).getInstance().attribute(testSet.get(0).getInstance().numAttributes()-1).value(0);
			} else {
				pred = testSet.get(0).getInstance().attribute(testSet.get(0).getInstance().numAttributes()-1).value(1);
			}
			FoldPredConf fpc = new FoldPredConf(foldIndex, pred, outValue);
			
			map.put(testSet.get(j).getIndex(), fpc);
		}
	}

	public void train() {
		int numEpoch = 0;
		while (numEpoch < this.maxEpoch) {
			// use SGD: update all weights after batchSize instances
			int count = 0; 
			// delta weight from hiddent to output
			Double[] deltaWeightHtoO = new Double[hiddenLayer.size()];
			initializeDoubleArray(deltaWeightHtoO);
			// delta weight from input to hidden
			Double[][] deltaWeightItoH  = new Double[inputLayer.size()][hiddenLayer.size()-1];
			initializeDoubleMatrix(deltaWeightItoH);
			// for each epoch, shuffle the training set
			Collections.shuffle(this.trainingSet);
			for (Instance insta:this.trainingSet) {
				Double teacherOutput = getTeacherOutput(this.firstLabel, insta);
				// Forward pass: calculate the output
				for (int i = 0; i < this.inputLayer.size()-1; i++) {
					inputLayer.get(i).setInput(insta.value(i));
					//System.out.println(inputLayer.get(i).getOutput());
				}
				for (int i = 0; i < this.hiddenLayer.size()-1; i++) {
					hiddenLayer.get(i).setInput(0.0);
					hiddenLayer.get(i).calculateOutput();
				}
				outputNode.setInput(0.0);
				outputNode.calculateOutput();
				Double outValue = outputNode.getOutput();

				// System.out.printf("%.6f\n", outValue);
				// backward pass
				for (int i = 0; i < hiddenLayer.size(); i++) {
					// Loss function: Cross Entropy
					deltaWeightHtoO[i] += -learningRate*hiddenLayer.get(i).getOutput()*(outValue - teacherOutput);
					// System.out.format("hiddenlayer %d, learning rate = %f, hiddenValue = %f, err = %f\n", i, learningRate, hiddenLayer.get(i).getOutput(),(outValue - teacherOutput));
					// Loss Function: MSE
					// deltaWeightHtoO[i] += learningRate*hiddenLayer.get(i).getOutput()*(teacherOutput-outValue)*outValue*(1.0-outValue);
				}
				for (int i = 0; i < inputLayer.size(); i++) {
					for (int j = 0; j < hiddenLayer.size()-1; j++) {
						// Loss Function: Cross Entropy
						deltaWeightItoH[i][j] += -learningRate*(outValue-teacherOutput)*
												(outputNode.parents.get(j).weight)*
												(hiddenLayer.get(j).getOutput()*(1.0-hiddenLayer.get(j).getOutput()))*
												inputLayer.get(i).getOutput();
						// Loss Function: MSE
						//deltaWeightItoH[i][j] += learningRate*inputLayer.get(i).getOutput()*
						//						hiddenLayer.get(j).getOutput()*(1.0-hiddenLayer.get(i).getOutput())*
						//						(1.0-outValue)*outValue*(teacherOutput-outValue)*outputNode.parents.get(j).weight;
					}
				}
				count++;
				if (count == batchSize) { // do the update
					// update on hidden to output weight
					for (int i = 0; i < hiddenLayer.size(); i++) {
						//System.out.format("Updated value: hidden %d, delta = %f\n",i, deltaWeightHtoO[i]);
						outputNode.parents.get(i).weight += deltaWeightHtoO[i]/(double)batchSize;
						//System.out.format("w%d = %f\n", i, outputNode.parents.get(i).weight);
					}
					// update on input to hidden weight
					for (int i = 0; i < inputLayer.size(); i++) {
						for (int j = 0; j < hiddenLayer.size()-1; j++) {
							hiddenLayer.get(j).parents.get(i).weight += deltaWeightItoH[i][j]/(double)batchSize;
							//System.out.format("w (%d %d) = %f\n", j, i, hiddenLayer.get(j).parents.get(i).weight);
						}
					}
					// reset the batch instances storage
					count = 0;
					initializeDoubleArray(deltaWeightHtoO);
					initializeDoubleMatrix(deltaWeightItoH);
				}
			}
			if (count != 0) { // which means there are some instances not yet been used
				for (int i = 0; i < hiddenLayer.size(); i++) {
					outputNode.parents.get(i).weight += deltaWeightHtoO[i]/(double)count;
				}
				// update on input to hidden weight
				for (int i = 0; i < inputLayer.size(); i++) {
					for (int j = 0; j < hiddenLayer.size()-1; j++) {
						hiddenLayer.get(j).parents.get(i).weight += deltaWeightItoH[i][j]/(double)count;
					}
				}				
			}
			numEpoch++;
		}
	}

	private void initializeDoubleMatrix(Double[][] m) {
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[0].length; j++) {
				m[i][j] = 0.0;
			}
		}
	}

	private void initializeDoubleArray(Double[] dArray) {
		for (int i = 0; i < dArray.length; i++) {
			dArray[i] = 0.0;
		}
	}

	private Double getTeacherOutput(String firstLabel, Instance insta) {
		if (insta.stringValue(insta.numAttributes()-1).equals(firstLabel)) {
			return 0.0;
		} else {
			return 1.0;
		}
	}
}