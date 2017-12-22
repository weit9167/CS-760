import java.util.*;
import weka.core.*;
import java.io.*;
import weka.core.converters.ArffLoader.ArffReader;


public class HW3 {
	
	public static void main(String[] args) {
		if (args.length < 4) {
			System.out.println("usage: HW3 <trainfile> <num_folds> <learning_rate> <num_epochs>");
			System.exit(-1);
		}
		/*
		Double[][] test = new Double[3][3];
		for (int k = 0; k < 2; k++) {
			initializeHiddenWeights(test);
			for (int i = 0; i < test.length; i++) {
				for (int j = 0; j < test[0].length; j++) {
					System.out.println(test[i][j]);
				}
			}
		}*/

		
		// reading the train file
		BufferedReader trainReader = null;
		try {
			trainReader = new BufferedReader(new FileReader(args[0]));
		} catch (IOException ex) {
			System.out.println("IOException thrown by BufferedReader");
			System.exit(-1);
		}
		ArffReader arffTrain = null;
		try {
			arffTrain = new ArffReader(trainReader);
		} catch (IOException ex) {
			System.out.println("IOException thrown by ArffReader");
			System.exit(-1);
		}
		Instances trainData = arffTrain.getData();

		// reading the number of folds
		int numFolds = Integer.valueOf(args[1]);
		if (numFolds < 1 || numFolds > trainData.numInstances()) {
			System.out.println("num_folds needs to be an integer that at least 1 and no larger than the number of total instances in given training set");
			System.exit(-1);
		}

		// reading the learning rate
		double learnRate = Double.parseDouble(args[2]);
		if (learnRate > 1 || learnRate <= 0) {
			System.out.println("Incorrect value for learning rate");
			System.exit(-1);
		}

		// reading the number of epochs
		int numEpochs = Integer.valueOf(args[3]);
		if (numEpochs < 0) {
			System.out.println("num_epochs needs to be an non-negative integer");
			System.exit(-1);
		}

		// catch the number of attributes (which is the number of nodes - 1 in input layer and hidden layer, 1 is for bias)
		int numAttr = trainData.numAttributes()-1;

		HashMap<Integer, FoldPredConf> map = new HashMap<Integer, FoldPredConf>();
		// Perform n-fold stratified sampleing cross validation
		ArrayList<ArrayList<InstanceIndexPair>> folds = new ArrayList<>();
		String firstLabelForTrain = trainData.attribute(trainData.numAttributes()-1).value(0);

		stratifyCV(trainData, folds, numFolds);
		// if fold_num ==1, train all and test all (not quite reasonable)
		if (numFolds == 1) {
			Double[][] hiddenWeights = new Double[numAttr][numAttr+1];
			Double[] outputWeights = new Double[numAttr+1];
			// initialize outputWeights
			initializeWeights(outputWeights, hiddenWeights);
			ArrayList<Instance> subTrain = makeSubTraining(folds, 100);
			NNImpl nn = new NNImpl(firstLabelForTrain, subTrain, numAttr+1, learnRate, numEpochs, hiddenWeights, outputWeights);
			nn.train();
			ArrayList<InstanceIndexPair> testSet = new ArrayList<InstanceIndexPair>(folds.get(0));
			nn.test(testSet, map, 0);
		}
		
		
		for (int j = 0; j < numFolds; j++) {
			// make weights vectors
			// Assigning the weight
			// since the hidden layer has the same amount of nodes as input layer, construct 2-d array to hold the weight
			Double[][] hiddenWeights = new Double[numAttr][numAttr+1];
			// since only one output node, 1-d array is enough
			Double[] outputWeights = new Double[numAttr+1];
			// initialize outputWeights
			initializeWeights(outputWeights, hiddenWeights);

			// get training set of this fold
			ArrayList<Instance> subTrain = makeSubTraining(folds, j);
			NNImpl nn = new NNImpl(firstLabelForTrain, subTrain, numAttr+1, learnRate, numEpochs, hiddenWeights, outputWeights);
			nn.train();

			// do the test on the left-out part
			ArrayList<InstanceIndexPair> testSet = new ArrayList<InstanceIndexPair>(folds.get(j));
			nn.test(testSet, map, j);

		}

		// used for part B
		//int correctCount = 0;
		// print the test result
		for (int i = 0; i < trainData.numInstances(); i++) {
			FoldPredConf fpc = map.get(i);
			System.out.print(fpc.getFold() + " ");
			System.out.print(fpc.getPredLabel() + " ");
			System.out.print(trainData.instance(i).stringValue(trainData.instance(i).numAttributes()-1) + " ");
			System.out.printf("%.6f\n", fpc.getConf());
			// for part B-1
			//if (fpc.getPredLabel().equals(trainData.instance(i).stringValue(trainData.instance(i).numAttributes()-1))) correctCount++;
		}

		// used for part B
		//double accuracy = (double)correctCount/(double)trainData.numInstances();
		//System.out.print("The accuracy is: ");
		//System.out.printf("%.6f\n", accuracy);

		// ROC
		/*
		// 1. sort the instances based on confidence
		HashMap<Integer, Double> rocMap = new HashMap<Integer, Double>();
		for (int i = 0; i < trainData.numInstances(); i++) {
			rocMap.put(i, map.get(i).getConf());
		}
		ArrayList<HashMap.Entry<Integer, Double>> rocList = new ArrayList<HashMap.Entry<Integer, Double>>(rocMap.entrySet());
		Collections.sort(rocList, new Comparator<Map.Entry<Integer, Double>>() {
			@Override
			public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
				return (o1.getValue()).compareTo(o2.getValue());
			}
		});
		int numFirst, numSecond;
		numFirst = countNumPos(firstLabelForTrain, trainData);
		numSecond = trainData.numInstances()-numFirst;
		int truePos = 0, falsePos = 0;
		int lastTP = 0;
		int i = 0;
		double fpRate = 0.0, tpRate = 0.0;
		HashMap.Entry<Integer, Double> lastEntry = rocList.get(0);
		for (HashMap.Entry<Integer, Double> entry:rocList) {
			if ((i>0) && (map.get(entry.getKey()).getPredLabel().equals(map.get(lastEntry.getKey()).getPredLabel())) 
				&& (trainData.instance(entry.getKey()).stringValue(numAttr).equals(firstLabelForTrain)) && 
				(truePos>lastTP)) {
					fpRate = (double)falsePos/numFirst;
					tpRate = (double)truePos/numSecond;
					System.out.println(fpRate + ", " + tpRate);
					lastTP = truePos;
			}
			if (!trainData.instance(entry.getKey()).stringValue(numAttr).equals(firstLabelForTrain)) {
				++truePos;
			} else {
				++falsePos;
			}
			fpRate = (double)falsePos/numFirst;
			tpRate = (double)truePos/numSecond;
			System.out.println(fpRate + ", " + tpRate);
			// sortedROCMap.put(entry.getKey(), entry.getValue());
			//System.out.println("key is: " + entry.getKey() + "   value is: " + entry.getValue());
			i++;
		}
		*/
		
	}

	private static int countNumPos(String firstLabel, Instances trainData) {
		int result = 0;
		for (int i = 0; i < trainData.numInstances(); i++) {
			if (trainData.instance(i).stringValue(trainData.numAttributes()-1).equals(firstLabel)) result++;
		}
		return result;
	}

	private static void stratifyCV(Instances trainingSet, ArrayList<ArrayList<InstanceIndexPair>> folds, int numFolds) {
		// stratified sampling
		// 1. get labels
		String firstLabel = trainingSet.attribute(trainingSet.numAttributes()-1).value(0);
		String secondLabel = trainingSet.attribute(trainingSet.numAttributes()-1).value(1);
		// 2. stratify instances by class label
		ArrayList<Integer> firstLabelIndex = new ArrayList<Integer>();
		ArrayList<Integer> secondLabelIndex = new ArrayList<Integer>();
		ArrayList<InstanceIndexPair> firstLabelSubInst = getSubInstances(firstLabel, firstLabelIndex, trainingSet);
		ArrayList<InstanceIndexPair> secondLabelSubInst = getSubInstances(secondLabel, secondLabelIndex, trainingSet);
		// 2.5 shuffle the index
		Collections.shuffle(firstLabelIndex);
		Collections.shuffle(secondLabelIndex);
		// 3. partition the above two subsets into k folds repectively
		// just dump the extra (mod remainders into the last fold)
		int numFirstLablePerFold = firstLabelIndex.size() / numFolds;
		int numSecondLabelPerFold = secondLabelIndex.size() / numFolds;
		ArrayList<ArrayList<InstanceIndexPair>> firstLabelFold = new ArrayList<ArrayList<InstanceIndexPair>>();
		ArrayList<ArrayList<InstanceIndexPair>> secondLabelFold = new ArrayList<ArrayList<InstanceIndexPair>>();
		int firstLabelPtr = 0, secondLabelPtr = 0;
		
		for (int i = 0; i < numFolds-1; i++) {
			ArrayList<InstanceIndexPair> toAdd = new ArrayList<InstanceIndexPair>();
			for (int j = 0; j < numFirstLablePerFold; j++) {
				// System.out.println("firstLabel: " + firstLabelIndex.get(firstLabelPtr));
				toAdd.add(firstLabelSubInst.get(firstLabelIndex.get(firstLabelPtr)));
				firstLabelPtr++;
			}
			firstLabelFold.add(new ArrayList<InstanceIndexPair>(toAdd));
			toAdd = new ArrayList<InstanceIndexPair>();
			for (int j = 0; j < numSecondLabelPerFold; j++) {
				// System.out.println("secondLabel: " + secondLabelIndex.get(secondLabelPtr));
				toAdd.add(secondLabelSubInst.get(secondLabelIndex.get(secondLabelPtr)));
				secondLabelPtr++;
			}
			secondLabelFold.add(new ArrayList<InstanceIndexPair>(toAdd));
		}
		// dump the rest
		ArrayList<InstanceIndexPair> toAdd = new ArrayList<InstanceIndexPair>();
		for (;firstLabelPtr < firstLabelSubInst.size(); firstLabelPtr++) {
			// System.out.println("rest firstLabel: " + firstLabelIndex.get(firstLabelPtr));
			toAdd.add(firstLabelSubInst.get(firstLabelIndex.get(firstLabelPtr)));
		}
		firstLabelFold.add(new ArrayList<InstanceIndexPair>(toAdd));
		toAdd = new ArrayList<InstanceIndexPair>();
		for (;secondLabelPtr < secondLabelSubInst.size(); secondLabelPtr++) {
			// System.out.println("rest secondLabel: " + secondLabelIndex.get(secondLabelPtr));
			toAdd.add(secondLabelSubInst.get(secondLabelIndex.get(secondLabelPtr)));
		}
		secondLabelFold.add(new ArrayList<InstanceIndexPair>(toAdd));
		for (int i = 0; i < secondLabelFold.size(); i++) {
			toAdd = new ArrayList<InstanceIndexPair>();
			toAdd = mergeInstances(firstLabelFold.get(i), secondLabelFold.get(i));
			folds.add(new ArrayList<InstanceIndexPair>(toAdd));
		}
	}

	private static ArrayList<InstanceIndexPair> mergeInstances(ArrayList<InstanceIndexPair> in1, ArrayList<InstanceIndexPair> in2) {
		ArrayList<InstanceIndexPair> result = new ArrayList<InstanceIndexPair>(in1);
		for (InstanceIndexPair i:in2) {
			result.add(i);
		}
		return result;
	}

	private static ArrayList<Instance> makeSubTraining(ArrayList<ArrayList<InstanceIndexPair>> folds, int leaveOut) {
		//Instances result = new Instances(tdata);
		//result.delete();
		ArrayList<Instance> holder = new ArrayList<Instance>();
		for (int i = 0; i < folds.size(); i++) {
			if (i == leaveOut) continue;
			else {
				for (int j = 0; j < folds.get(i).size(); j++) {
					holder.add(folds.get(i).get(j).getInstance());
					//result.add(folds.get(i).get(j).getInstance());
				}
			}
		}
		return holder;
	}

	private static ArrayList<InstanceIndexPair> getSubInstances(String label, ArrayList<Integer> index, Instances trainingSet) {
		ArrayList<InstanceIndexPair> result = new ArrayList<InstanceIndexPair>();
		int count = 0;
		for (int i = 0; i < trainingSet.numInstances(); i++) {
			if (trainingSet.instance(i).stringValue(trainingSet.numAttributes()-1).equals(label)) {
				result.add(new InstanceIndexPair(trainingSet.instance(i), i));
				index.add(count);
				count++;
			}	
		}
		return result;
	}

	private static void initializeWeights(Double[] weightsVector, Double[][] weightsMatrix) {
		Random r = new Random();
		for (int i = 0; i < weightsVector.length; i++) {
			weightsVector[i] = r.nextDouble()*2.0-1.0;
		}

		for (int i = 0; i < weightsMatrix.length; i++) {
			for (int j = 0; j < weightsMatrix[i].length; j++) {
				weightsMatrix[i][j] = r.nextDouble()*2.0-1.0;
			}
			//weights[i][weights[i].length-1] = -r.nextDouble()*0.1;
		}
		//weights[weights.length-1] = -r.nextDouble()*0.1;
	}

}