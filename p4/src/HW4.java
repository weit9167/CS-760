import java.util.*;
import weka.core.*;
import java.io.*;
import weka.core.converters.ArffLoader.ArffReader;

public class HW4 {

	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("usage: bayes <train-set-file> <test-set-file> <n|t>");
			System.exit(-1);
		}

		BufferedReader trainReader = null;
		try {
			trainReader = new BufferedReader(new FileReader(args[0]));
		} catch (IOException ex) {
			System.out.println("IOException thrown by BufferedReader when reading in training data");
			System.exit(-1);
		}
		ArffReader arffTrain = null;
		try {
			arffTrain = new ArffReader(trainReader);
		} catch (IOException ex) {
			System.out.println("IOException thrown by ArffReader when reading in training data");
			System.exit(-1);
		}
		Instances trainData = arffTrain.getData();

		BufferedReader testReader = null;
		try {
			testReader = new BufferedReader(new FileReader(args[1]));
		} catch (IOException ex) {
			System.out.println("IOException thrown by BufferedReader when reading in test data");
			System.exit(-1);
		}
		ArffReader arffTest = null;
		try {
			arffTest = new ArffReader(testReader);
		} catch (IOException ex) {
			System.out.println("IOException thrown by ArffReader when reading in test data");
			System.exit(-1);
		}
		Instances testData = arffTest.getData();

		// Part1
		
		if (args[2].equals("n")) {
			// do naive bayes
			NaiveBayesImpl nb = new NaiveBayesImpl(trainData);
			nb.train();
			for (int i = 0; i < testData.numAttributes()-1; i++) {
				System.out.println(testData.attribute(i).name() + " " + testData.attribute(testData.numAttributes()-1).name());
			}
			System.out.print("\n");
			nb.test(testData);
		} else if (args[2].equals("t")) {
			TANImpl tanb = new TANImpl(trainData);
			tanb.train();
			tanb.test(testData);
		} else {
			System.out.println("Incorrect mode character: the mode must be one of \'n\' and \'t\'");
			System.exit(-1);
		}
		/*
		// Part2
		//  use stratified 10-fold cross validation on the chess-KingRookVKingPawn.arff data set to compare naive Bayes and TAN
		ArrayList<ArrayList<Instance>> folds = new ArrayList<>();
		stratifyCV(trainData, folds, 10);
		double[] nbAccu = new double[10];
		double[] tanAccu = new double[10];
		for (int i = 0; i < 10; i++) {
			// make sub Training data
			Instances subTrain = makeSubTraining(folds, i, trainData);
			Instances testSet = makeSubTest(folds.get(i), trainData);
			// naive bayes
			NaiveBayesImpl nb = new NaiveBayesImpl(subTrain);
			nb.train();
			// do test on  the left-out part
			System.out.println("Current fold index: " + i);
			int nbCorrectCount = nb.test(testSet);
			nbAccu[i] = ((double)nbCorrectCount)/((double)testSet.size());

			// TAN
			TANImpl tanb = new TANImpl(subTrain);
			tanb.train();
			System.out.println("Current fold index: " + i);
			int tanCorrectCount = tanb.test(testSet);
			tanAccu[i] = ((double)tanCorrectCount)/((double)testSet.size());
		}
		System.out.println("The 10 accuracies calculated from Naive Bayes:");
		for (int i = 0; i < 10; i++) {
			System.out.printf("%.12f\n", nbAccu[i]);
		}
		System.out.println("The 10 accuracies calculated from TAN:");
		for (int i = 0; i < 10; i++) {
			System.out.printf("%.12f\n", tanAccu[i]);
		}*/
	}
	/*
	private static Instances makeSubTraining(ArrayList<ArrayList<Instance>> folds, int leaveOut, Instances tData) {
		Instances result = new Instances(tData);
		result.delete();
		//ArrayList<Instance> holder = new ArrayList<Instance>();
		for (int i = 0; i < folds.size(); i++) {
			if (i == leaveOut) continue;
			else {
				for (int j = 0; j < folds.get(i).size(); j++) {
					result.add(folds.get(i).get(j));
				}
			}
		}
		return result;
	}
	
	private static Instances makeSubTest(ArrayList<Instance> part, Instances tData) {
		Instances result = new Instances(tData);
		result.delete();
		for (int i = 0; i < part.size(); i++) {
			result.add(part.get(i));
		}
		return result;
	}
	
	private static void stratifyCV(Instances trainingSet, ArrayList<ArrayList<Instance>> folds, int numFolds) {
		// stratified sampling
		// 1. get labels
		String firstLabel = trainingSet.attribute(trainingSet.numAttributes()-1).value(0);
		String secondLabel = trainingSet.attribute(trainingSet.numAttributes()-1).value(1);
		// 2. stratify instances by class label
		ArrayList<Integer> firstLabelIndex = new ArrayList<Integer>();
		ArrayList<Integer> secondLabelIndex = new ArrayList<Integer>();
		ArrayList<Instance> firstLabelSubInst = getSubInstances(firstLabel, firstLabelIndex, trainingSet);
		ArrayList<Instance> secondLabelSubInst = getSubInstances(secondLabel, secondLabelIndex, trainingSet);
		// 2.5 shuffle the index
		Collections.shuffle(firstLabelIndex);
		Collections.shuffle(secondLabelIndex);
		// 3. partition the above two subsets into k folds repectively
		// just dump the extra (mod remainders into the last fold)
		int numFirstLablePerFold = firstLabelIndex.size() / numFolds;
		int numSecondLabelPerFold = secondLabelIndex.size() / numFolds;
		ArrayList<ArrayList<Instance>> firstLabelFold = new ArrayList<ArrayList<Instance>>();
		ArrayList<ArrayList<Instance>> secondLabelFold = new ArrayList<ArrayList<Instance>>();
		int firstLabelPtr = 0, secondLabelPtr = 0;
		
		for (int i = 0; i < numFolds-1; i++) {
			ArrayList<Instance> toAdd = new ArrayList<Instance>();
			for (int j = 0; j < numFirstLablePerFold; j++) {
				// System.out.println("firstLabel: " + firstLabelIndex.get(firstLabelPtr));
				toAdd.add(firstLabelSubInst.get(firstLabelIndex.get(firstLabelPtr)));
				firstLabelPtr++;
			}
			firstLabelFold.add(new ArrayList<Instance>(toAdd));
			toAdd = new ArrayList<Instance>();
			for (int j = 0; j < numSecondLabelPerFold; j++) {
				// System.out.println("secondLabel: " + secondLabelIndex.get(secondLabelPtr));
				toAdd.add(secondLabelSubInst.get(secondLabelIndex.get(secondLabelPtr)));
				secondLabelPtr++;
			}
			secondLabelFold.add(new ArrayList<Instance>(toAdd));
		}
		// dump the rest
		ArrayList<Instance> toAdd = new ArrayList<Instance>();
		for (;firstLabelPtr < firstLabelSubInst.size(); firstLabelPtr++) {
			// System.out.println("rest firstLabel: " + firstLabelIndex.get(firstLabelPtr));
			toAdd.add(firstLabelSubInst.get(firstLabelIndex.get(firstLabelPtr)));
		}
		firstLabelFold.add(new ArrayList<Instance>(toAdd));
		toAdd = new ArrayList<Instance>();
		for (;secondLabelPtr < secondLabelSubInst.size(); secondLabelPtr++) {
			// System.out.println("rest secondLabel: " + secondLabelIndex.get(secondLabelPtr));
			toAdd.add(secondLabelSubInst.get(secondLabelIndex.get(secondLabelPtr)));
		}
		secondLabelFold.add(new ArrayList<Instance>(toAdd));
		for (int i = 0; i < secondLabelFold.size(); i++) {
			toAdd = new ArrayList<Instance>();
			toAdd = mergeInstances(firstLabelFold.get(i), secondLabelFold.get(i));
			folds.add(new ArrayList<Instance>(toAdd));
		}
	}

	private static ArrayList<Instance> getSubInstances(String label, ArrayList<Integer> index, Instances trainingSet) {
		ArrayList<Instance> result = new ArrayList<Instance>();
		int count = 0;
		for (int i = 0; i < trainingSet.numInstances(); i++) {
			if (trainingSet.instance(i).stringValue(trainingSet.numAttributes()-1).equals(label)) {
				result.add(trainingSet.instance(i));
				index.add(count);
				count++;
			}	
		}
		return result;
	}

	private static ArrayList<Instance> mergeInstances(ArrayList<Instance> in1, ArrayList<Instance> in2) {
		ArrayList<Instance> result = new ArrayList<Instance>(in1);
		for (Instance i:in2) {
			result.add(i);
		}
		return result;
	}
	*/

}