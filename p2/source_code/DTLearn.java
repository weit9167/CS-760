import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.FileReader;
import java.io.StringReader;
import weka.core.converters.ArffLoader.ArffReader;

import weka.core.*;
import java.util.*;

// libaraies used for part 2 & 3
import java.io.BufferedWriter;
import java.io.PrintWriter;


public class DTLearn {

	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("usage: dt-learn <train-set-file> <test-set-file> m");
			System.exit(-1);
		}
		// load the training data
		BufferedReader trainReader = null;
		try {
			trainReader = new BufferedReader(new FileReader(args[0]));
		} catch (IOException ex) {
			System.err.println("IOException thrown by BufferedReader");
			System.exit(-1);
		}
		ArffReader arffTrain = null;
		try {
			arffTrain = new ArffReader(trainReader);
		} catch (IOException ex) {
			System.err.println("IOException thrown by ArffReader");
			System.exit(-1);
		}
		Instances trainData = arffTrain.getData();
		// load the test data
		BufferedReader testReader = null;
		try {
			testReader = new BufferedReader(new FileReader(args[1]));
		} catch (IOException ex) {
			System.err.println("IOException thrown by BufferedReader");
			System.exit(-1);
		}
		ArffReader arffTest = null;
		try {
			arffTest = new ArffReader(testReader);
		} catch (IOException ex) {
			System.err.println("IOException thrown by ArffReader");
			System.exit(-1);
		}
		Instances testData = arffTest.getData();

		// special case: m > number of training instance
		if (Integer.valueOf(args[2]) > trainData.numInstances()) {
			String common = commonLabel(trainData);
			special(testData, common);
			return;
		}

		// Part 1
		// make the DT
		DTTrain trainning = new DTTrain(trainData, Integer.valueOf(args[2]));

		// print the DT
		DTNode root = trainning.treeRoot;
		
		for (DTNode rootBranch:root.children) {
			preOrderPrinting(rootBranch, 0, root.numInstances);
		}

		// do the test
		System.out.println("<Predictions for the Test Set Instances>");
		
		int numCorrect = 0;
		int index = 0;
		for (Instance insta:testData) {
			if (test(insta, root, ++index))
				numCorrect++;
		}
		System.out.print("Number of correctly classified: " + numCorrect 
			+ " Total number of test instances: " + testData.numInstances());



		/**
		 * Part 2 Calculation: remove the comment symbol when using
		 */
		/*
		try {
			PrintWriter file2 = new PrintWriter("part2.txt", "UTF-8");
			int totalInsta = trainData.numInstances();
			int persent5 = totalInsta/20;
			int persent10 = totalInsta/10;
			int persent20 = totalInsta/5;
			int persent50 = totalInsta/2;
			// for each persentage, generate the corresponding 10 trainning sets and compute the avg accuracy
			file2.println("5% accuracies:");
			part2Cal(trainData, testData, persent5, file2);
			file2.println("10% accuracies:");
			part2Cal(trainData, testData, persent10, file2);
			file2.println("20% accuracies:");
			part2Cal(trainData, testData, persent20, file2);
			file2.println("50% accuracies:");
			part2Cal(trainData, testData, persent50, file2);
			file2.println("100% accuracy:");
			DTTrain subTraining100 = new DTTrain(trainData, 4); // m = 4 for part 2
			DTNode subRoot100 = subTraining100.treeRoot;
			int correct100 = 0;
			for (Instance insta:testData) {
				if (testWithoutPrint(insta, subRoot100)) correct100++;
			}
			file2.println((double)correct100/(double)testData.numInstances());
			file2.close();
		} catch (IOException ex) {
			System.err.println("IOException thrown by PrintWriter");
			System.exit(-1);
		}
		// Part 2 end 
		*/

		/**
		 * Part 3 Calculation: remove the comment symbol when using
		 */
		/*
		try {
			PrintWriter file3 = new PrintWriter("part3.txt", "UTF-8");
			int correctM2 = 0;
			int correctM5 = 0;
			int correctM10 = 0;
			int correctM20 = 0;
			DTTrain trainM2 = new DTTrain(trainData, 2);
			for (Instance insta:testData) {
				if (testWithoutPrint(insta, trainM2.treeRoot)) correctM2++;
			}
			file3.println("M=2 accuracy: " + (double)correctM2/(double)testData.numInstances());
			DTTrain trainM5 = new DTTrain(trainData, 5);
			for (Instance insta:testData) {
				if (testWithoutPrint(insta, trainM5.treeRoot)) correctM5++;
			}
			file3.println("M=5 accuracy: " + (double)correctM5/(double)testData.numInstances());
			DTTrain trainM10 = new DTTrain(trainData, 10);
			for (Instance insta:testData) {
				if (testWithoutPrint(insta, trainM10.treeRoot)) correctM10++;
			}
			file3.println("M=10 accuracy: " + (double)correctM10/(double)testData.numInstances());
			DTTrain trainM20 = new DTTrain(trainData, 20);
			for (Instance insta:testData) {
				if (testWithoutPrint(insta, trainM20.treeRoot)) correctM20++;
			}
			file3.println("M=20 accuracy: " + (double)correctM20/(double)testData.numInstances());
			file3.close();
		} catch (IOException ex) {
			System.err.println("IOException thrown by PrintWriter");
			System.exit(-1);
		}
		// part 3 end
		*/

	}


	private static void special(Instances data, String common) {
		System.out.println("<Predictions for the Test Set Instances>");
		int count = 0;
		int index = 1;
		for (Instance insta:data) {
			System.out.println(index + ": Actual: " + insta.toString(insta.numAttributes()-1) + " Predicted: " + common);
			index++;
			if (insta.toString(insta.numAttributes()-1).equals(common)) count++;
		}
		System.out.print("Number of correctly classified: " + count 
			+ " Total number of test instances: " + data.numInstances());
	}

	private static String commonLabel(Instances data) {
		int total = data.numInstances();
		int numFirstLabel = 0;
		String firstLabel = data.attribute(data.numAttributes()-1).value(0);
		String secondLabel = data.attribute(data.numAttributes()-1).value(1);
		for (int i = 0; i < total; i++) {
			if (data.instance(i).toString(data.numAttributes()-1).equals(firstLabel)) numFirstLabel++;
		}
		int numSecondLabel = total - numFirstLabel;
		if (numFirstLabel >= numSecondLabel) return firstLabel;
		else return secondLabel;
	}

	private static void preOrderPrinting(DTNode node, int height, ArrayList<ArrayList<Integer>> count) {
		if (node.isLeaf()) {
			for (int j = 0; j < height; j++) {
					System.out.print("|\t");
			}
			if (node.attribute.type() == Attribute.NOMINAL) {
				System.out.print(node.attribute.name() + " = " + node.attrValue + " [");
				int indexOfValue = node.attribute.indexOfValue(node.attrValue);
				System.out.print(count.get(indexOfValue).get(0) + " " + count.get(indexOfValue).get(1));
				System.out.println("]: " + node.label);
				return;
			}
			if (node.attribute.type() == Attribute.NUMERIC) {
				if (node.side == 0)	{
					System.out.print(node.attribute.name() + " <= ");
					System.out.printf("%.6f", node.threshold);
					System.out.print(" [" + count.get(0).get(0) + " " + count.get(0).get(1));
					System.out.println("]: " + node.label);
				} else if (node.side == 1) {
					System.out.print(node.attribute.name() + " > ");
					System.out.printf("%.6f", node.threshold);
					System.out.print(" ["+ count.get(1).get(0) + " " + count.get(1).get(1));
					System.out.println("]: " + node.label);
				} else {
					System.err.println("wrong side number!!!!!!!!");
					System.exit(-1);
				}
				return;
			}
		} else {
			for (int j = 0; j < height; j++) {
					System.out.print("|\t");
			}
			if (node.attribute.type() == Attribute.NOMINAL) {
				System.out.print(node.attribute.name() + " = " + node.attrValue + " [");
				int indexOfValue = node.attribute.indexOfValue(node.attrValue);
				System.out.print(count.get(indexOfValue).get(0) + " " + count.get(indexOfValue).get(1));
				System.out.println("]");
				for (DTNode child:node.children) {
					preOrderPrinting(child, height+1, node.numInstances);
				}
				return;
			}
			if (node.attribute.type() == Attribute.NUMERIC) {
				if (node.side == 0)	{
					System.out.print(node.attribute.name() + " <= ");
					System.out.printf("%.6f", node.threshold);
					System.out.print(" [" + count.get(0).get(0) + " " + count.get(0).get(1));
					System.out.println("]");
					for (DTNode child:node.children) {
						preOrderPrinting(child, height+1, node.numInstances);
					}
					return;
				} else if (node.side == 1) {
					System.out.print(node.attribute.name() + " > ");
					System.out.printf("%.6f", node.threshold);
					System.out.print(" ["+ count.get(1).get(0) + " " + count.get(1).get(1));
					System.out.println("]");
					for (DTNode child:node.children) {
						preOrderPrinting(child, height+1, node.numInstances);
					}
					return;
				} else {
					System.err.println("wrong side number!!!!!!!!");
					System.exit(-1);
				}
				return;
			}
		}
	}

	private static boolean test(Instance insta, DTNode root, int index) {
		DTNode curr = root;
		System.out.print(index + ": Actual: " + insta.toString(insta.numAttributes()-1) + " Predicted: ");
		Attribute currAttr = root.children.get(0).attribute;
		while (!curr.isLeaf()) {
			if (currAttr.type() == Attribute.NOMINAL) {
				for (DTNode child:curr.children) {
					if (child.attrValue.equals(insta.toString(currAttr))) {
						curr = child;
						if (!curr.isLeaf()) currAttr = curr.children.get(0).attribute;
						break;
					}
				}
			} else if (currAttr.type() == Attribute.NUMERIC) {
				if (insta.value(currAttr) <= curr.children.get(0).threshold+10e-6) {
						curr = curr.children.get(0); 
						if (!curr.isLeaf()) currAttr = curr.children.get(0).attribute;
				} else {
					curr = curr.children.get(1);
					if (!curr.isLeaf()) currAttr = curr.children.get(0).attribute;
				}
			} else {
				System.err.println("Wrong attribute type!");
				System.exit(-1);
			}
		}
		System.out.println(curr.label);
		if (curr.label.equals(insta.toString(insta.numAttributes()-1))) return true;
		return false;
	}

	/**
	 * Part 2 and 3 helper methods: remove the comment symbol when using
	 */
	/*
	private static boolean testWithoutPrint(Instance insta, DTNode root) {
		DTNode curr = root;
		Attribute currAttr = root.children.get(0).attribute;
		while (!curr.isLeaf()) {
			if (currAttr.type() == Attribute.NOMINAL) {
				for (DTNode child:curr.children) {
					if (child.attrValue.equals(insta.toString(currAttr))) {
						curr = child;
						if (!curr.isLeaf()) currAttr = curr.children.get(0).attribute;
						break;
					}
				}
			} else if (currAttr.type() == Attribute.NUMERIC) {
				if (insta.value(currAttr) <= curr.children.get(0).threshold+10e-6) {
						curr = curr.children.get(0); 
						if (!curr.isLeaf()) currAttr = curr.children.get(0).attribute;
				} else {
					curr = curr.children.get(1);
					if (!curr.isLeaf()) currAttr = curr.children.get(0).attribute;
				}
			} else {
				System.err.println("Wrong attribute type!");
				System.exit(-1);
			}
		}
		if (curr.label.equals(insta.toString(insta.numAttributes()-1))) return true;
		return false;
	}

	private static void part2Cal(Instances trainData, Instances testData, int numPer, PrintWriter file2) {
		int totalInsta = trainData.numInstances();
		for (int i = 0; i < 10; i++) {
			// generate random index for training instance
			HashSet<Integer> index = new HashSet<Integer>();
			while (index.size() < numPer) {
				int tmp = (int)(Math.random()*totalInsta);
				if (tmp == totalInsta) tmp--;
				if (!index.contains(tmp)) index.add(tmp);
			}
			Instances subset = new Instances(trainData);
			subset.delete();
			for (Integer in:index) {
				subset.add(trainData.instance(in));
			}
			DTTrain subTraining = new DTTrain(subset, 4); // m = 4 for part 2
			DTNode subRoot = subTraining.treeRoot;
			// do the test
			int correct = 0;
			for (Instance insta:testData) {
				if (testWithoutPrint(insta, subRoot)) correct++;
			}
			file2.println((double)correct/(double)testData.numInstances());
		}
	}
	*/
	// Part 2 and 3 helper end
}