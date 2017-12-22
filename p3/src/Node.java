import java.util.*;

public class Node {
	private int type = -1; // 0 = input, 1 = biasToHidden, 2 = hidden, 3 = biasToOutput, 4 = output
	public ArrayList<NodeWeightPair> parents = null; // ArrayList that contans the parents (node+weight) info if applicable

	private Double inputValue = 0.0;
	private Double outputValue = 0.0; // Output value of a node: same as input value for an input node, 1.0 for bias nodes, and calculated based on sigmoid function for hidden & output nodes
	private Double sum = 0.0; // sum if wi*xi

	public Node(int type) {
		// check if valid argument
		if (type > 4 || type < 0) {
			System.err.println("Incorrect Node type parameter");
			System.exit(-1);
		}
		this.type = type;
		if (type == 2 || type == 4) {
			parents = new ArrayList<NodeWeightPair>();
		}
		if (type == 1 || type == 3) {
			inputValue = 1.0;
		}

	}

	public void setInput(Double inputValue) {
		// for input layer
		if (type == 0) {
			this.inputValue = inputValue;
		} else if (type == 2 || type == 4) {
			this.inputValue = 0.0;
			for(int i = 0; i < parents.size(); i++) {
				this.inputValue += parents.get(i).weight * parents.get(i).node.getOutput();
			}
		} else {
			System.out.println("setInput called by incorrect node");
			System.exit(-1);
		}

	}

	public void calculateOutput() {
		if (type == 2 || type == 4) {
			// sigmoid activation function
			// System.out.println(this.inputValue);
			this.outputValue = sigmoid(this.inputValue);
		} else {
			System.out.println("calculateOutput called by incorrect node");
			System.exit(-1);
		}
	}

	private Double sigmoid(Double input) {
		return (1.0/(1.0+Math.exp(-input)));
	}

	public double calSum() {
		return sum;
	}

	public double getOutput() {
		if (type == 0) { // inputNode
			return inputValue;
		} else if (type == 1 || type == 3) { // bias node
			return 1.00;
		} else {
			return outputValue;
		}
	}
}