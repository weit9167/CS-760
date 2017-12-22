import java.util.*;
import weka.core.*;
import java.io.*;

public class AttrValuesPair{
	public Double[][] label1Table = null;
	public Double[][] label2Table = null;
	private int numValueXi = 0;
	private int numValueXj = 0;
	private int totalInsta = 0;
	private int numValueY = 0;

	public AttrValuesPair(int numValueXi, int numValueXj, int numValueY, int totalInsta) {
		this.numValueXi = numValueXi;
		this.numValueXj = numValueXj;
		this.label1Table = new Double[numValueXi][numValueXj];
		this.label2Table = new Double[numValueXi][numValueXj];
		this.numValueY = numValueY;
		this.totalInsta = totalInsta;
		initializeTable(label1Table, label2Table);
	}

	private void initializeTable(Double[][] table1, Double[][] table2) {
		for (int i = 0; i < table1.length; i++) {
			for (int j = 0; j < table1[0].length; j++) {
				table1[i][j] = 0.0;
				table2[i][j] = 0.0;
			}
		}
	}

	// after fill in the count, convert the values in the table into probablity
	public void calcProb() {
		int totalWithCount = this.totalInsta + this.numValueY*this.numValueXi*this.numValueXj;
		for (int i = 0; i < this.numValueXi; i++) {
			for (int j = 0; j < this.numValueXj; j++) {
				this.label1Table[i][j] = ((double)(label1Table[i][j]+1))/((double)totalWithCount);
				this.label2Table[i][j] = ((double)(label2Table[i][j]+1))/((double)totalWithCount);
			}
		}
	}

	public void calcCondProb(int numLabel1) {
		int totalLabel1WithCount = numLabel1 + this.numValueXi*this.numValueXj;
		int totalLabel2WithCount = this.totalInsta - numLabel1 + this.numValueXi*this.numValueXj;
		for (int i = 0; i < this.numValueXi; i++) {
			for (int j = 0; j < this.numValueXj; j++) {
				this.label1Table[i][j] = ((double)(label1Table[i][j]+1))/((double)totalLabel1WithCount);
				this.label2Table[i][j] = ((double)(label2Table[i][j]+1))/((double)totalLabel2WithCount);
			}
		}
	}
}