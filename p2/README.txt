Author: Wei Tang
Student ID: 907 176 1275

List of Contents:
Folders:
1. source_code: contains all the *.java files and *.class files, and a Manifest.txt and weka.jar;
   When compile, cd into this folder, use command:
	javac -cp weka.jar *.java
   If want to generate the dt-learn.jar file, under the same path, use command:
	jar cfm dt-learn.jar Manifest.txt *.class
   To add dt-learn as a command, use:
	chmod +x dt-learn
2. part_2_and_3_raw_data: contains the raw *.txt data files for part2 and part2, two datasets
3. train_data: contains the *.arff files for training
4. test_data: contains the *.arff files for testing
5. test_sol: contains the *.txt files of testing solution

Files:
1. README.txt: this file
2. dt-learn.jar: the jar file of decision tree learner
3. weka.jar: the jar file of weka package
4. hw1.pdf: the file for part2 & 3
5. dt-learn: the executable. Before running the program, make sure that:
	dt-learn.jar
	weka.jar
	<train-set-file>
	<test-set-file>
   the above four files are all in the same path. To run the program, use the command:
	./dt-learn <train-set-file> <test-set-file> m
