import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.HashMap;
import java.text.DecimalFormat;

public class analysis {
	public static void main(String[] args) throws FileNotFoundException {
		HashMap<String, Integer> truths = new HashMap<String, Integer>();
		truths.put("true negative", 0);
		truths.put("true positive", 0);
		truths.put("false negative", 0);
		truths.put("false positive", 0);
		/* Uncomment for each data set */
		 Scanner inputFile = new Scanner(new File("DatasetA.txt"));
		// Scanner inputFile = new Scanner(new File("DatasetB.txt"));
		// Scanner inputFile = new Scanner(new File("DatasetC.txt"));

		String[] inputData = new String[12000];
		while(inputFile.hasNextLine()) {
			inputData = inputFile.nextLine().split(",");
			
			/* Uncomment for each data set */
			 Double value = ((20 * Double.parseDouble(inputData[0])) + (7 * Double.parseDouble(inputData[1])) - 832500);
			// Double value = ((8 * Double.parseDouble(inputData[0])) + (3 * Double.parseDouble(inputData[1])) - 360000);
			// Double value = ((10 * Double.parseDouble(inputData[0])) + (7 * Double.parseDouble(inputData[1])) - 655000);
	
			if ((value > 0) && (Integer.parseInt(inputData[2]) == 0)) {
				truths.put("true positive", truths.get("true positive") + 1); 
			}
			else if ((value > 0) && (Integer.parseInt(inputData[2]) == 1
					)) {
				truths.put("false positive", truths.get("false positive") + 1); 
			}
			else if ((value < 0) && (Integer.parseInt(inputData[2]) == 1)) {
				truths.put("true negative", truths.get("true negative") + 1); 
			}
			else {
				truths.put("false negative", truths.get("false negative") + 1); 
			}
		}
		DecimalFormat df2 = new DecimalFormat("#.####");
		System.out.println("False positives: " + truths.get("false positive") + "   False negatives: " + truths.get("false negative"));
		double accuracy = (((double)truths.get("true positive") + truths.get("true negative")) / 4000);
		double error = 1- accuracy;
		double tp = (double)truths.get("true positive")/ (truths.get("true positive") + truths.get("false negative"));
		double tn = (double)truths.get("true negative")/ (truths.get("true negative") + truths.get("false positive"));
		double fp = (double)truths.get("false positive")/ (truths.get("true negative") + truths.get("false positive"));
		double fn = (double)truths.get("false negative")/ (truths.get("true positive") + truths.get("false negative"));
		System.out.println("Accuracy = " + df2.format(accuracy));
		System.out.println("Error = " + df2.format(error));
		System.out.println("True Positive Rate = " + df2.format(tp));
		System.out.println("False Positive Rate = " + df2.format(fp));
		System.out.println("True Negative Rate = " + df2.format(tn));
		System.out.println("False Negative Rate = " + df2.format(fn));
		inputFile.close();
	}
}
