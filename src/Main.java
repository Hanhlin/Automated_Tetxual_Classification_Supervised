package nini.nlp;

import java.util.*;

public class Main {

	public static void main(String[] args) throws Exception {
		
		// ====== pre-process documents, generate term-freq matrix, tf-idf matrix ======
		
        List<String> documentStrings = preProcess.readData("train");
        List<List<String>> termsList = preProcess.vectorize(documentStrings, "stopwords.txt");

        Map<String, Integer> termIndex = preProcess.getTermsIndex(termsList);
        double[][] tf = preProcess.getTermFreqMatrix(termsList, termIndex);
//        printInfo(tf, "Term-frequency Matrix:");
        
        double[][] tf_idf = preProcess.getTfIdf(tf);
//        printInfo(tf_idf, "TF-IDF Matrix:");
 
        
        // ========== generate keywords per folder ==========
        
        preProcess.getTopTerms(tf_idf, termIndex, "topics.txt");
        Kmeans model = new Kmeans(tf_idf, 3);
        model.cluster();
        
        String[] cluster = new String[3];
        cluster[model.clusterAssignments[0]] = "C1";
        cluster[model.clusterAssignments[8]] = "C4";
        cluster[model.clusterAssignments[16]] = "C7";
        System.out.println("C1 is assigned to label " + model.clusterAssignments[0]);
        System.out.println("C4 is assigned to label " + model.clusterAssignments[8]);
        System.out.println("C7 is assigned to label " + model.clusterAssignments[16]);

        // ========== read test data ========== 
        
        List<String> tests =  preProcess.readData("test");
        double[][] test_tf_idf = preProcess.getTest_TfIdf(tf, termIndex, tests, "stopwords.txt");
        preProcess.knn(3, tf_idf, test_tf_idf, model.clusterAssignments, cluster);
        preProcess.fKnn(3, 2, tf_idf, test_tf_idf, model.clusterAssignments, cluster);
		
	}
	
	public static void printInfo(double[][] matrix, String title) {
		
		System.out.println(title);
        for (double[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println("\n");
	}

}
