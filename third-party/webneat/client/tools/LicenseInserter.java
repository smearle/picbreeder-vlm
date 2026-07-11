/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.tools;

public class LicenseInserter {
	private static final String [] license = {
		"/*",
		" * Unlicensed intellectual property of the University of Central Florida for",
		" * internal usage only. You may not distribute this code to anyone. You may",
		" * not use this code (as source or compiled) or information obtained from",
		" * this code without permission.",
		" *",
		" * Picbreeder Project",
		" * Evolutionary Complexity Research Group",
		" * School of Electrical Engineering and Computer Science",
		" * 2006-2007",
		" */",
		""
	};

	public static void main(String []args) throws Exception {
		goDir(new java.io.File("client"));
		goDir(new java.io.File("test"));
		//goFile(new java.io.File("client/tools/LicenseInserter.java"));
	}
	
	private static void goDir(java.io.File directory) throws Exception {
		for(java.io.File f : directory.listFiles())
			if(f.isDirectory())
				goDir(f);
			else if(f.getName().endsWith(".java"))
				goFile(f);
	}
	
	private static void goFile(java.io.File file) throws Exception {
		System.out.println("Processing: " + file.getName());
		
		java.util.Scanner in = new java.util.Scanner(file);
		boolean ok = true;
		
		for(String s : license)
			if(!in.hasNextLine() || !in.nextLine().equals(s))
				ok = false;
		
		if(ok)
			return;

		in.close();
		
		java.io.File temp = java.io.File.createTempFile("code", ".java");
		java.io.PrintWriter out = new java.io.PrintWriter(new java.io.BufferedOutputStream(new java.io.FileOutputStream(temp)));
		
		for(String s : license)
			out.println(s);
		
		in = new java.util.Scanner(file);
		while(in.hasNextLine())
			out.println(in.nextLine());
		
		in.close();
		out.close();
		
		in = new java.util.Scanner(temp);
		out = new java.io.PrintWriter(new java.io.BufferedOutputStream(new java.io.FileOutputStream(file)));
		
		while(in.hasNextLine()) {
			out.println(in.nextLine());
		}

		in.close();
		out.close();
	}
}
