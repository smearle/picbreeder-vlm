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

package client.server;

public class FileInitialization implements Initialization {
	private String genome = null, series = null, path = null, ext = ".xml";
	
	public FileInitialization() {
		Object [] values = {"Series", "Genome", "Null"};
		
		Object selected = javax.swing.JOptionPane.showInputDialog(client.MainComponentInstance.get(), "How do you want to start?", "Initialization", javax.swing.JOptionPane.QUESTION_MESSAGE, null, values, values[0]);
		if(selected == values[0])
			series = promptFile();
		else if(selected == values[1])
			genome = promptFile();
	}
	
	public FileInitialization(String path, String extension) {
		this(path, extension, false);
	}
	
	public FileInitialization(String path, String extension, boolean isGenome) {
		this.path = path.endsWith(java.io.File.separator) ? path : path + java.io.File.separator;
		this.ext = extension.startsWith(".") ? extension : "." + extension;
		
		if(isGenome)
			genome = "rep" + ext;
		else
			series = "main" + ext;
	}
	
	public String getParameter(String param) {
		if(param.equals("Genome")) return genome;
		if(param.equals("Series")) return series;
		if(param.equals("Path")) return path;
		if(param.equals("Extension")) return ext;
		if(param.equals("Parameters")) return "config" + java.io.File.separator + "default.xml";
		return null;
	}

	private String promptFile() {
		javax.swing.JFileChooser fc = new javax.swing.JFileChooser();
		if(fc.showOpenDialog(client.MainComponentInstance.get()) == javax.swing.JFileChooser.APPROVE_OPTION) {
			try {
				java.io.File file = fc.getSelectedFile();
				ext = file.getName().substring(file.getName().lastIndexOf('.'));
				path = file.getParent() + java.io.File.separator;
				
				return file.getName();
			}
			catch(Exception e) {
				e.printStackTrace();
				return null;
			}
		}
		return null;
	}
}
