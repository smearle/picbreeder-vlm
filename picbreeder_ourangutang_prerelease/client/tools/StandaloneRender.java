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

import client.evolution.GeneticFactoryInstance;
import client.evolution.Genome;
import client.utilities.ArgumentParser;

public class StandaloneRender {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		if(args.length==0)
		{
			System.out.println("Usage: java client.tools.Renderer [options]");
			System.out.println("Options:");
			System.out.println("    -o outputFile");
			System.out.println("    -g genomeId");
			System.out.println("    -w width");
			System.out.println("    -h height");
			return;			
		}
		
		ArgumentParser parser = new ArgumentParser(args);
		try{
		String width = parser.findArgument("-w");
		String height = parser.findArgument("-h");
		String output = parser.findArgument("-o");
		int genome = Integer.parseInt(parser.findArgument("-g"));
		
		client.server.ServerConnectionInstance.set(new client.server.SOAPServer());
		
		client.server.ServerConnectionInstance.get().initialize(new client.tools.RendererInitialization(genome));
		
		java.io.InputStream streamIn = client.server.ServerConnectionInstance.get().getGenome();
		java.io.FileOutputStream streamOut=new java.io.FileOutputStream("./tempGenome.xml");
		int c;
        while ((c = streamIn.read()) != -1) 
        {
           streamOut.write(c);
        }
		
		String[] parameters={"-i","./tempGenome.xml","-o",output,"-w",width,"-h",height};
		client.tools.Renderer.main(parameters);
		new java.io.File("./tempGenome.xml").delete();
		}catch(Exception e){e.printStackTrace();};
	}
	public void render(String width, String height, int genome, String output)
	{
		try{
		client.server.ServerConnectionInstance.set(new client.server.SOAPServer());
		
		client.server.ServerConnectionInstance.get().initialize(new client.tools.RendererInitialization(genome));
		
		java.io.InputStream streamIn = client.server.ServerConnectionInstance.get().getGenome();
		java.io.FileOutputStream streamOut=new java.io.FileOutputStream("./tempGenome.xml");
		int c;
        while ((c = streamIn.read()) != -1) 
        {
           streamOut.write(c);
        }
		
		String[] parameters={"-i","./tempGenome.xml","-o",output,"-w",width,"-h",height};
		client.tools.Renderer.main(parameters);
		new java.io.File("./tempGenome.xml").delete();
		}catch(Exception e){e.printStackTrace();};
	}

}
