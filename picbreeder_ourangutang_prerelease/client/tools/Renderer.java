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

import client.cppn.*;
import client.evolution.*;
import client.evolution.impl.DefaultGeneticFactory;
import client.utilities.ArgumentParser;
import client.utilities.XML;
import java.util.zip.*;
import java.io.BufferedInputStream;

/**
 * 
 * 
 */
public class Renderer {
	public static void main(String []args) {
		if(args.length == 0) {
			System.out.println("Usage: java client.tools.Renderer [options]");
			System.out.println("Options:");
			System.out.println("    -o outputFile");
			System.out.println("    -i genomeFile");
			System.out.println("    -w width");
			System.out.println("    -h height");
			System.out.println("    -v verbose");
			//System.out.println("    -c input from stdin");
			return;
		}
		
		ArgumentParser parser = new ArgumentParser(args);
		client.ParameterTableInstance.set(new test.TestParameters());
		
		String genomeFile = parser.findArgument("-i");
		int width = Integer.parseInt(parser.findArgument("-w"));
		int height = Integer.parseInt(parser.findArgument("-h"));
		String output = parser.findArgument("-o");
		boolean verbose = parser.hasOption("-v");
		
		try {
			Genome g = GeneticFactoryInstance.get().createInvalidGenome();
			XML.loadFromFile(g, genomeFile);
			
			if(g == null) {
				System.out.println("Invalid genome id number.");
				System.exit(-1);
			}
			
			Network network = CPPNFactoryInstance.get().createNetwork(g);
			test.ImagePhenotype image = new test.ImagePhenotype(width, height);

			long start = System.currentTimeMillis();
			
			client.renderers.RenderingAlgorithm alg = new client.renderers.algorithms.Background();
			alg.render(network, image);
			
			long end = System.currentTimeMillis();
			
			if(verbose)
				System.out.println("Rendering took: " + ((end-start)/1000.0) + " seconds.");
			
			image.save(output);
			
			if(verbose)
				System.out.println("Wrote to: " + output);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
