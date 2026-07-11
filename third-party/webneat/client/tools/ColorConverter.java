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
import client.evolution.Series;
import client.utilities.ArgumentParser;
import client.utilities.XML;

public class ColorConverter {
	public static void main(String []args) throws Exception {
		if(args.length == 0) {
			System.out.println("Usage: java client.tools.Converter [options]");
			System.out.println("Options:");
			System.out.println("    -o outputFile");
			System.out.println("    -g genome id");
			return;
		}
		
		ArgumentParser parser = new ArgumentParser(args);
		String in = parser.findArgument("-g");
		String out = parser.findArgument("-o");

		client.server.ServerConnectionInstance.set(new client.server.SOAPServer());
		client.server.ServerConnectionInstance.get().initialize(new client.tools.RendererInitialization(Integer.parseInt(in)));
		
		client.ParameterTableInstance.set(new test.TestParameters());
		
		Series s = GeneticFactoryInstance.get().createInvalidSeries();
		s.setCurrentBranch(in);
		
		Genome g = GeneticFactoryInstance.get().createInvalidGenome();

		java.io.InputStream streamIn = client.server.ServerConnectionInstance.get().getGenome();
		XML.load(g, streamIn);
		streamIn.close();
		
		
		java.util.Collection <Genome> p = new java.util.LinkedList<Genome> ();
		p.add(g);
		
		client.evolution.Generator gen = new client.evolution.generators.InkToHSB();
		//client.evolution.Generator gen = new client.evolution.generators.InkToRGB();
		g = gen.generate(p);
		
		XML.store(g, new java.io.File(out));
		
	}
}
