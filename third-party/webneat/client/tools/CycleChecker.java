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

import client.cppn.CPPNFactoryInstance;
import client.cppn.Network;
import client.evolution.GeneticFactoryInstance;
import client.evolution.Genome;
import client.evolution.Link;
import client.evolution.Node;
import client.utilities.ArgumentParser;
import client.utilities.XML;

public class CycleChecker {
	
	public static void main(String []args) throws Exception {
		if(args.length == 0) {
			System.out.println("Usage: java client.tools.CycleChecker [options]");
			System.out.println("Options:");
			System.out.println("    -g genomeId");
			return;
		}
		
		ArgumentParser parser = new ArgumentParser(args);
		int genomeId = Integer.parseInt(parser.findArgument("-g"));
		
		client.ParameterTableInstance.set(new test.TestParameters());
		//client.server.ServerConnectionInstance.set(new client.server.SOAPServer());
		//client.server.ServerConnectionInstance.get().initialize(new client.tools.RendererInitialization(genomeId));
		
		//java.io.InputStream streamIn = client.server.ServerConnectionInstance.get().getGenome();
		
		Genome g = GeneticFactoryInstance.get().createInvalidGenome();
		//client.utilities.XML.load(g, streamIn);
		
		client.utilities.XML.loadFromFile(g, "tempGenome.xml");
		
		int n = g.countNodes();
		
		boolean [][]adj = new boolean[n][n];
		
		java.util.Map <Node, Integer> vs = new java.util.TreeMap <Node, Integer> ();
		
		System.out.println("Checking " + n + " nodes!");
		
		// build the graph adjacency matrix
		int i = 0, j = 0;
		for(Node u : g.getNodes())
			vs.put(u, i++);
		
		for(Link link : g.getLinks())
			adj[vs.get(g.getNode(link.getSourceMarking()))][vs.get(g.getNode(link.getDestinationMarking()))] = true;
		
		// run floyds as a transitive closure
		for(int a = 0; a < n; a++)
			for(int b = 0; b < n; b++)
				for(int c = 0; c < n; c++)
					if(adj[b][a] && adj[a][c])
						adj[b][c] = true;
		
		// verify no cycle exists by looking for u->v and v->u
		for(i = 0; i < n; i++)
			for(j = 0; j < n; j++)
				if(i != j && adj[i][j] && adj[j][i])
					System.out.println("Found Cycle!");
		
		System.out.println("DONE");
	}
}
