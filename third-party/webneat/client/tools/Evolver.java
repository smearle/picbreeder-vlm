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

import client.ImageDatabaseInstance;
import client.MainComponentInstance;
import client.evolution.GeneticFactoryInstance;
import client.evolution.Genome;
import client.evolution.Individual;
import client.evolution.Series;
import client.gui.SeriesInstance;
import client.gui.SessionHandler;
import client.server.ServerConnection;
import client.utilities.ArgumentParser;

public class Evolver {
	public static void main(String []args) throws Exception {
		client.utilities.Random.instance().setSeed(System.currentTimeMillis());
		
		ArgumentParser parser = new ArgumentParser(args);
		
		final int min = Integer.parseInt(parser.findArgument("--min"));
		final int max = Integer.parseInt(parser.findArgument("--max"));
		final int genomeId = Integer.parseInt(parser.findArgument("-g"));
		final int generations = Integer.parseInt(parser.findArgument("-n"));
		final String output = parser.findArgument("-o");
		final int threads = parser.hasOption("--threads") ? Integer.parseInt(parser.findArgument("--threads")) : 4;
		
		// copy+paste from client.gui.Applet.start()
		ServerConnection server = new client.server.SOAPServer();
		server.initialize(new RendererInitialization(genomeId));
		SessionHandler.beginSession(server);

		GeneticFactoryInstance.get().setScheme("both");
		Series series = null;
		
		if(client.server.ServerConnectionInstance.get().hasSeries()) {
			series = GeneticFactoryInstance.get().createInvalidSeries();
			
			java.io.InputStream xml = client.server.ServerConnectionInstance.get().getSeries();
			client.utilities.XML.load(series, xml);
		}
		else if(client.server.ServerConnectionInstance.get().hasGenome()) {
			Genome genome = GeneticFactoryInstance.get().createInvalidGenome();

			java.io.InputStream xml = client.server.ServerConnectionInstance.get().getGenome();
			
			client.utilities.XML.load(genome, xml);
			series = GeneticFactoryInstance.get().createBranchSeries(genome);
		}
		else
			series = GeneticFactoryInstance.get().createRootSeries();
	
		client.server.ServerConnectionInstance.get().clientStarted();
		SeriesInstance.set(series);
		
		// new code to evolve
		
		// use color! remove these 2 lines if you want greyscale
		for(Individual indiv : series.getCurrentGeneration())
			indiv.getGenome().setDominantPhenotype(1);
		
		for(int i = 0; i < generations; i++) {
			int targetCount = client.utilities.Random.instance().nextInt(max - min + 1) + min;
			
			// generate a permutation of 0 through size-1 to random pick the parents
			int []indices = new int[series.getCurrentGeneration().getSize()];
			for(int j = 0; j < indices.length; j++)
				indices[j] = j;
			
			for(int j = 0; j < indices.length; j++) {
				int ind = client.utilities.Random.instance().nextInt(indices.length);
				int temp = indices[j];
				indices[j] = indices[ind];
				indices[ind] = temp;
			}
			
			// mark first targetCount genomes
			for(int j = 0; j < targetCount; j++)
				series.getIndividualFromCurrentGeneration(indices[j]).select();
			
			// spawn generation
			series.spawn();
		}
		
		// pick a rep
		int rep = client.utilities.Random.instance().nextInt(series.getCurrentGeneration().getSize());
		Individual representative = series.getIndividualFromCurrentGeneration(rep);
		representative.select();
		client.renderers.RenderingAlgorithm alg = new client.renderers.algorithms.Parallel(threads);
		alg.render(representative);
		
		// save result
		((test.ImagePhenotype)representative.getDominantPhenotype()).save(output);
	}
}
