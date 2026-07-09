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

import java.io.*;
import java.util.*;
import client.evolution.*;
import client.server.*;
import client.utilities.*;

/**
 * This will verify a series evolved properly.
 * <p>
 * Invoked with 1 argument, this will check the given directory
 * assuming that it is a root branch. Invoked with 2 arguements, this
 * will check the first argument's directory using the second argument's
 * directory to look for the "branch from" genome.
 * <p>
 * Currently, I just make sure all markings existed in a previous
 * generation and that the parent id is in the previous generation.
 * 
 * @author Nick
 */

public class Verifier {
	static final String GENOME_FILE = "rep";
	static final String SERIES_FILE = "main";
	static final String FILE_EXTENSION = "zip";
	
	public static void main(String []args) throws Exception {
		if(args.length < 1 || args.length > 2) {
			System.out.println("invalid arguments");
			System.exit(-1);
		}
		
		for(int i = 0; i < args.length; i++)
			if(!args[i].endsWith(File.separator))
				args[i] += File.separator;
		
		ServerConnection server = new LocalServer();
		server.initialize(new client.server.FileInitialization(args[0], "." + FILE_EXTENSION));
		ServerConnectionInstance.set(server);
		client.ParameterTableInstance.set(server.getParameters());

		Series series = GeneticFactoryInstance.get().createInvalidSeries();
		XML.loadFromFile(series, args[0] + SERIES_FILE + "." + FILE_EXTENSION);
		
		client.gui.SeriesInstance.set(series);
		
		Set <Identifier> parentIdentifiers = new TreeSet <Identifier> ();
		Set <Marking> allMarkings = new TreeSet <Marking> ();
		long highestMarking = -1;
		
		
		// initialize the branch from genes and make sure the identifier matches
		if(args.length == 2) {
			Genome parent = GeneticFactoryInstance.get().createInvalidGenome();
			XML.loadFromFile(parent, args[1] + GENOME_FILE + "." + FILE_EXTENSION);
			
			for(Gene g : parent.getGenes())
				allMarkings.add(g.getMarking());
			
			String parentBranch = parent.getIdentifier().getBranch();
			parentIdentifiers.add(parent.getIdentifier());
			
			if(!parentBranch.equals(series.getPreviousBranch())) {
				System.out.println("Parent branch does not match!");
				System.exit(-1);
			}
		}
		else {
			if(series.getPreviousBranch() != null) {
				System.out.println("Parent branch does not match!");
				System.exit(-1);
			}
		}
		
		// loop through the generation, making sure genes were added in order
		// or existed in some ancestor
		for(int i = 0; i < series.getLength(); i++) {
			Generation gen = series.getGeneration(i);
			
			// make sure the parents exist in the previous generation
			for(Individual ind : gen)
				if(ind.getGenome() != null) {
					if(ind.getGenome().getParentIdentifiers().size() == 0)
						if(i != 0 || args.length == 2) {
							System.out.println(ind.getGenome().getIdentifier() + " parents are missing... not a root generation!");
							System.exit(-1);
						}

					for(Identifier pi : ind.getGenome().getParentIdentifiers())
						if(!parentIdentifiers.contains(pi)) {
							System.out.println(ind.getGenome().getIdentifier() + " parent " + pi + " was not found!");
							System.exit(-1);
						}
				}
			
			// add this generation as parents for the next
			parentIdentifiers.clear();
			for(Individual ind : gen)
				if(ind.getGenome() != null)
					parentIdentifiers.add(ind.getGenome().getIdentifier());
			
			// make sure any "old" looking markings exist in the parent markings
			for(Individual ind : gen) {
				if(ind.getGenome() != null)
					for(Gene g : ind.getGenome().getGenes())
						if(!g.getMarking().usesCurrentBranch() || g.getMarking().getId() <= highestMarking)
							if(!allMarkings.contains(g.getMarking())) {
								System.out.println(ind.getGenome().getIdentifier() + " has a bad gene... " + g.getMarking());
								System.exit(-1);
							}
			}
			
			// add the current generation markings to the set of known markings
			for(Individual ind : gen)
				if(ind.getGenome() != null)
					for(Gene g : ind.getGenome().getGenes())
						if(g.getMarking().usesCurrentBranch()) {
							allMarkings.add(g.getMarking());
							highestMarking = Math.max(highestMarking, g.getMarking().getId());
						}
		}
		
	}
}
