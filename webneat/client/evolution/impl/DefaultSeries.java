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

package client.evolution.impl;

import java.util.*;
import org.w3c.dom.*;
import client.evolution.*;
import client.server.ServerException;

class DefaultSeries implements Series {
	private static final int INITIAL_PREFETCH = 10; // number of generations to retrieve when starting
	private static final int LOOK_AHEAD = 5; // number of generations to look back when prefetching
	
	private Genome branchFrom;
	private Identifier branchFromIdentifier;
	private List <GenerationImposter> generations;
	private int current;
	private int prefetchIndex;
	private int earliestSpawn;
	
	public DefaultSeries(boolean spawnPopulation) {		
		generations = new LinkedList <GenerationImposter> ();
		branchFrom = null;
		branchFromIdentifier = null;
		
		if(spawnPopulation) {
			initializeFirstGeneration();
		}
		else {
			current = -1;
			prefetchIndex = 0;
			earliestSpawn = Integer.MAX_VALUE;
			//generations.add(new GenerationImposter(GeneticFactoryInstance.get().createInvalidGeneration()));
		}
	}
	
	public DefaultSeries(Genome branchFrom) {
		generations = new LinkedList <GenerationImposter> ();
		this.branchFrom = branchFrom;
		branchFromIdentifier = branchFrom.getIdentifier();
		
		initializeFirstGeneration();
	}
	
	public void initializeFirstGeneration() {
		// TODO does not correctly reset database
			
		current = 0;
		prefetchIndex = 0;
		earliestSpawn = 0;
		
		// root generation redo
		if(branchFrom == null) {
			Generation g = GeneticFactoryInstance.get().createRootGeneration();
			generations.clear();
			generations.add(new GenerationImposter(g));
		}
		// branch first generation redo
		else {
			LinkedList <Genome> parent = new LinkedList <Genome> ();
			parent.add(branchFrom);
			
			Generation g = GeneticFactoryInstance.get().createGeneration(parent, 0);
			generations.clear();
			generations.add(new GenerationImposter(g));
		}
	}
	
	public String getPreviousBranch() {
		if(branchFromIdentifier == null)
			return null;
		else
			return branchFromIdentifier.getBranch();
	}
	
	private void prefetch(int generation) {
		generation = Math.max(generation, 0);
		
		if(generation < prefetchIndex) {
			prefetchIndex = generation;
			
			GenerationImposter g = generations.get(generation);
			
			if(!g.isLoaded())
				g.prefetch();
		}
	}
	
	public int getLength() {
		return generations.size();
	}
	
	public int getPosition() {
		return current;
	}
	
	public Genome findGenome(long genomeId) {
		try {
			for(GenerationImposter g : generations)
				if(g.isLoaded())
					for(Individual i : g.getGeneration())
						if(i.getGenome().getIdentifier().getId() == genomeId && i.getGenome().getIdentifier().usesCurrentBranch())
							return i.getGenome();
		}
		catch(Exception e) {
		}
		
		return null;
	}
	
	public Generation getGeneration(int gen) throws ServerException {
		return generations.get(gen).getGeneration();
	}
	
	public Individual getIndividualFromGeneration(int gen, int ind) throws ServerException {
		return getGeneration(gen).getIndividual(ind);
	}
	
	public Generation getCurrentGeneration() throws ServerException {
		restoreCurrentGeneration();
		return getGeneration(current);
	}
	
	public Individual getIndividualFromCurrentGeneration(int ind) throws ServerException {
		return getCurrentGeneration().getIndividual(ind);
	}
	
	public void setCurrentBranch(String name) {
		BranchIdentifier.getCurrentBranch().setName(name);
	}
	
	public String getCurrentBranch() {
		return BranchIdentifier.getCurrentBranch().getName();
	}
	
	public void spawn()
			throws EvolutionException {
		ArrayList <Genome> parents = new ArrayList <Genome> ();
		
		try {
			for(Individual ind : getCurrentGeneration())
				if(ind.isSelected())
					parents.add(ind.getGenome());
		}
		catch(client.server.ServerException e) {
			e.printStackTrace();
			// should NEVER happen
		}
		
		if(parents.size() == 0)
			throw new EvolutionException("No parents selected!");

		earliestSpawn = Math.min(earliestSpawn, current);
		current++;
		
		// included because it makes the garbage collector work!
		for(int i = current; i < generations.size(); i++)
			generations.set(i, null);
		
		generations = generations.subList(0, current);
		Generation g = GeneticFactoryInstance.get().createGeneration(parents, current);
		generations.add(new GenerationImposter(g));
	}
	
	public void goForward() {
		if(canGoForward()) {
			current++;
			// here for now to make easier
			try {
				DatabaseInstance.get().addGeneration(getCurrentGeneration());
			}
			catch(client.server.ServerException e) {
				e.printStackTrace();
				// should NEVER happen
			}
		}
	}
	
	public void goBack() {
		if(canGoBack()) {
			// here for now to make easier
			try {
				DatabaseInstance.get().removeGeneration(getCurrentGeneration());
			}
			catch(client.server.ServerException e) {
				e.printStackTrace();
				// should NEVER happen
			}
			current--;
		}
		
		prefetch(current - LOOK_AHEAD);
	}
	
	public boolean canGoForward() {
		return current < getLength() - 1;
	}
	
	public boolean canGoBack() {
		return current > 0;
	}
	
	public void prune() {
		try {
			earliestSpawn = Math.min(earliestSpawn, current);
			
			client.evolution.Database db = client.evolution.DatabaseInstance.get();
			
			// changes the selected status to reflect whether
			// or not a genome is an ancestor of the genomes
			// being saved
			Set <Genome> valid = new TreeSet <Genome> ();
			Set <Genome> parents = new TreeSet <Genome> ();
			
			// commented out if statament
			// if the bug when saving multiple times occurs
			for(Individual ind : getCurrentGeneration())
				if(ind.isSelected())
					valid.add(ind.getGenome());
			
			// always remove the current generation and add it so the color status
			// is saved for the current generation
			db.removeGeneration(getCurrentGeneration());
			db.addGeneration(getCurrentGeneration());
			
			// go through the generations in reverse order, maintaining parents
			// by selecting them (and consequently deselecting non-parents)
			for(int generation = current; generation >= earliestSpawn; generation--) {
				parents.clear();
				
				
				// go through the generation
				// if the genome is in the valid set, add its
				// parents to the parents set.
				for(Individual ind : getGeneration(generation)) {
					// do old storage files need to be updated because pruning?
					boolean inSet = valid.contains(ind.getGenome());
					
					if(inSet != ind.isSelected()) {
						db.removeGeneration(getGeneration(generation));
						db.addGeneration(getGeneration(generation));
					}
						
					if(inSet) {
						ind.select();
						parents.addAll(ind.getGenome().getParents());
					}
					else
						ind.deselect();
				}
				
				// swap parents for the next iteration
				Set <Genome> temp = valid;
				valid = parents;
				parents = temp;
			}
		}
		catch(client.server.ServerException e) {
			e.printStackTrace();
			// should NEVER happen
		}
	}
	
	// called to undo the prune algorithm
	private void restoreCurrentGeneration() throws client.server.ServerException {
		Generation currentGeneration = getGeneration(current);
		
		boolean restoreNescessary = false;
		for(Individual ind : currentGeneration)
			if(!ind.hasGenome()) {
				restoreNescessary = true;
				break;
			}
		
		if(!restoreNescessary)
			return;
		
		if(current == 0) {
			// has no branch, must be a root generation
			if(branchFromIdentifier == null)
				currentGeneration.restore(null);
			
			// must have been branched, get that genome and mutate it
			else {
				if(branchFrom == null)
					branchFrom = loadRepresentative();
					
				LinkedList <Genome> parents = new LinkedList <Genome> ();
				parents.add(branchFrom);
				currentGeneration.restore(parents);
			}
		}
		else {
			List <Genome> parents = new ArrayList <Genome> ();
			Generation previousGeneration = getGeneration(current-1);
			
			for(Individual ind : previousGeneration)
				if(ind.hasGenome())
					parents.add(ind.getGenome());
			
			currentGeneration.restore(parents);
		}
	}
	
	private Genome loadRepresentative() throws client.server.ServerException {
		java.io.InputStream xml = client.server.ServerConnectionInstance.get().getRepresentativeGenome(branchFromIdentifier.getBranch());
		Genome g = GeneticFactoryInstance.get().createInvalidGenome();
		client.utilities.XML.load(g, xml);
		return g;
	}

	public String getElementName() {
		return "series";
	}
	
	public void load(Element xmlElement) {
		setCurrentBranch(xmlElement.getAttribute("branch"));
		
		Element branchElement = (Element) xmlElement.getElementsByTagName("branchFrom").item(0);
		if(branchElement.hasAttribute("branch") && branchElement.hasAttribute("id")) {
			branchFromIdentifier = GeneticFactoryInstance.get().createInvalidIdentifier();
			branchFromIdentifier.load(branchElement);
		}
		
		NodeList list = xmlElement.getElementsByTagName("generation");
		for(int i = 0; i < list.getLength(); i++) {
			GenerationImposter g = new GenerationImposter();
			g.load((Element) list.item(i));
			current++;
			prefetchIndex++;
			generations.add(g);
		}
		
		for(int i = 0; i < INITIAL_PREFETCH; i++)
			prefetch(current - i);
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		prune();
		
		xmlElement.setAttribute("branch", BranchIdentifier.getCurrentBranch().getName());
		
		Element branchElement = xmlDocument.createElement("branchFrom");
		if(branchFromIdentifier != null)
			branchFromIdentifier.store(branchElement, xmlDocument);
		xmlElement.appendChild(branchElement);
		
		for(int i = 0; i <= current; i++)
			client.utilities.XML.storeElement(generations.get(i), xmlElement, xmlDocument);
	}
	
	public void notifySaveSuccessful() {
		DatabaseInstance.get().clearServer();
		
		for(int i = 0; i <= current; i++)
			generations.get(i).notifySuccessfulSave();
	}
}
