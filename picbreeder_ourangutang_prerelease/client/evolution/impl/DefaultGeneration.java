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

import client.*;
import client.evolution.Generation;
import client.evolution.Genome;
import client.evolution.Individual;
import client.evolution.GeneticFactoryInstance;

import java.util.*;
import org.w3c.dom.*;


class DefaultGeneration implements Generation {
	private final int size;
		
	private List <Individual> generation;
	private int number;
	
	/**
	 * Initializes the generation to the default topology provided
	 * <code>spawnGeneration</code> is true.  This should only
	 * be called when no parent information is known (at the root of
	 * evolution).  Otherwise, the generation is intended for loading.
	 * <p>
	 * The parameter table is searched to create the networks.  Essentially
	 * all inputs will be linked to all outputs with random weights.  This
	 * happens for each individual.
	 */
	public DefaultGeneration(boolean spawnGeneration) {
		size = ParameterTableInstance.get().getInteger("evolution", "population size");
		
		generation = new ArrayList <Individual> (size);
		
		if(spawnGeneration) {
			number = 0;
			for(int i = 0; i < size; i++)
				generation.add(GeneticFactoryInstance.get().createRootIndividual());
		}
		else {
			number = -1;
			for(int i = 0; i < size; i++)
				generation.add(GeneticFactoryInstance.get().createInvalidIndividual());
		}
		
	}
	
	public DefaultGeneration(Collection <Genome> parents, int number) {
		size = ParameterTableInstance.get().getInteger("evolution", "population size");
		
		generation = new ArrayList <Individual> (size);
		this.number = number;
		
		for(int i = 0; i < size; i++)
			generation.add(GeneticFactoryInstance.get().createIndividual(parents));
	}
	
	public Individual getRepresentative() {
		for(Individual ind : generation)
			if(ind.isSelected())
				return ind;
		
		return null;
	}
	
	public int getSize() {
		return size;
	}
	
	// inverts the "prune" function by generating the missing individuals
	// from the previous generation's population
	public void restore(Collection <Genome> parents) {
		for(Individual ind : this) {
			if(!ind.hasGenome())
				ind.setGenome(GeneticFactoryInstance.get().createGenome(parents));
			else
				restoreParents(ind, parents);
		}
	}
	
	// rebinds the parents to the individual
	// this is needed to correctly map the loaded identifiers
	// to the genomes in memory
	private void restoreParents(Individual ind, Collection <Genome> parents) {
		if(ind.getGenome().getParentIdentifiers() != null) {
			for(client.evolution.Identifier id : ind.getGenome().getParentIdentifiers())
				for(Genome g : parents)
					if(id.equals(g.getIdentifier()))
						ind.getGenome().addParent(g);
		}
	}

	public void setNumber(int generationNumber) {
		number = generationNumber;
	}
	
	public int getNumber() {
		return number;
	}
	
	public Individual getIndividual(int number) {
		return generation.get(number);
	}
	
	public Collection <Individual> getIndividuals() {
		return generation;
	}
	
	public Iterator <Individual> iterator() {
		return generation.iterator();
	}
	
	public String getElementName() {
		return "generation";
	}
	
	public void load(Element xmlElement) {
		number = Integer.parseInt(xmlElement.getAttribute("number"));
		
		NodeList list = xmlElement.getElementsByTagName("genome");
		for(int i = 0; i < list.getLength(); i++) {
			Genome g = GeneticFactoryInstance.get().createInvalidGenome();
			g.load((Element) list.item(i));
			generation.get(i).setGenome(g);
			generation.get(i).select();
		}
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		int size = 0;
		
		for(Individual x : generation)
			if(x.isSelected()) { // pruning
				client.utilities.XML.storeElement(x.getGenome(), xmlElement, xmlDocument);
				size++;
			}

		xmlElement.setAttribute("size", Integer.toString(size));
		xmlElement.setAttribute("number", Integer.toString(number));
	}
}
