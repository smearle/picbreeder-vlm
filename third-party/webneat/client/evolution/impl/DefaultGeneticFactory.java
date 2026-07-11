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

import client.*;
import client.evolution.*;
import client.evolution.generators.GeneratorScheme;


/**
 * Default implementation of the GeneticFactory.
 * <p>
 * Note: JT noticed that some methods are not thread-safe. If the
 * system starts to hang we need to fix that.
 * 
 * @author Nick Beato
 */

public class DefaultGeneticFactory implements GeneticFactory {
	private History history;
	private Genome initialGenome = null;
	private final GeneratorScheme generator;
	private final Generator updater;
	private long nextMarking = 0;
	private long nextIdentifier = 0;
	
	public DefaultGeneticFactory() {
		// best I could think of to invoke these in the package :(
		BranchIdentifier.reset();
		
		
		history = new History();
		
		generator = new GeneratorScheme();
		generator.addScheme("structure", new client.evolution.generators.AffinityGenerator("grey"));
		generator.addScheme("color", new client.evolution.generators.AffinityGenerator("color"));
		generator.addScheme("both", new client.evolution.generators.SuperSweetGenerator());
		
		updater = new client.evolution.generators.InkToHSB();
	}
	
	public void setScheme(String name) {
		generator.pickScheme(name);
	}

	public void setSliderCoeffecient(double coefficient) {
		client.evolution.generators.AbstractLinkWeightMutator.setInterpolationParameter(coefficient);
	}
	
	public Genome updateGenome(Genome genome) {
		Collection <Genome> parents = new LinkedList <Genome> ();
		parents.add(genome);
		return updater.generate(parents);
	}
	
	public Genome createInvalidGenome() {
		return new DefaultGenome();
	}
	
	public Genome copyGenome(Genome genome) {
		return new DefaultGenome(createIdentifier(), genome);
	}
	
	public Genome createGenome(Collection <Genome> parents) {
		if(parents == null || parents.size() == 0)
			return createRootGenome();
		else
			return generator.generate(parents);
	}

	public Series createInvalidSeries() {
		return new MemoryConstrainedSeries(false);
	}
	
	public Series createRootSeries() {
		return new MemoryConstrainedSeries(true);
	}
	
	public Series createBranchSeries(Genome branchFrom) {
		//return new MemoryConstrainedSeries(updateGenome(branchFrom));
		return new MemoryConstrainedSeries(branchFrom);
	}

	public Individual createInvalidIndividual() {
		return new test.ColorAndGreyIndividual();
	}
	
	public Individual createRootIndividual() {
		return createIndividual(createRootGenome());
	}
	
	private Genome createRootGenome() {
		if(initialGenome == null) {
			synchronized(this) {
				if(initialGenome == null)
					initialGenome = createInitialGenome();
			}
		}
		
		Genome g = copyGenome(initialGenome);
		g.randomize();
		return updateGenome(g);
	}
	
	private Genome createInitialGenome() {
		history.clear();
		
		String [] in = ParameterTableInstance.get().getSetAsArray("inputs");
		String [] out = ParameterTableInstance.get().getSetAsArray("outputs");
		int hidden = ParameterTableInstance.get().getInteger("evolution", "hidden nodes");
				
		return new DefaultGenome(createIdentifier(), in, out, hidden);
	}
	
	public Individual createIndividual(Genome genome) {
		return new test.ColorAndGreyIndividual(genome);
	}
	
	public Individual createIndividual(Collection <Genome> parents) {
		return new test.ColorAndGreyIndividual(createGenome(parents));
	}
	
	public Generation createRootGeneration() {
		history.clear();
		return new DefaultGeneration(true);
	}
	
	public Generation createInvalidGeneration() {
		history.clear();
		return new DefaultGeneration(false);
	}
	
	public Generation createGeneration(Collection <Genome> parents, int generationNumber) {
		history.clear();
		
		if(generationNumber == 0)
		{
			Generation g = new DefaultGeneration(parents, 0);
			for(Individual ind : g)
				ind.setGenome(updateGenome(ind.getGenome()));
				//((client.evolution.generators.Mutator)updater).mutate(ind.getGenome());
			return g;
		}
		else
			return new DefaultGeneration(parents, generationNumber);
	}

	public Link createLink(Node from, Node to) {
		Marking x = history.findLinkMarkingFromNodes(from, to);
		
		if(x == null) {
			x = createMarking();
			history.updateLinkMarkingFromNodes(from, to, x);
		}
		
		return new DefaultLink(x, from.getMarking(), to.getMarking());
	}
	
	
	public Link createInvalidLink() {
		return new DefaultLink();
	}

	public Link copyLink(Link link) {
		return new DefaultLink(link);
	}
	
	public Node createNode(Link link, String affinity) {
		Marking x = history.findNodeMarkingFromLink(link);
		
		if(x == null) {
			x = createMarking();
			history.updateNodeMarkingFromLink(link, x);
		}
		
		return new DefaultNode(x, affinity);
	}
	
	public Node createInvalidNode() {
		return new DefaultNode();
	}
	
	public Node createNode(String name, String type, String affinity) {
		return new DefaultNode(createMarking(), name, type, affinity);
	}
	
	public Node copyNode(Node node) {
		return new DefaultNode(node);
	}
	
	public Phenotype createPhenotype() {
		return new test.ImagePhenotype();
	}
	
	public Marking createMarking() {
		return new DefaultMarking(nextMarking++);
	}
	
	public Identifier createIdentifier() {
		return new DefaultIdentifier(nextIdentifier++);
	}
	
	public Marking createInvalidMarking() {
		return new DefaultMarking(-1);
	}

	public Identifier createInvalidIdentifier() {
		return new DefaultIdentifier(-1);
	}
	
	public void reserveMarking(Marking marking) {
		if(marking.usesCurrentBranch())
			nextMarking = Math.max(nextMarking, marking.getId() + 1);
	}
	
	public void reserveIdentifier(Identifier identifier) {
		if(identifier.usesCurrentBranch())
			nextIdentifier = Math.max(nextIdentifier, identifier.getId() + 1);
	}
}
