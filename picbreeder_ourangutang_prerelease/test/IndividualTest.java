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

package test;

import client.evolution.*;
import client.cppn.*;

public class IndividualTest implements Individual {
	boolean selected = false;
	Genome genome = null;
	Phenotype phenotype = null;
	Network network = null;
	boolean rendered = false;
	
	public IndividualTest() {
		this(null);
	}
	
	public IndividualTest(Genome g) {
		genome = g;
		phenotype = GeneticFactoryInstance.get().createPhenotype();
		
		//if(g.isValid())
		//	network = CPPNFactoryInstance.get().createNetwork(genome);
		//else
			network = null;
	}

	public Genome getGenome() {
		return genome;
	}

	public Phenotype getPhenotype() {
		if(phenotype == null)
			phenotype = GeneticFactoryInstance.get().createPhenotype();
		return phenotype;
	}

	public boolean isSelected() {
		return selected;
	}

	public void select() {
		selected = true;
	}

	public void deselect() {
		selected = false;
	}
	
	public Network getNetwork() {
		if(network == null)
			network = CPPNFactoryInstance.get().createNetwork(genome);
		return network;
	}
	
	public boolean hasGenome() {
		return genome != null;
	}
	
	public void setGenome(Genome g) {
		genome = g;
		network = null;
	}
	
	public boolean isRendered() {
		return rendered;
	}
	
	public void setRendered(boolean value) {
		rendered = value;
	}
	
	public void conserveMemory() {
		phenotype = null;
		network = null;
		rendered = false;
	}
}
