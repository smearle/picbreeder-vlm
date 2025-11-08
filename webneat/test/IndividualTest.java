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

import java.awt.image.ImageObserver;

import client.evolution.*;

public class IndividualTest implements Individual {
	boolean selected = false;
	Genome genome = null;
	Phenotype phenotype = null;
	boolean rendered = false;
	
	public IndividualTest() {
		this(null);
	}
	
	public IndividualTest(Genome g) {
		genome = g;
		phenotype = GeneticFactoryInstance.get().createPhenotype();
	}

	public Genome getGenome() {
		return genome;
	}
	
	public int countPhenotypes() {
		return 1;
	}

	public Phenotype getPhenotype(int index) {
		if(phenotype == null)
			phenotype = GeneticFactoryInstance.get().createPhenotype();
		return phenotype;
	}

	public Phenotype getDominantPhenotype() {
		return getPhenotype(genome.getDominantPhenotype());
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
	
	public boolean hasGenome() {
		return genome != null;
	}
	
	public void setGenome(Genome g) {
		genome = g;
	}
	
	public boolean isRendered() {
		return rendered;
	}
	
	public void setRendered(boolean value) {
		rendered = value;
	}
	
	public void conserveMemory() {
		phenotype = null;
		rendered = false;
	}
	

	// blah to get rid of compiler crap
	public void notifyUpdated() {
		
	}
	
	public void notifyCompleted() {
		
	}
	
	public void setQuality(int quality) {
	}
	
	public int getQuality() {
		return 0;
	}
	
	public void setObserver(ImageObserver im) {
	}
}
