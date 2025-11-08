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
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;

import javax.imageio.ImageIO;

import client.evolution.*;

public class ColorAndGreyIndividual implements Individual {
	boolean selected = false;
	Genome genome = null;
	Phenotype[] phenotypes = null;
	boolean rendered = false;
	private int quality;
	private ImageObserver observer = null;
	
	public ColorAndGreyIndividual() {
		this(null);
	}
	
	public ColorAndGreyIndividual(Genome g) {
		quality = Integer.MAX_VALUE;
		genome = g;
		phenotypes = new Phenotype[2];
		phenotypes[0] = GeneticFactoryInstance.get().createPhenotype();
		phenotypes[1] = GeneticFactoryInstance.get().createPhenotype();
	}

	public ColorAndGreyIndividual(Genome g, Phenotype a, Phenotype b) {
		quality = Integer.MAX_VALUE;
		genome = g;
		phenotypes = new Phenotype[2];
		phenotypes[0] = a;
		phenotypes[1] = b;
	}
	
	public Genome getGenome() {
		return genome;
	}
	
	public int countPhenotypes() {
		return 2;
	}

	public Phenotype getPhenotype(int index) {
		if(phenotypes[index] == null)
			phenotypes[index] = GeneticFactoryInstance.get().createPhenotype();
		return phenotypes[index];
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
		// TODO verify usage
		genome = GeneticFactoryInstance.get().updateGenome(g);
	}
	
	public boolean isRendered() {
		return rendered;
	}
	
	public void setRendered(boolean value) {
		rendered = value;
	}
	
	public void conserveMemory() {
		phenotypes[0] = null;
		phenotypes[1] = null;
		rendered = false;
	}

	public void notifyUpdated() {
		if(observer != null)
			for(int i = 0; i < phenotypes.length; i++)
				observer.imageUpdate((java.awt.Image)phenotypes[i], ImageObserver.SOMEBITS, 0, 0, phenotypes[i].getWidth(), phenotypes[i].getHeight());
	}
	
	public void notifyCompleted() {
		if(observer != null)
			for(int i = 0; i < phenotypes.length; i++)
				observer.imageUpdate((java.awt.Image)phenotypes[i], ImageObserver.ALLBITS, 0, 0, phenotypes[i].getWidth(), phenotypes[i].getHeight());
	}
	
	public void setQuality(int quality) {
		this.quality = quality;
	}
	
	public int getQuality() {
		return quality;
	}
	
	public void setObserver(ImageObserver im) {
		observer = im;
	}
}
