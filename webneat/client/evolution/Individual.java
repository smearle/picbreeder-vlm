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

package client.evolution;

/**
 * An Individual represents a genome and any other information
 * needed during evolution. It is primarily intended as a storage
 * class.
 *
 * @author Nick Beato
 */

public interface Individual {
	/**
	 * Retrieves the genome of this individual.
	 * 
	 * @return This individual's genome
	 */
	
	public Genome getGenome();
	
	/**
	 * Queries the number of phenotypes this individual has.
	 * 
	 * @return The number of phenotypes
	 */
	public int countPhenotypes();
	
	/**
	 * Retrieves the phenotype of this individual.
	 * 
	 * @param index The phenotype index
	 * @return This individual's phenotype
	 */
	public Phenotype getPhenotype(int index);
	
	/**
	 * Retrieves the dominant phenotype of this individual.
	 * 
	 * @return This individual's dominant phenotype
	 */
	public Phenotype getDominantPhenotype();
	
	/**
	 * Determines if this individual is currently selected by the user for
	 * the next generation.
	 * 
	 * @return <code>true</code> if this individual is selected,
	 * <code>false</code> otherwise
	 */
	public boolean isSelected();
	
	/**
	 * Selects this individual as a parent for the next generation.
	 */
	public void select();
	
	/**
	 * Deselects this individual as a parent for the next generation.
	 * <p>
	 * Note, the word <i>unselect</i> implies that the individual is
	 * not selected in its current state.  The verb <i>deselect</i> means
	 * to remove from selection, which is intended.
	 */
	public void deselect();
	
	/**
	 * Indicates whether or not the phenotype is updated.
	 * 
	 * @return <code>true</code> if the phenotype is rendered, <code>false</code> otherwise.
	 */
	public boolean isRendered();
	
	/**
	 * Indicates whether or not the phenotype is updated.  This is forcefully
	 * set by the renderers when updating the gui.
	 * 
	 * @param value The value of the flag
	 */
	public void setRendered(boolean value);
	
	/**
	 * Indicates whether or not this individual has a genome. Only
	 * one thing will cause this situation to arise. The individual
	 * is loaded from an incomplete series.  In this case, the series
	 * will have to regenerate the genome from the previous generation.
	 * 
	 * @return <code>true</code> if a genome exists, <code>false</code> otherwise.
	 */
	public boolean hasGenome();
	
	/**
	 * Set this individual's genome.  This could be disastrous. Only invoke this
	 * when restoring an individual after it has been pruned.
	 * 
	 * @param genome The genome
	 */
	public void setGenome(Genome genome);
	
	/**
	 * Instructs the individual to free any resources that it can restore later.
	 * This is done to keep memory usage down.  In most cases, the only field that
	 * should be non-null after this call is the genome itself, since the phenotype,
	 * network, etc. can be recomputed from the genome.
	 */
	public void conserveMemory();

	/**
	 * Notifies the individual that its phenotypes are updated.
	 */
	public void notifyUpdated();
	
	/**
	 * Notifies the individual that its phenotypes are completed.
	 */
	public void notifyCompleted();	
	
	/**
	 * Sets the quality level of this individual's phenotypes.
	 * The quality should be non-negative. 
	 * A higher value indicates better quality.
	 * <p>
	 * While the client.renderers.algorithms are responsible for
	 * this value, it is usually proportional to the average number
	 * of samples per pixel.
	 * 
	 * @param value The quality
	 */
	public void setQuality(int value);
	
	/**
	 * Gets the quality level of this individual's phenotypes.
	 * The quality is a non-negative integer.
	 * A higher value indicates better quality.
	 * <p>
	 * While the client.renderers.algorithms are responsible for
	 * this value, it is usually proportional to the average number
	 * of samples per pixel.
	 * 
	 * @return The quality
	 */
	public int getQuality();
}
